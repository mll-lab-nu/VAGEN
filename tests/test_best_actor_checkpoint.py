"""Keep the actor from the best-validating step -- and only that.

Two things must stay true or this becomes a liability rather than a convenience:
it must not write `latest_checkpointed_iteration.txt` (a resume would then rewind to
whichever step validated best, which is not where training was), and it must not save the
critic or optimiser (this copy exists to be used, not to resume from).

And three things must stay true or it is not there when it is wanted: verl's checkpoint
retention must not delete it, a resumed run must not replace it with something worse, and
a save that dies half-way must not leave it half old, half new.
"""

from __future__ import annotations

import inspect
import json
import logging
import os

import pytest

from vagen.training.trainer.mixin import VagenLogicMixin, VagenV0Mixin


class _Cfg(dict):
    def __getattr__(self, k):
        try:
            return self[k]
        except KeyError as e:
            raise AttributeError(k) from e

    def get(self, k, d=None):
        return dict.get(self, k, d)


class _WG:
    def __init__(self):
        self.calls = []

    def save_checkpoint(self, local_path, remote_path=None, global_step=None, **kw):
        self.calls.append({"path": local_path, "remote": remote_path, "step": global_step, **kw})
        os.makedirs(local_path, exist_ok=True)


def _trainer(tmp_path, save_best=True, best_actor=None, resume_mode=None):
    trainer = _Cfg(default_local_dir=str(tmp_path), save_best_actor=save_best)
    if best_actor is not None:
        trainer["best_actor"] = _Cfg(best_actor)
    if resume_mode is not None:
        trainer["resume_mode"] = resume_mode
    t = object.__new__(VagenV0Mixin)
    t.config = _Cfg(trainer=trainer)
    t.actor_rollout_wg = _WG()
    t.metrics = {}
    t.global_steps = 0
    return t


def _validate_at(t, step, reward, env="sokoban"):
    t.global_steps = step
    t.metrics = {f"val-core/{env}/reward/mean@1": reward, "some/other": 99.0}
    t._vagen_maybe_save_best_actor()


def test_it_saves_only_on_a_new_best(tmp_path):
    t = _trainer(tmp_path)
    for step, r in ((20, 0.5), (40, 0.4), (60, 0.7), (80, 0.7), (100, 0.9)):
        _validate_at(t, step, r)
    assert [c["step"] for c in t.actor_rollout_wg.calls] == [20, 60, 100], (
        "saved on a step that did not improve, or missed one that did"
    )


def test_equal_is_not_better(tmp_path):
    """★ Ties must not re-save. Otherwise a plateaued run rewrites the checkpoint every
    validation, which is a lot of IO for no new information."""
    t = _trainer(tmp_path)
    _validate_at(t, 20, 0.5)
    _validate_at(t, 40, 0.5)
    assert len(t.actor_rollout_wg.calls) == 1


def test_it_does_nothing_on_a_step_that_did_not_validate(tmp_path):
    """`_fit_validate` runs every step and only sometimes validates, so an absent metric
    means 'no information', not 'scored zero'."""
    t = _trainer(tmp_path)
    _validate_at(t, 20, 0.5)
    t.global_steps, t.metrics = 21, {"actor/entropy": 0.3}
    t._vagen_maybe_save_best_actor()
    assert len(t.actor_rollout_wg.calls) == 1


def test_a_step_before_any_validation_saves_nothing(tmp_path):
    """★ The order that matters. Treating a missing metric as 0.0 is invisible once a
    real score has been seen -- 0.0 never beats it -- but on the *first* steps it saves an
    untrained actor and sets the bar to zero. Reading absence as no-information is the
    only reading that is right in both orders."""
    t = _trainer(tmp_path)
    for step in (1, 2, 3):
        t.global_steps, t.metrics = step, {"actor/entropy": 0.5}
        t._vagen_maybe_save_best_actor()
    assert t.actor_rollout_wg.calls == [], "saved before validation ever ran"
    assert getattr(t, "_vagen_best_val", None) is None, "a non-validating step set the bar"


def test_it_saves_the_actor_only_and_in_its_own_directory(tmp_path):
    """★ The load-bearing one. The critic and optimiser belong to the resume checkpoint,
    not to this. Saved beside `best_actor/` and renamed into it, never saved there
    directly: see test_the_kept_actor_survives_periodic_rotation."""
    t = _trainer(tmp_path)
    _validate_at(t, 20, 0.5)
    call = t.actor_rollout_wg.calls[0]
    assert call["path"] == os.path.join(str(tmp_path), "best_actor.tmp")
    assert (tmp_path / "best_actor").is_dir() and not (tmp_path / "best_actor.tmp").exists()
    assert not hasattr(t, "critic_wg") or not getattr(t, "_saved_critic", False)


def test_it_never_writes_latest_checkpointed_iteration(tmp_path):
    """★ If it did, a resume would rewind training to whichever step validated best."""
    t = _trainer(tmp_path)
    for step, r in ((20, 0.5), (40, 0.9)):
        _validate_at(t, step, r)
    assert not (tmp_path / "latest_checkpointed_iteration.txt").exists()
    src = inspect.getsource(VagenV0Mixin._vagen_maybe_save_best_actor)
    assert "latest_checkpointed_iteration" not in src.split('"""')[2]


def test_it_records_which_step_won_and_on_what(tmp_path):
    t = _trainer(tmp_path)
    _validate_at(t, 60, 0.77)
    meta = json.load(open(tmp_path / "best_actor" / "best.json"))
    assert meta == {"global_step": 60, "metric": "val-core/*/reward/mean@1", "mode": "max",
                    "score": pytest.approx(0.77)}


def test_several_environments_are_averaged(tmp_path):
    """A two-env run must not select on whichever key sorts first."""
    t = _trainer(tmp_path)
    t.global_steps = 20
    t.metrics = {"val-core/sokoban/reward/mean@1": 0.2, "val-core/frozenlake/reward/mean@1": 0.8}
    assert t._vagen_best_val_score() == pytest.approx(0.5)


def test_the_switch_turns_it_off(tmp_path):
    t = _trainer(tmp_path, save_best=False)
    _validate_at(t, 20, 0.9)
    assert t.actor_rollout_wg.calls == []


def test_a_failed_save_does_not_end_the_run(tmp_path):
    """Bookkeeping must not take down training that is going fine."""
    t = _trainer(tmp_path)

    def boom(*a, **k):
        raise RuntimeError("disk full")

    t.actor_rollout_wg.save_checkpoint = boom
    _validate_at(t, 20, 0.9)   # must not raise


def test_the_hook_is_actually_called_after_validation():
    """A guard nothing calls is not a guard."""
    assert "_vagen_maybe_save_best_actor()" in inspect.getsource(VagenV0Mixin._fit_validate)


# ------------------------------------------------------------------ retention
class _RotatingWG(_WG):
    """An actor worker group whose rank-0 checkpoint manager is verl's own: the same
    retention code, called in the order `FSDPCheckpointManager.save_checkpoint` calls it."""

    def __init__(self):
        super().__init__()
        from verl.utils.checkpoint.checkpoint_manager import BaseCheckpointManager

        self.manager = object.__new__(BaseCheckpointManager)  # __init__ wants torch.distributed
        self.manager.previous_saved_paths = []

    def save_checkpoint(self, local_path, remote_path=None, global_step=None, max_ckpt_to_keep=None, **kw):
        self.manager.ensure_checkpoint_capacity(max_ckpt_to_keep)
        super().save_checkpoint(local_path, remote_path, global_step, max_ckpt_to_keep=max_ckpt_to_keep, **kw)
        with open(os.path.join(local_path, "weights"), "w") as fh:
            fh.write(str(global_step))
        self.manager.register_checkpoint(local_path, max_ckpt_to_keep)


def _periodic_save(t, keep):
    """The actor half of verl's `_save_checkpoint`."""
    path = os.path.join(t.config.trainer.default_local_dir, f"global_step_{t.global_steps}", "actor")
    t.actor_rollout_wg.save_checkpoint(path, None, t.global_steps, max_ckpt_to_keep=keep)


@pytest.mark.parametrize("rewards,winner", [((0.9, 0.5, 0.4, 0.3), 10), ((0.3, 0.4, 0.5, 0.6), 40)])
@pytest.mark.parametrize("order", ["v0_validate_then_save", "v1_save_then_validate"])
@pytest.mark.parametrize("keep", [1, 2])
def test_the_kept_actor_survives_periodic_rotation(tmp_path, keep, order, rewards, winner):
    """★ The reason it is renamed into place. verl's checkpoint manager registers every path
    it saves for `max_actor_ckpt_to_keep` retention; saved at `best_actor/`, the copy was
    deleted by the next periodic save -- under V0, whose validation precedes the save, by
    the save of the very step that validated."""
    t = _trainer(tmp_path)
    t.actor_rollout_wg = _RotatingWG()
    steps = (10, 20, 30, 40)
    for step, reward in zip(steps, rewards):
        t.global_steps = step
        if order == "v1_save_then_validate":
            _periodic_save(t, keep)
        _validate_at(t, step, reward)
        if order == "v0_validate_then_save":
            _periodic_save(t, keep)
    best = tmp_path / "best_actor"
    assert (best / "weights").read_text() == str(winner), "the kept actor is gone or stale"
    assert json.loads((best / "best.json").read_text())["global_step"] == winner
    assert not (tmp_path / "best_actor.tmp").exists() and not (tmp_path / "best_actor.old").exists()
    # ... and retention still does its own job on the periodic checkpoints: the newest is
    # there and no more than `keep` are. (Not always exactly `keep`: verl's rank-0 list
    # still counts the staging path until it ages out, as it counted `best_actor/` before.)
    kept = sorted(int(p.parent.name.split("_")[-1]) for p in tmp_path.glob("global_step_*/actor"))
    assert kept[-1] == steps[-1] and len(kept) <= keep
    if keep == 1:
        assert kept == [steps[-1]]


# ------------------------------------------------------------------ what "best" means
def _validate_metrics(t, step, **metrics):
    t.global_steps = step
    t.metrics = {k.replace("__", "/"): v for k, v in metrics.items()}
    t._vagen_maybe_save_best_actor()


def test_it_selects_on_the_configured_metric(tmp_path):
    """reward = success + 0.1 x format: step 20 solves more and formats worse. On reward
    (the default) step 10 wins; on success, step 20."""
    rows = ((10, 0.50, 1.0), (20, 0.55, 0.3))
    on_reward = _trainer(tmp_path / "reward")
    on_success = _trainer(tmp_path / "success",
                          best_actor={"metric": "val-aux/sokoban/traj_success/mean@1"})
    for t in (on_reward, on_success):
        for step, success, fmt in rows:
            t.global_steps = step
            t.metrics = {"val-core/sokoban/reward/mean@1": success + 0.1 * fmt,
                         "val-aux/sokoban/traj_success/mean@1": success}
            t._vagen_maybe_save_best_actor()
    assert [c["step"] for c in on_reward.actor_rollout_wg.calls] == [10]
    assert [c["step"] for c in on_success.actor_rollout_wg.calls] == [10, 20]
    meta = json.load(open(tmp_path / "success" / "best_actor" / "best.json"))
    assert meta == {"global_step": 20, "metric": "val-aux/sokoban/traj_success/mean@1",
                    "mode": "max", "score": pytest.approx(0.55)}


def test_min_mode_keeps_the_lowest(tmp_path):
    t = _trainer(tmp_path, best_actor={"metric": "val-aux/sokoban/num_turns/mean@1", "mode": "min"})
    for step, turns in ((10, 5.0), (20, 6.0), (30, 4.0), (40, 4.0)):
        _validate_metrics(t, step, **{"val-aux__sokoban__num_turns__mean@1": turns})
    assert [c["step"] for c in t.actor_rollout_wg.calls] == [10, 30], "min mode, ties not better"


def test_a_pattern_averages_its_matches(tmp_path):
    t = _trainer(tmp_path, best_actor={"metric": "val-aux/*/traj_success/mean@1"})
    t.metrics = {"val-aux/sokoban/traj_success/mean@1": 0.2,
                 "val-aux/frozenlake/traj_success/mean@1": 0.6,
                 "val-aux/sokoban/traj_success/best@2/mean": 1.0}
    assert t._vagen_best_val_score() == pytest.approx(0.4)


def test_an_unknown_mode_is_refused_before_training(tmp_path):
    t = _trainer(tmp_path, best_actor={"mode": "maximum"})
    with pytest.raises(ValueError, match="trainer.best_actor.mode"):
        t._vagen_best_actor_rule()
    assert "_vagen_best_actor_rule()" in inspect.getsource(VagenLogicMixin._vagen_init)


def test_a_metric_that_matches_nothing_says_so_once(tmp_path, caplog):
    """A typo must not mean silently never keeping a best actor."""
    t = _trainer(tmp_path, best_actor={"metric": "val-aux/sokoban/traj_sucess/mean@1"})
    with caplog.at_level(logging.WARNING, logger="vagen.training.trainer.mixin"):
        _validate_at(t, 20, 0.5)
        _validate_at(t, 40, 0.6)
    assert t.actor_rollout_wg.calls == []
    assert sum("matches no validation metric" in r.getMessage() for r in caplog.records) == 1


def test_a_non_finite_score_is_no_information(tmp_path):
    """NaN compares false both ways: taken as a score it would win once and then never be
    beaten."""
    t = _trainer(tmp_path)
    for step, r in ((20, float("nan")), (40, 0.5), (60, float("nan")), (80, 0.6)):
        _validate_at(t, step, r)
    assert [c["step"] for c in t.actor_rollout_wg.calls] == [40, 80]


# ------------------------------------------------------------------ resume
def test_a_resumed_run_keeps_the_better_actor_it_already_has(tmp_path):
    """★ A requeued job is a new process. Starting its bar from nothing let the first
    validation after the resume, however bad, replace the best actor of the attempt before."""
    _validate_at(_trainer(tmp_path), 20, 0.7)
    resumed = _trainer(tmp_path)  # same directory; verl's default resume_mode is auto
    _validate_at(resumed, 40, 0.3)
    assert resumed.actor_rollout_wg.calls == [], "a worse step replaced the kept actor"
    assert json.load(open(tmp_path / "best_actor" / "best.json"))["global_step"] == 20
    _validate_at(resumed, 60, 0.8)
    assert [c["step"] for c in resumed.actor_rollout_wg.calls] == [60]


def test_a_fresh_run_does_not_inherit_the_bar(tmp_path):
    _validate_at(_trainer(tmp_path), 20, 0.7)
    fresh = _trainer(tmp_path, resume_mode="disable")
    _validate_at(fresh, 10, 0.3)
    assert [c["step"] for c in fresh.actor_rollout_wg.calls] == [10]


def test_a_record_kept_under_another_rule_is_not_compared(tmp_path):
    """A reward is not a success rate: a changed metric starts a new bar."""
    _validate_at(_trainer(tmp_path), 20, 0.7)
    t = _trainer(tmp_path, best_actor={"metric": "val-aux/sokoban/traj_success/mean@1"})
    _validate_metrics(t, 40, **{"val-aux__sokoban__traj_success__mean@1": 0.2})
    assert [c["step"] for c in t.actor_rollout_wg.calls] == [40]


def test_a_record_from_before_the_rule_was_configurable_still_counts(tmp_path):
    (tmp_path / "best_actor").mkdir()
    (tmp_path / "best_actor" / "best.json").write_text(
        json.dumps({"global_step": 20, "val_core_reward": 0.7}))
    t = _trainer(tmp_path)
    _validate_at(t, 40, 0.5)
    assert t.actor_rollout_wg.calls == []


# ------------------------------------------------------------------ atomicity
def test_a_save_that_dies_half_way_leaves_the_previous_best_whole(tmp_path):
    t = _trainer(tmp_path)
    t.actor_rollout_wg = _RotatingWG()
    _validate_at(t, 20, 0.5)
    survivor = t.actor_rollout_wg.save_checkpoint

    def dies(local_path, *args, **kwargs):
        os.makedirs(local_path, exist_ok=True)
        with open(os.path.join(local_path, "weights"), "w") as fh:
            fh.write("partial")
        raise RuntimeError("node lost")

    t.actor_rollout_wg.save_checkpoint = dies
    _validate_at(t, 40, 0.9)  # must not raise
    best = tmp_path / "best_actor"
    assert (best / "weights").read_text() == "20"
    assert json.load(open(best / "best.json"))["global_step"] == 20
    assert not (tmp_path / "best_actor.tmp").exists()
    # The bar still describes what is on disk, so 0.6 -- worse than the lost 0.9 -- is kept.
    t.actor_rollout_wg.save_checkpoint = survivor
    _validate_at(t, 60, 0.6)
    assert (best / "weights").read_text() == "60"


def test_a_swap_cut_short_between_its_renames_is_undone(tmp_path):
    _validate_at(_trainer(tmp_path), 20, 0.7)
    # Died after moving the kept copy aside, with the new one still unfinished.
    os.rename(tmp_path / "best_actor", tmp_path / "best_actor.old")
    (tmp_path / "best_actor.tmp").mkdir()
    resumed = _trainer(tmp_path)
    _validate_at(resumed, 40, 0.3)
    assert resumed.actor_rollout_wg.calls == [], "the restored record did not set the bar"
    assert json.load(open(tmp_path / "best_actor" / "best.json"))["global_step"] == 20
    assert not (tmp_path / "best_actor.old").exists() and not (tmp_path / "best_actor.tmp").exists()


def test_tidying_up_never_raises_and_never_drops_the_only_copy(tmp_path, monkeypatch):
    """It runs on the validation path; a filesystem error there must cost a warning, not
    the run, and must not delete the one copy of the previous best."""
    import vagen.training.trainer.mixin as mixin

    _validate_at(_trainer(tmp_path), 20, 0.7)
    os.rename(tmp_path / "best_actor", tmp_path / "best_actor.old")

    def refuse(src, dst):
        raise OSError("read-only file system")

    monkeypatch.setattr(mixin.os, "replace", refuse)
    resumed = _trainer(tmp_path)
    _validate_at(resumed, 40, 0.3)  # must not raise
    assert (tmp_path / "best_actor.old" / "best.json").is_file()
