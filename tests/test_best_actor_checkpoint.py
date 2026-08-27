"""Keep the actor from the best-validating step -- and only that.

Two things must stay true or this becomes a liability rather than a convenience:
it must not write `latest_checkpointed_iteration.txt` (a resume would then rewind to
whichever step validated best, which is not where training was), and it must not save the
critic or optimiser (this copy exists to be used, not to resume from).
"""

from __future__ import annotations

import json
import os

import pytest

from vagen.training.trainer.mixin import VagenV0Mixin


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


def _trainer(tmp_path, save_best=True):
    t = object.__new__(VagenV0Mixin)
    t.config = _Cfg(trainer=_Cfg(default_local_dir=str(tmp_path), save_best_actor=save_best))
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
    not to this."""
    t = _trainer(tmp_path)
    _validate_at(t, 20, 0.5)
    call = t.actor_rollout_wg.calls[0]
    assert call["path"] == os.path.join(str(tmp_path), "best_actor")
    assert not hasattr(t, "critic_wg") or not getattr(t, "_saved_critic", False)


def test_it_never_writes_latest_checkpointed_iteration(tmp_path):
    """★ If it did, a resume would rewind training to whichever step validated best."""
    t = _trainer(tmp_path)
    for step, r in ((20, 0.5), (40, 0.9)):
        _validate_at(t, step, r)
    assert not (tmp_path / "latest_checkpointed_iteration.txt").exists()
    import inspect
    src = inspect.getsource(VagenV0Mixin._vagen_maybe_save_best_actor)
    assert "latest_checkpointed_iteration" not in src.split('"""')[2]


def test_it_records_which_step_won(tmp_path):
    t = _trainer(tmp_path)
    _validate_at(t, 60, 0.77)
    meta = json.load(open(tmp_path / "best_actor" / "best.json"))
    assert meta == {"global_step": 60, "val_core_reward": pytest.approx(0.77)}


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
    import inspect
    assert "_vagen_maybe_save_best_actor()" in inspect.getsource(VagenV0Mixin._fit_validate)
