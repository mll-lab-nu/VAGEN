"""Validation rollouts reach wandb as episodes, with their frames.

The unit that matters is the episode, not the model call: a trajectory is several calls,
and after a compaction several conversations. The mixin regroups them; when a loop
publishes no episode ids there is nothing to regroup, and verl's own table is used.
"""

from __future__ import annotations

import pytest

from vagen.training.trainer.mixin import VagenV0Mixin


class _Cfg(dict):
    """Config that answers both attribute and .get access, like OmegaConf."""

    def __getattr__(self, k):
        try:
            return self[k]
        except KeyError as e:
            raise AttributeError(k) from e


class _Recorder:
    """Stands in for EpisodeTableLogger: renders in-process, records what it would log."""

    def __init__(self):
        self.episode_calls = []

    def submit(self, rows, n, step, strategy="balanced", success_ratio=0.5):
        from vagen.utils.episode_log import episode_rows, select_episodes

        self.episode_calls.append(
            (["wandb"], select_episodes(episode_rows(rows), n, strategy, success_ratio), step)
        )


class _Base:
    """Stands in for verl's RayPPOTrainer."""

    def __init__(self):
        self.base_calls = []

    def _maybe_log_val_generations(self, inputs, outputs, scores, extras=None):
        self.base_calls.append((inputs, outputs, scores, (extras or {}).get('image_data')))


class _Trainer(VagenV0Mixin, _Base):
    def __init__(self, **trainer_cfg):
        super().__init__()
        cfg = {
            "log_val_generations": 2,
            "project_name": "p",
            "experiment_name": "e",
            "logger": ["wandb"],
            "replace_image_tokens_for_logging": False,
        }
        cfg.update(trainer_cfg)
        self.config = _Cfg(trainer=_Cfg(cfg))
        self.global_steps = 7
        self._vagen_val_logger = _Recorder()


def _episode_batch(n_episodes=3, turns_per=2):
    """Rows as validation produces them: one per model call, episodes interleaved."""
    inputs, outputs, scores, images, g, t, ti, c = [], [], [], [], [], [], [], []
    for turn in range(turns_per):
        for ep in range(n_episodes):
            inputs.append(f"prompt{ep}")
            outputs.append(f"ep{ep}-turn{turn}")
            scores.append(float(turn))
            images.append([])
            g.append(0)
            t.append(ep)
            ti.append(turn)
            c.append(f"conv{ep}")
    extras = {"group_idx": g, "traj_idx": t, "turn_idx": ti, "conversation_id": c,
              "traj_success": [1.0 if ep == 0 else 0.0 for _ in range(len(g))]}
    return inputs, outputs, scores, images, extras


def test_calls_are_regrouped_into_episodes():
    t = _Trainer(log_val_generations=10)
    i, o, s, im, ex = _episode_batch(n_episodes=3, turns_per=2)
    t._maybe_log_val_generations(i, o, s, extras={**ex, 'image_data': im})

    assert len(t._vagen_val_logger.episode_calls) == 1
    _, episodes, step = t._vagen_val_logger.episode_calls[0]
    assert step == 7
    assert len(episodes) == 3, "six calls should be three two-turn episodes"
    assert all(e["turns"] == 2 for e in episodes)


def test_both_turns_of_an_episode_are_in_its_cell():
    t = _Trainer(log_val_generations=10)
    i, o, s, im, ex = _episode_batch(n_episodes=1, turns_per=3)
    t._maybe_log_val_generations(i, o, s, extras={**ex, 'image_data': im})
    (_, episodes, _) = t._vagen_val_logger.episode_calls[0]
    html = episodes[0]["html"]
    for turn in range(3):
        assert f"ep0-turn{turn}" in html


def test_it_honours_the_requested_count():
    t = _Trainer(log_val_generations=2)
    i, o, s, im, ex = _episode_batch(n_episodes=5, turns_per=2)
    t._maybe_log_val_generations(i, o, s, extras={**ex, 'image_data': im})
    (_, episodes, _) = t._vagen_val_logger.episode_calls[0]
    assert len(episodes) == 2


def test_zero_means_off():
    """A table of full transcripts every step is expensive; the switch has to gate."""
    t = _Trainer(log_val_generations=0)
    i, o, s, im, ex = _episode_batch()
    t._maybe_log_val_generations(i, o, s, extras={**ex, 'image_data': im})
    assert t._vagen_val_logger.episode_calls == []
    assert t.base_calls == []


def test_without_episode_ids_it_falls_back_to_verls_table():
    """A text-only loop publishes no ids, and must not lose its table."""
    t = _Trainer()
    t._maybe_log_val_generations(["a"], ["b"], [1.0], extras={"image_data": [None]})
    assert t._vagen_val_logger.episode_calls == []
    assert len(t.base_calls) == 1


def test_upstream_collects_and_forwards_what_regrouping_needs():
    """The override is useless unless _validate actually passes these.

    Overriding a method upstream calls without the episode columns is the failure this
    guards: every test above still passes, and the dashboard shows the flat table forever.
    """
    import inspect

    from verl.trainer.ppo.ray_trainer import RayPPOTrainer

    from vagen.training.trainer.mixin import VagenV0Mixin

    for col in ("image_data", "group_idx", "traj_idx", "episode_id", "conversations"):
        assert col in VagenV0Mixin.val_log_columns, f"{col} no longer requested"
    # Not turn_idx or conversation_id: the merge folds an episode's rows into one, so an
    # episode has many of each and the merge drops them. Requesting them made the
    # diagnostic report 0/256 forever.
    for col in ("turn_idx", "conversation_id"):
        assert col not in VagenV0Mixin.val_log_columns, (
            f"{col} identifies a row, and after the merge a row is a whole episode"
        )

    src = inspect.getsource(RayPPOTrainer._validate)
    assert "self.val_log_columns" in src, "_validate no longer honours the column list"
    assert "extras=" in src, "_validate no longer forwards the columns"
    assert "sample_extras" in src and "reward_extra_infos_dict}" in src, (
        "_validate no longer forwards both the requested columns and the env's metrics"
    )

    sig = inspect.signature(RayPPOTrainer._maybe_log_val_generations)
    assert "extras" in sig.parameters


def test_the_agent_loop_publishes_the_columns_the_logger_regroups_on():
    """The other end of the same contract: the loop has to emit these per row."""
    import inspect

    from vagen.training.agent_loop import gym_loop

    src = inspect.getsource(gym_loop.GymLoop._outputs)
    for col in ("group_idx", "traj_idx", "turn_idx", "conversation_id"):
        assert f'"{col}"' in src, f"the loop stopped publishing {col}"


def test_the_environments_verdict_reaches_the_table():
    """success was forwarded by upstream and then dropped here, so the column existed
    and was always empty -- which reads as "nothing ever succeeded"."""
    t = _Trainer(log_val_generations=10)
    i, o, s_, im, ex = _episode_batch(n_episodes=2, turns_per=2)
    ex["traj_success"] = [0.0, 1.0, 0.0, 1.0]
    t._maybe_log_val_generations(i, o, s_, extras={**ex, 'image_data': im})
    (_, episodes, _) = t._vagen_val_logger.episode_calls[0]
    assert {e["success"] for e in episodes} == {0.0, 1.0}, "verdict lost between upstream and the table"


def test_a_column_of_nones_does_not_shadow_a_good_one():
    """`a or b` prefers a list of Nones over a usable list, because any non-empty list is
    truthy. That short-circuit sent every validation down the fallback path and logged
    verl's flat table for three whole runs, while the episode columns were right there."""
    t = _Trainer(log_val_generations=4)
    i, o, s, im, ex = _episode_batch(n_episodes=2, turns_per=2)
    ex["episode_id"] = [None] * len(o)      # absent, as after a merge that dropped it
    t._maybe_log_val_generations(i, o, s, extras={**ex, 'image_data': im})
    assert t._vagen_val_logger.episode_calls, "fell back despite group_idx being present"
    assert t.base_calls == []


def test_episode_id_is_preferred_when_present():
    t = _Trainer(log_val_generations=4)
    i, o, s, im, ex = _episode_batch(n_episodes=2, turns_per=2)
    ex["episode_id"] = [f"ep{j % 2}" for j in range(len(o))]
    ex["group_idx"] = [None] * len(o)
    t._maybe_log_val_generations(i, o, s, extras={**ex, 'image_data': im})
    assert t._vagen_val_logger.episode_calls, "episode_id alone was not enough"


def test_with_neither_it_still_falls_back():
    t = _Trainer(log_val_generations=4)
    i, o, s, im, ex = _episode_batch(n_episodes=2, turns_per=2)
    ex["episode_id"] = [None] * len(o)
    ex["group_idx"] = [None] * len(o)
    t._maybe_log_val_generations(i, o, s, extras={**ex, 'image_data': im})
    assert t._vagen_val_logger.episode_calls == []
    assert len(t.base_calls) == 1


def test_the_diagnostic_reports_on_columns_that_exist():
    """It exists to catch a dropped column. Naming one that was renamed makes it report
    a permanent false alarm, which is worse than not reporting."""
    from vagen.training.trainer.mixin import VagenV0Mixin
    from vagen.utils.episode_log import describe_columns

    line = describe_columns({}, 4)
    for key in line.replace("=", " ").split():
        if "/" in key:
            continue
        assert key in VagenV0Mixin.val_log_columns, (
            f"the diagnostic reports on {key!r}, which is not a column we request"
        )
