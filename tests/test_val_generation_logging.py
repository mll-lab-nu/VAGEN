"""Validation rollouts reach wandb as episodes, with their frames.

The unit that matters is the episode, not the model call: a trajectory is several calls,
and after a compaction several conversations. The mixin regroups them; when a loop
publishes no episode ids there is nothing to regroup, and verl's own table is used.
"""

from __future__ import annotations

import pytest

from vagen.trainer.mixin import VagenV0Mixin


class _Cfg(dict):
    """Config that answers both attribute and .get access, like OmegaConf."""

    def __getattr__(self, k):
        try:
            return self[k]
        except KeyError as e:
            raise AttributeError(k) from e


class _Recorder:
    def __init__(self):
        self.calls = []
        self.episode_calls = []

    def log(self, loggers, samples, step):
        self.calls.append((loggers, samples, step))

    def log_episodes(self, loggers, episodes, step):
        self.episode_calls.append((loggers, episodes, step))


class _Base:
    """Stands in for verl's RayPPOTrainer."""

    def __init__(self):
        self.base_calls = []

    def _maybe_log_val_generations(self, inputs, outputs, scores, images=None, extras=None):
        self.base_calls.append((inputs, outputs, scores, images))


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
    extras = {"group_idx": g, "traj_idx": t, "turn_idx": ti, "conversation_id": c}
    return inputs, outputs, scores, images, extras


def test_calls_are_regrouped_into_episodes():
    t = _Trainer(log_val_generations=10)
    i, o, s, im, ex = _episode_batch(n_episodes=3, turns_per=2)
    t._maybe_log_val_generations(i, o, s, images=im, extras=ex)

    assert len(t._vagen_val_logger.episode_calls) == 1
    _, episodes, step = t._vagen_val_logger.episode_calls[0]
    assert step == 7
    assert len(episodes) == 3, "six calls should be three two-turn episodes"
    assert all(e["turns"] == 2 for e in episodes)


def test_both_turns_of_an_episode_are_in_its_cell():
    t = _Trainer(log_val_generations=10)
    i, o, s, im, ex = _episode_batch(n_episodes=1, turns_per=3)
    t._maybe_log_val_generations(i, o, s, images=im, extras=ex)
    (_, episodes, _) = t._vagen_val_logger.episode_calls[0]
    html = episodes[0]["html"]
    for turn in range(3):
        assert f"ep0-turn{turn}" in html


def test_it_honours_the_requested_count():
    t = _Trainer(log_val_generations=2)
    i, o, s, im, ex = _episode_batch(n_episodes=5, turns_per=2)
    t._maybe_log_val_generations(i, o, s, images=im, extras=ex)
    (_, episodes, _) = t._vagen_val_logger.episode_calls[0]
    assert len(episodes) == 2


def test_zero_means_off():
    """A table of full transcripts every step is expensive; the switch has to gate."""
    t = _Trainer(log_val_generations=0)
    i, o, s, im, ex = _episode_batch()
    t._maybe_log_val_generations(i, o, s, images=im, extras=ex)
    assert t._vagen_val_logger.episode_calls == []
    assert t._vagen_val_logger.calls == []
    assert t.base_calls == []


def test_without_episode_ids_it_falls_back_to_verls_table():
    """A text-only loop publishes no ids, and must not lose its table."""
    t = _Trainer()
    t._maybe_log_val_generations(["a"], ["b"], [1.0], images=[None], extras={})
    assert t._vagen_val_logger.episode_calls == []
    assert len(t.base_calls) == 1


def test_upstream_collects_and_forwards_what_regrouping_needs():
    """The override is useless unless _validate actually passes these.

    Overriding a method upstream calls without the episode columns is the failure this
    guards: every test above still passes, and the dashboard shows the flat table forever.
    """
    import inspect

    from verl.trainer.ppo.ray_trainer import EXTRA_LOG_COLUMNS, RayPPOTrainer

    for col in ("group_idx", "traj_idx", "turn_idx", "conversation_id"):
        assert col in EXTRA_LOG_COLUMNS, f"{col} no longer collected"

    src = inspect.getsource(RayPPOTrainer._validate)
    assert 'non_tensor_batch.get("image_data")' in src, "_validate no longer collects frames"
    assert "extras=sample_extras" in src, "_validate no longer forwards the episode columns"

    sig = inspect.signature(RayPPOTrainer._maybe_log_val_generations)
    assert "images" in sig.parameters and "extras" in sig.parameters


def test_the_agent_loop_publishes_the_columns_the_logger_regroups_on():
    """The other end of the same contract: the loop has to emit these per row."""
    import inspect

    from vagen.agent_loop import gym_loop

    src = inspect.getsource(gym_loop.GymLoop._outputs)
    for col in ("group_idx", "traj_idx", "turn_idx", "conversation_id"):
        assert f'"{col}"' in src, f"the loop stopped publishing {col}"
