"""Validation samples reach wandb with the frame the model was looking at.

The logger this uses is the one already on main. It survived the deletion of the
vendored trainer while its only caller did not, so it sat importable and unreferenced --
which reads exactly like a working feature until someone checks the dashboard.
"""

from __future__ import annotations

import sys
import types

import numpy as np
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

    def log(self, loggers, samples, step):
        self.calls.append((loggers, samples, step))


class _Base:
    """Stands in for verl's RayPPOTrainer."""

    def __init__(self):
        self.base_calls = []

    def _maybe_log_val_generations(self, inputs, outputs, scores, images=None):
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


def _samples(n=3, with_images=True):
    return (
        [f"in{i}" for i in range(n)],
        [f"out{i}" for i in range(n)],
        [float(i) for i in range(n)],
        [f"img{i}" for i in range(n)] if with_images else [None] * n,
    )


def test_images_ride_along_as_a_fourth_column():
    t = _Trainer()
    i, o, s, im = _samples()
    t._maybe_log_val_generations(i, o, s, images=im)

    assert len(t._vagen_val_logger.calls) == 1
    loggers, samples, step = t._vagen_val_logger.calls[0]
    assert (loggers, step) == (["wandb"], 7)
    assert all(len(row) == 4 for row in samples), "image column dropped"
    # Every logged row keeps its own image, not another row's.
    for inp, out, score, img in samples:
        idx = inp.removeprefix("in")
        assert (out, score, img) == (f"out{idx}", float(idx), f"img{idx}")


def test_it_honours_the_requested_count():
    t = _Trainer(log_val_generations=2)
    t._maybe_log_val_generations(*_samples(5)[:3], images=_samples(5)[3])
    _, samples, _ = t._vagen_val_logger.calls[0]
    assert len(samples) == 2


def test_zero_means_off():
    """The switch has to actually gate: a table logged every step is expensive."""
    t = _Trainer(log_val_generations=0)
    i, o, s, im = _samples()
    t._maybe_log_val_generations(i, o, s, images=im)
    assert t._vagen_val_logger.calls == []
    assert t.base_calls == []


def test_a_text_only_task_still_gets_verls_table():
    """No images is the normal case for a text task; it must not lose its table."""
    t = _Trainer()
    i, o, s, im = _samples(with_images=False)
    t._maybe_log_val_generations(i, o, s, images=im)
    assert t._vagen_val_logger.calls == [], "sent an all-None image column to the table"
    assert len(t.base_calls) == 1, "did not fall back to verl's logger"


def test_upstream_collects_and_forwards_the_column():
    """The override is useless unless _validate actually passes images.

    Overriding a method upstream never calls with an images argument is the failure
    this guards: every test above would still pass, and the dashboard would show the
    three-column table forever.
    """
    import inspect

    from verl.trainer.ppo.ray_trainer import RayPPOTrainer

    src = inspect.getsource(RayPPOTrainer._validate)
    assert 'non_tensor_batch.get("image_data")' in src, "_validate no longer collects images"
    assert "images=sample_images" in src, "_validate no longer forwards the image column"

    sig = inspect.signature(RayPPOTrainer._maybe_log_val_generations)
    assert "images" in sig.parameters, "base signature dropped the images parameter"


def test_shuffle_is_stable_across_steps():
    """Two steps must show the same prompts, or the table is not a progression."""
    seen = []
    for step in (1, 2):
        t = _Trainer()
        t.global_steps = step
        i, o, s, im = _samples(6)
        t._maybe_log_val_generations(i, o, s, images=im)
        seen.append([row[0] for row in t._vagen_val_logger.calls[0][1]])
    assert seen[0] == seen[1]
