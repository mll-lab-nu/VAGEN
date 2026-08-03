"""Unit tests for vagen/trainer/mixin.py.

The mixin is exercised against a fake base trainer rather than a real
``SeparateRayPPOTrainer``: the point of the two-layer split is that the logic does not
depend on a cluster, and these must stay runnable on CPU in under a second.

What is deliberately NOT covered here (needs a real run, tracked in Phase 1):
ordering against the actual ``fit_step``, DataProto/tensor plumbing through Ray, and
the HF Hub client itself.
"""

import os
import types
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from vagen.trainer.logic import IGNORE_RETURN
from vagen.trainer.mixin import VagenV0Mixin


# --------------------------------------------------------------------- fakes


class _FakeBase:
    """Stands in for SeparateRayPPOTrainer. Records which hooks super() reached."""

    def __init__(self):
        self.calls = []

    def _fit_compute_advantage(self, batch):
        self.calls.append("super_advantage")
        return batch

    def _fit_save_checkpoint(self, force=False):
        self.calls.append(f"super_save(force={force})")

    def _save_checkpoint(self):
        self.calls.append("forced_save")

    def _balance_batch(self, batch, metrics, logging_prefix=None):
        self.calls.append(f"balance:{logging_prefix}")


class _Trainer(VagenV0Mixin, _FakeBase):
    def __init__(self, cfg, hf=None):
        super().__init__()
        self.config = cfg
        self.metrics = {}
        self.timing_raw = {}
        self.global_steps = 7
        self.actor_rollout_wg = types.SimpleNamespace(world_size=4)
        self._hf_upload_manager = hf or MagicMock(**{"should_upload.return_value": False})


def _cfg(adv="gae", filter_enable=False, balance=True, local_dir="/nonexistent"):
    return OmegaConf.create(
        {
            "algorithm": {"adv_estimator": adv},
            "trainer": {"balance_batch": balance, "default_local_dir": local_dir},
            "filter": {"enable": filter_enable, "name": "noop", "filter_kwargs": {}},
        }
    )


def _batch(n=4, width=3):
    """A real DataProto -- `pad_dataproto_to_divisor` asserts on the type, and the
    filter path is only meaningful if that call actually runs."""
    from verl import DataProto

    returns = torch.full((n, width), IGNORE_RETURN)
    returns[:, 0] = 1.0  # one supervised anchor per row, as turn-level GAE emits
    return DataProto.from_single_dict(
        {
            "returns": returns,
            "response_mask": torch.ones(n, width, dtype=torch.long),
            "attention_mask": torch.ones(n, width, dtype=torch.long),
        }
    )


# ------------------------------------------------------------- value_mask


def test_value_mask_written_for_sentinel_estimator():
    """★ The regression that mattered: `no_concat_gae` must get a value_mask."""
    t = _Trainer(_cfg(adv="no_concat_gae"))
    out = t._fit_compute_advantage(_batch())

    assert "value_mask" in out.batch
    assert out.batch["value_mask"].tolist() == [[1, 0, 0]] * 4


def test_value_mask_not_written_for_plain_gae():
    t = _Trainer(_cfg(adv="gae"))
    assert "value_mask" not in t._fit_compute_advantage(_batch()).batch


def test_super_advantage_runs_first():
    """Our work reads `returns`, so verl must have computed them already."""
    t = _Trainer(_cfg(adv="no_concat_gae"))
    t._fit_compute_advantage(_batch())
    assert t.calls[0] == "super_advantage"


# ----------------------------------------------------------------- metrics


def test_custom_metrics_land_on_self_metrics(monkeypatch):
    import vagen.trainer.mixin as m

    monkeypatch.setattr(m, "METRIC_REGISTRY", {"foo": lambda b: 1.5})
    t = _Trainer(_cfg())
    t._fit_compute_advantage(_batch())

    assert t.metrics["custom_metrics/train/foo"] == 1.5


def test_broken_metric_does_not_abort_the_step(monkeypatch):
    import vagen.trainer.mixin as m

    def boom(_):
        raise RuntimeError("nope")

    monkeypatch.setattr(m, "METRIC_REGISTRY", {"bad": boom})
    t = _Trainer(_cfg())
    t._fit_compute_advantage(_batch())  # must not raise

    assert t.metrics["custom_metrics/train/_failed/bad"] == 1.0


# ------------------------------------------------------------------ filter


def test_filter_disabled_is_a_no_op():
    t = _Trainer(_cfg(filter_enable=False))
    t._fit_compute_advantage(_batch())
    assert not any(c.startswith("balance") for c in t.calls)


def test_filter_runs_and_rebalances(monkeypatch):
    """★ Placement guard. The filter must shrink the batch *before* the critic/actor
    updates; `fit_step` calls _fit_compute_advantage immediately before them, so
    seeing the filter fire from this hook is what we want. (Hanging it off
    _fit_experimental would run it after the actor update -- a no-op.)"""
    import vagen.trainer.mixin as m

    seen = {}

    def filt(batch, metrics, **kw):
        seen["called"] = True
        metrics["filter/kept"] = 2
        return batch, metrics

    monkeypatch.setattr(m, "FILTER_REGISTRY", {"noop": filt})
    t = _Trainer(_cfg(filter_enable=True))
    t._fit_compute_advantage(_batch())

    assert seen.get("called")
    assert t.metrics["filter/kept"] == 2
    assert "balance:filtered_global_seqlen" in t.calls


def test_filter_skips_rebalance_when_balance_batch_off(monkeypatch):
    import vagen.trainer.mixin as m

    monkeypatch.setattr(m, "FILTER_REGISTRY", {"noop": lambda b, mt, **kw: (b, mt)})
    t = _Trainer(_cfg(filter_enable=True, balance=False))
    t._fit_compute_advantage(_batch())

    assert not any(c.startswith("balance") for c in t.calls)


def test_missing_filter_section_is_tolerated():
    """Configs predating the filter feature must not crash."""
    cfg = _cfg()
    del cfg["filter"]
    t = _Trainer(cfg)
    t._fit_compute_advantage(_batch())  # must not raise


# -------------------------------------------------------------- checkpoint


def test_no_upload_leaves_save_to_verl():
    hf = MagicMock(**{"should_upload.return_value": False})
    t = _Trainer(_cfg(), hf=hf)
    t._fit_save_checkpoint()

    assert t.calls == ["super_save(force=False)"]
    hf.flush.assert_not_called()
    hf.maybe_upload.assert_not_called()


def test_upload_forces_a_save_when_verl_did_not(tmp_path):
    """★ An upload-only step needs a checkpoint on disk to upload from."""
    hf = MagicMock(**{"should_upload.return_value": True})
    t = _Trainer(_cfg(local_dir=str(tmp_path)), hf=hf)
    t._fit_save_checkpoint()

    assert t.calls == ["super_save(force=False)", "forced_save"]
    hf.maybe_upload.assert_called_once_with(7)


def test_upload_does_not_double_save_when_verl_already_saved(tmp_path):
    hf = MagicMock(**{"should_upload.return_value": True})
    t = _Trainer(_cfg(local_dir=str(tmp_path)), hf=hf)
    os.makedirs(tmp_path / "global_step_7")  # pretend super() wrote it

    t._fit_save_checkpoint()

    assert t.calls == ["super_save(force=False)"], "must not save twice"
    hf.maybe_upload.assert_called_once_with(7)


def test_flush_happens_before_save():
    """max_*_ckpt_to_keep deletes old checkpoints on save; an in-flight upload reading
    one of them would race the delete."""
    hf = MagicMock(**{"should_upload.return_value": True})
    order = []
    hf.flush.side_effect = lambda: order.append("flush")

    class _T(_Trainer):
        def _save_checkpoint(self):
            order.append("save")
            super()._save_checkpoint()

    t = _T(_cfg(), hf=hf)
    t._fit_save_checkpoint()

    assert order[0] == "flush", f"flush must precede save, got {order}"


def test_save_checkpoint_forwards_extra_arguments():
    """★ FullyAsyncTrainer declares `_fit_save_checkpoint(self, force=False)` and calls
    it with force=True. A fixed no-arg override sits earlier in the MRO and turns that
    into a TypeError, so the mixin has to pass arguments through."""
    hf = MagicMock(**{"should_upload.return_value": False})
    t = _Trainer(_cfg(), hf=hf)

    t._fit_save_checkpoint(force=True)

    assert t.calls == ["super_save(force=True)"]


# ------------------------------------------------------------- concrete trainer
# Static checks only: instantiating needs a Ray cluster, and what can silently go
# wrong here is the class layout, not the runtime behaviour.


def test_mixin_overrides_win_the_mro():
    """★ With the bases reversed the mixin's _fit_* overrides never run and nothing
    raises -- value_mask just quietly stops being written, which is the same silent
    failure mode this whole path already produced once."""
    from verl.experimental.separation.ray_trainer import SeparateRayPPOTrainer

    from vagen.trainer.ppo_trainer import VagenPPOTrainer

    mro = VagenPPOTrainer.__mro__
    assert mro.index(VagenV0Mixin) < mro.index(SeparateRayPPOTrainer)
    assert VagenPPOTrainer._fit_compute_advantage is VagenV0Mixin._fit_compute_advantage
    assert VagenPPOTrainer._fit_save_checkpoint is VagenV0Mixin._fit_save_checkpoint


def test_super_from_the_mixin_reaches_the_trainer():
    """The overrides call super(); if the mixin were not co-operative in this MRO the
    base implementation would be skipped entirely."""
    from verl.experimental.separation.ray_trainer import SeparateRayPPOTrainer

    from vagen.trainer.ppo_trainer import VagenPPOTrainer

    after_mixin = VagenPPOTrainer.__mro__[VagenPPOTrainer.__mro__.index(VagenV0Mixin) + 1 :]
    assert SeparateRayPPOTrainer in after_mixin


def test_base_still_provides_the_hooks_we_override():
    """A rename upstream would leave our override sitting on a method nobody calls."""
    from verl.experimental.separation.ray_trainer import SeparateRayPPOTrainer

    for hook in ("_fit_compute_advantage", "_fit_save_checkpoint"):
        assert hook in vars(SeparateRayPPOTrainer), f"{hook} is gone from SeparateRayPPOTrainer"


def test_actor_rollout_placement_is_implemented():
    """★ SeparateRayPPOTrainer leaves _create_actor_rollout_classes abstract -- it is
    the one method a concrete subclass must supply. Inheriting the base's
    `raise NotImplementedError` only surfaces at init_workers, i.e. after a cluster has
    been spun up and models loaded."""
    from verl.experimental.separation.ray_trainer import SeparateRayPPOTrainer

    from vagen.trainer.ppo_trainer import VagenPPOTrainer

    assert "_create_actor_rollout_classes" in vars(SeparateRayPPOTrainer), (
        "base no longer declares the hook; re-check whether an override is still needed"
    )
    assert VagenPPOTrainer._create_actor_rollout_classes is not (
        SeparateRayPPOTrainer._create_actor_rollout_classes
    )


def test_base_init_models_expects_the_role_we_register():
    """_init_models indexes all_wg by str(Role.ActorRollout). Registering under
    ActorRolloutRef ('actor_rollout_ref') would only fail there, as a bare KeyError."""
    import inspect

    from verl.experimental.separation.ray_trainer import SeparateRayPPOTrainer
    from verl.trainer.ppo.ray_trainer import Role

    src = inspect.getsource(SeparateRayPPOTrainer._init_models)
    assert "Role.ActorRollout]" in src or "str(Role.ActorRollout)" in src
    assert str(Role.ActorRollout) != str(Role.ActorRolloutRef)


def test_placeholder_prompt_tensors_leave_the_batch():
    """★ verl 0.8 stopped popping input_ids/attention_mask/position_ids for generation.
    Left in place, they collide with the generated tensors at
    batch.union(gen_batch_output), which asserts shared keys are the same object."""
    import torch
    from verl import DataProto

    from vagen.trainer.ppo_trainer import VagenPPOTrainer

    batch = DataProto.from_single_dict(
        {
            "input_ids": torch.zeros(2, 1, dtype=torch.long),
            "attention_mask": torch.zeros(2, 1, dtype=torch.long),
            "position_ids": torch.zeros(2, 1, dtype=torch.long),
            "uid": np.array(["a", "b"], dtype=object),
            "env_name": np.array(["sokoban"] * 2, dtype=object),
        }
    )
    gen = VagenPPOTrainer._get_gen_batch(object(), batch)

    assert "input_ids" not in batch.batch, "placeholder survived; union will assert"
    assert "input_ids" in gen.batch
    # uid is a reward-model key: it stays behind so the batch can be realigned later,
    # and is copied onto the gen batch so the loop can score in flight.
    assert "uid" in batch.non_tensor_batch and "uid" in gen.non_tensor_batch
    assert "env_name" in gen.non_tensor_batch and "env_name" not in batch.non_tensor_batch


def test_get_gen_batch_tolerates_absent_placeholders():
    """A dataset that supplies real prompts must not trip the pop."""
    import torch
    from verl import DataProto

    from vagen.trainer.ppo_trainer import VagenPPOTrainer

    batch = DataProto.from_single_dict(
        {"input_ids": torch.zeros(2, 1, dtype=torch.long), "uid": np.array(["a", "b"], dtype=object)}
    )
    gen = VagenPPOTrainer._get_gen_batch(object(), batch)
    assert "input_ids" in gen.batch
