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

import vagen.custom_advantage  # noqa: F401  registers the estimators these lists read
from vagen.custom_advantage import CRITIC_ESTIMATORS, TRAJECTORY_ESTIMATORS
from vagen.trainer.logic import IGNORE_RETURN
from vagen.trainer.mixin import VagenV0Mixin

#: Read from the registry, never listed. A parametrize list that goes stale keeps
#: passing while quietly not covering whatever was added last.
SPANNING = sorted(TRAJECTORY_ESTIMATORS)
VALUE_BASED = sorted(CRITIC_ESTIMATORS)


# --------------------------------------------------------------------- fakes


class _FakeBase:
    """Stands in for SeparateRayPPOTrainer. Records which hooks super() reached."""

    def __init__(self):
        self.calls = []

    def _fit_compute_advantage(self, batch):
        self.calls.append("super_advantage")
        return batch

    def _fit_collect_metrics(self, batch):
        # The real base fills critic/score/* here, several hooks after advantage.
        self.calls.append("super_collect_metrics")

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
        # what _vagen_init would set up; tests bypass it since it builds a real uploader
        self._vagen_image_actors = {}
        self._vagen_image_futures = []


def _cfg(adv="gae", filter_enable=False, balance=True, local_dir="/nonexistent", harness="concat"):  # noqa: D103
    return OmegaConf.create(
        {
            "algorithm": {"adv_estimator": adv},
            "trainer": {
                "balance_batch": balance,
                "default_local_dir": local_dir,
                "harness": harness,
            },
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


# --------------------------------------------- estimator vs. context policy


@pytest.mark.parametrize("adv", ["gae", "grpo", "reinforce_plus_plus"])
@pytest.mark.parametrize("harness", ["no_concat", "compact"])
def test_row_local_estimator_is_refused_under_a_splitting_harness(adv, harness):
    """★ The pairing that fails silently. verl's estimators score one row and open each
    with nextvalues=0; under these policies a row is one turn, so turn t is never
    credited with anything that happened after it. Nothing downstream notices."""
    t = _Trainer(_cfg(adv=adv, harness=harness))
    with pytest.raises(ValueError, match="scores one row at a time"):
        t._vagen_check_estimator_spans_the_layout()


@pytest.mark.parametrize("adv", SPANNING)
@pytest.mark.parametrize("harness", ["concat", "no_concat", "compact"])
def test_trajectory_estimators_are_allowed_everywhere(adv, harness):
    """The other half of the claim: these stitch the rows back together, so every policy
    is fine. If one of them were rejected the guard would be over-broad, which is the
    failure mode a permissive test never catches."""
    import vagen.custom_advantage  # noqa: F401  registers the estimators

    _Trainer(_cfg(adv=adv, harness=harness))._vagen_check_estimator_spans_the_layout()


@pytest.mark.parametrize("adv", ["gae", "grpo"])
def test_row_local_estimator_is_fine_under_concat(adv):
    """Concat puts a whole episode in one row, so row-local *is* trajectory-level there.
    Rejecting it would break every existing script."""
    _Trainer(_cfg(adv=adv, harness="concat"))._vagen_check_estimator_spans_the_layout()


def test_the_check_runs_at_startup():
    """★ A guard nothing calls is not a guard. Pins the call into `_vagen_init` so
    deleting it there is caught here rather than by a wasted training run."""
    import inspect

    from vagen.trainer.mixin import VagenLogicMixin

    src = inspect.getsource(VagenLogicMixin._vagen_init)
    assert "_vagen_check_estimator_spans_the_layout()" in src


def test_the_guard_reads_the_registry_rather_than_a_local_list():
    """The set of safe estimators must come from the registry the estimators populate.
    A list kept in the trainer would drift the moment one is added."""
    import inspect

    from vagen.trainer.mixin import VagenLogicMixin

    src = inspect.getsource(VagenLogicMixin._vagen_check_estimator_spans_the_layout)
    assert "spans_rows" in src
    for name in SPANNING:
        assert name not in src, f"{name} is hard-coded in the trainer; read the registry"


# ------------------------------------------------------------- critic guard


@pytest.mark.parametrize("adv", VALUE_BASED)
def test_a_value_based_estimator_without_a_critic_is_refused(adv):
    """★ verl builds a critic from `critic.enable`, and when that is unset it falls back
    to `adv_estimator == "gae"` -- the literal string. Every estimator here fails that
    test, so `values` reads as zeros and GAE becomes a whitened discounted reward sum.
    The run starts, uses half the memory, trains faster, and says so only in a warning
    that reads "Disabled critic as algorithm.adv_estimator != gae"."""
    import vagen.custom_advantage  # noqa: F401  registers them

    t = _Trainer(_cfg(adv=adv))
    t.use_critic = False
    with pytest.raises(ValueError, match="no critic will be built"):
        t._vagen_check_estimator_has_its_critic()


@pytest.mark.parametrize("adv", VALUE_BASED)
def test_the_same_estimators_pass_with_a_critic(adv):
    import vagen.custom_advantage  # noqa: F401

    t = _Trainer(_cfg(adv=adv))
    t.use_critic = True
    t._vagen_check_estimator_has_its_critic()


@pytest.mark.parametrize("adv", ["trajectory_grpo", "grpo"])
def test_critic_free_estimators_are_not_required_to_have_one(adv):
    """Over-broad would be just as bad: GRPO needs no critic and must not be forced to
    pay for one."""
    import vagen.custom_advantage  # noqa: F401

    t = _Trainer(_cfg(adv=adv))
    t.use_critic = False
    t._vagen_check_estimator_has_its_critic()


def test_the_critic_check_runs_at_startup():
    """A guard nothing calls is not a guard."""
    import inspect

    from vagen.trainer.mixin import VagenLogicMixin

    assert "_vagen_check_estimator_has_its_critic()" in inspect.getsource(VagenLogicMixin._vagen_init)


def test_the_real_trainer_calls_vagen_init():
    """★ Both guards live in `_vagen_init`, and the two tests above only pin that they are
    *inside* it. Deleting the one line that calls it left the entire suite green -- the
    guards existed and never ran. This pins the seam itself."""
    import inspect

    from vagen.trainer.ppo_trainer import VagenPPOTrainer

    src = inspect.getsource(VagenPPOTrainer.__init__)
    assert "self._vagen_init()" in src, (
        "nothing calls _vagen_init, so every startup guard in it is dead"
    )
    # And after super(), which is what makes self.config and self.use_critic exist.
    assert src.index("super().__init__") < src.index("self._vagen_init()")


# ------------------------------------------------------------- value_mask


def test_value_mask_written_for_sentinel_estimator():
    """★ The regression that mattered: an estimator that anchors one return per turn
    must get a value_mask, or the critic regresses towards the -100 sentinel almost
    everywhere while its loss falls and nothing looks wrong."""
    t = _Trainer(_cfg(adv="turn_level_gae"))
    out = t._fit_compute_advantage(_batch())

    assert "value_mask" in out.batch
    assert out.batch["value_mask"].tolist() == [[1, 0, 0]] * 4


def test_value_mask_not_written_for_plain_gae():
    t = _Trainer(_cfg(adv="gae"))
    assert "value_mask" not in t._fit_compute_advantage(_batch()).batch


def test_super_advantage_runs_first():
    """Our work reads `returns`, so verl must have computed them already."""
    t = _Trainer(_cfg(adv="turn_level_gae"))
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


# --------------------------------------------------------------------- image dumps


def _img_cfg(enable=True, dump_dir="/tmp/x", max_pending=2):
    cfg = _cfg()
    cfg.trainer.log_image = {"enable": enable, "max_pending": max_pending, "png_compress_level": 0}
    cfg.trainer.rollout_data_dir = dump_dir
    return cfg


def _img_batch(images=None):
    import types as _t

    return _t.SimpleNamespace(non_tensor_batch={"image_data": images} if images is not None else {})


def test_images_are_not_dumped_when_disabled(monkeypatch):
    t = _Trainer(_img_cfg(enable=False))
    t._vagen_dump_images(_img_batch([["frame"]]))
    assert t._vagen_image_futures == []


def test_images_are_not_dumped_without_a_destination():
    t = _Trainer(_img_cfg(dump_dir=None))
    t._vagen_dump_images(_img_batch([["frame"]]))
    assert t._vagen_image_futures == []


def test_batch_without_images_is_skipped():
    """Text-only environments must not create an actor per step."""
    t = _Trainer(_img_cfg())
    t._vagen_dump_images(_img_batch(None))
    assert t._vagen_image_actors == {}


def test_in_flight_writes_are_capped(monkeypatch):
    """★ An environment that renders every turn queues frames faster than they are
    written; without the cap the driver's object store grows without bound."""
    import vagen.trainer.mixin as m

    monkeypatch.setattr(m, "METRIC_REGISTRY", {}, raising=False)
    fake_ray = MagicMock()
    fake_ray.wait.side_effect = lambda futs, num_returns: (futs[:1], futs[1:])
    monkeypatch.setitem(__import__("sys").modules, "ray", fake_ray)

    actor = MagicMock()
    actor.dump_images.remote.side_effect = lambda **kw: f"future{len(kw)}"
    t = _Trainer(_img_cfg(max_pending=2))
    t._vagen_image_actors["/tmp/x"] = actor

    for _ in range(5):
        t._vagen_dump_images(_img_batch([["frame"]]))

    assert len(t._vagen_image_futures) <= 2, t._vagen_image_futures
    assert fake_ray.get.called, "a completed write must be reaped, not just dropped"


def test_flush_before_save_drains_pending_writes(monkeypatch):
    """★ Saving can delete directories an in-flight write is still targeting."""
    fake_ray = MagicMock()
    monkeypatch.setitem(__import__("sys").modules, "ray", fake_ray)

    hf = MagicMock(**{"should_upload.return_value": False})
    t = _Trainer(_cfg(), hf=hf)
    t._vagen_image_futures = ["f1", "f2"]

    t._fit_save_checkpoint()

    fake_ray.get.assert_called_once_with(["f1", "f2"])
    assert t._vagen_image_futures == []


# --------------------------------------------------------- image token collapsing


class _Proc:
    image_token = "<|image_pad|>"


def test_image_token_runs_are_collapsed_in_dumps():
    """★ One frame expands to hundreds of repeats of the same token, which buries the
    prompt in the JSONL dump."""
    t = _Trainer(_cfg())
    t.processor = _Proc()
    seen = {}

    class _Base(_FakeBase):
        def _dump_generations(self, inputs, outputs, *a, **k):
            seen["inputs"], seen["outputs"] = inputs, outputs

    t.__class__ = type("_T", (VagenV0Mixin, _Base), {})
    t._dump_generations(["a <|image_pad|><|image_pad|><|image_pad|> b"], ["out"], None, None, None, "/tmp")

    assert seen["inputs"] == ["a <image> b"]
    assert seen["outputs"] == ["out"]


def test_collapsing_can_be_turned_off():
    cfg = _cfg()
    cfg.trainer.replace_image_tokens_for_logging = False
    t = _Trainer(cfg)
    t.processor = _Proc()
    seen = {}

    class _Base(_FakeBase):
        def _dump_generations(self, inputs, outputs, *a, **k):
            seen["inputs"] = inputs

    t.__class__ = type("_T", (VagenV0Mixin, _Base), {})
    t._dump_generations(["a <|image_pad|> b"], ["out"], None, None, None, "/tmp")

    assert seen["inputs"] == ["a <|image_pad|> b"]


def test_unknown_processor_leaves_text_alone_rather_than_failing():
    """★ A processor that declares no image token must not break the dump -- the log is
    then merely long, which is not worth failing a training step over."""
    from vagen.utils.image_token_utils import replace_image_tokens_for_logging

    class _Bare:
        pass

    import warnings as _w

    with _w.catch_warnings(record=True) as caught:
        _w.simplefilter("always")
        out = replace_image_tokens_for_logging(["a <|image_pad|> b"], _Bare())

    assert out == ["a <|image_pad|> b"]
    assert any("declares no image token" in str(c.message) for c in caught)


def test_image_token_is_read_from_the_tokenizer_when_only_an_id_is_exposed():
    """Processors vary in where they keep the placeholder; a per-family table is what
    this avoids."""
    from vagen.utils.image_token_utils import get_image_token

    class _Tok:
        def convert_ids_to_tokens(self, i):
            return "<IMG>" if i == 7 else None

    class _P:
        image_token_id = 7
        tokenizer = _Tok()

    assert get_image_token(_P()) == "<IMG>"


# ------------------------------------- the capability lives on the harness, not a list


def test_the_splitting_property_is_declared_by_each_harness():
    """★ Every registered harness must answer the question itself.

    The trainer used to carry `SPLITTING_HARNESSES = ("no_concat", "compact")`. A tuple
    of names is the arrangement the guard's own docstring rejects for estimators, and for
    the same reason: a harness added later is treated as non-splitting by default, so the
    guard silently stops guarding for it and a row-local estimator is accepted under a
    policy that truncates every trajectory.
    """
    from vagen.core.harness import BaseHarness
    from vagen.harness import HARNESSES

    assert HARNESSES, "no harnesses registered; this test is vacuous"
    for name, cls in HARNESSES.items():
        assert issubclass(cls, BaseHarness), f"{name} is not a BaseHarness"
        assert isinstance(cls.splits_episode_across_rows, bool), (
            f"{name} does not declare splits_episode_across_rows"
        )
    assert HARNESSES["concat"].splits_episode_across_rows is False
    assert HARNESSES["no_concat"].splits_episode_across_rows is True
    assert HARNESSES["compact"].splits_episode_across_rows is True


def test_the_trainer_does_not_keep_its_own_list_of_them():
    """The structural half: the tuple must not come back.

    Checked against every mixin class in the module rather than one name. A
    `not hasattr` assertion is the kind that passes for free if the name it is given
    stops existing, so the class list is derived instead of written down -- the first
    draft of this test named a class that does not exist and would have been vacuous had
    the import not failed outright.
    """
    import inspect

    from vagen.trainer import mixin

    classes = [c for _, c in inspect.getmembers(mixin, inspect.isclass)
               if c.__module__ == mixin.__name__]
    assert classes, "found no mixin classes; this test is vacuous"
    for cls in classes:
        assert not hasattr(cls, "SPLITTING_HARNESSES"), (
            f"{cls.__name__} carries the central tuple again; a harness's layout "
            "belongs on the harness"
        )


def test_a_harness_defined_outside_this_repo_is_honoured():
    """★ The point of the refactor: any BaseHarness subclass, not the three we ship.

    Registers a splitting harness the trainer has never heard of and requires the guard
    to fire for it. Under the old tuple this passed silently -- the unknown name was not
    listed, so the run proceeded with an estimator that drops every turn after the first.
    """
    from vagen.core.harness import BaseHarness
    from vagen.harness import HARNESSES

    class ThirdPartyHarness(BaseHarness):
        splits_episode_across_rows = True

        def next_call(self):  # pragma: no cover - never invoked
            raise NotImplementedError

    HARNESSES["third_party"] = ThirdPartyHarness
    try:
        t = _Trainer(_cfg(adv="gae", harness="third_party"))
        with pytest.raises(ValueError, match="scores one row at a time"):
            t._vagen_check_estimator_spans_the_layout()
    finally:
        del HARNESSES["third_party"]


def test_an_unregistered_harness_is_assumed_to_split():
    """Fail safe. A name that resolves to nothing must not be read as concat -- refusing
    a row-local estimator is an error message, allowing one is a wrong run."""
    t = _Trainer(_cfg(adv="gae", harness="not_registered_anywhere"))
    assert t._vagen_harness_splits_rows() is True
    with pytest.raises(ValueError, match="scores one row at a time"):
        t._vagen_check_estimator_spans_the_layout()


def test_a_metric_returning_a_mapping_is_spread_over_subkeys(monkeypatch):
    """A turn count is only meaningful as min/max/mean together, so a metric may return
    a mapping rather than forcing three registry entries for one concept."""
    import vagen.trainer.mixin as m

    monkeypatch.setattr(m, "METRIC_REGISTRY", {"spread": lambda b: {"min": 1.0, "max": 9.0}})
    t = _Trainer(_cfg())
    t._fit_compute_advantage(_batch())

    assert t.metrics["custom_metrics/train/spread/min"] == 1.0
    assert t.metrics["custom_metrics/train/spread/max"] == 9.0
    assert "custom_metrics/train/spread" not in t.metrics


# --------------------------------------------- split padding (no_concat divisibility)


def _split_batch(n, width=3):
    """A batch shaped like one that has reached _vagen_after_advantage."""
    from verl import DataProto

    return DataProto.from_single_dict(
        {
            "attention_mask": torch.ones(n, width, dtype=torch.long),
            "response_mask": torch.ones(n, width, dtype=torch.long),
            "advantages": torch.full((n, width), 2.0),
            "token_level_scores": torch.full((n, width), 3.0),
        }
    )


def _pad_cfg(actor_mini=16, critic_mini=0, critic_enable=False, rollout_n=1):
    cfg = _cfg()
    cfg.actor_rollout_ref = OmegaConf.create(
        {"actor": {"ppo_mini_batch_size": actor_mini}, "rollout": {"n": rollout_n}}
    )
    cfg.critic = OmegaConf.create({"enable": critic_enable, "ppo_mini_batch_size": critic_mini})
    return cfg


def test_a_row_count_that_does_not_divide_the_mini_batch_is_padded_up():
    """no_concat produced 76 rows against a mini batch of 16 and the step died on
    `AssertionError: 76 % 16 != 0`. The world size alone would not have caught it."""
    t = _Trainer(_pad_cfg(actor_mini=16))
    t.actor_rollout_wg = types.SimpleNamespace(world_size=4)
    out = t._vagen_pad_rows_for_split(_split_batch(76))
    assert out.batch.batch_size[0] == 80  # lcm(4, 16) = 16 -> next multiple
    assert t.metrics["custom_metrics/train/split_pad_rows"] == 4.0


def test_padding_rows_cannot_reach_a_loss():
    """Repeating a real row unmasked -- what pad_dataproto_to_divisor does -- would weight
    that episode more heavily than the others."""
    t = _Trainer(_pad_cfg(actor_mini=16))
    t.actor_rollout_wg = types.SimpleNamespace(world_size=4)
    out = t._vagen_pad_rows_for_split(_split_batch(76))
    tail = out.batch[76:]
    assert tail["response_mask"].sum() == 0
    assert tail["advantages"].abs().sum() == 0
    assert tail["token_level_scores"].abs().sum() == 0
    # the real rows are untouched
    assert out.batch[:76]["response_mask"].sum() == 76 * 3


def test_an_already_divisible_batch_is_returned_untouched():
    t = _Trainer(_pad_cfg(actor_mini=16))
    t.actor_rollout_wg = types.SimpleNamespace(world_size=4)
    b = _split_batch(80)
    assert t._vagen_pad_rows_for_split(b) is b
    assert "custom_metrics/train/split_pad_rows" not in t.metrics


def test_the_critic_mini_batch_is_honoured_too():
    """Both the actor's and the critic's split assert on the same row count."""
    t = _Trainer(_pad_cfg(actor_mini=16, critic_mini=24, critic_enable=True))
    t.actor_rollout_wg = types.SimpleNamespace(world_size=4)
    out = t._vagen_pad_rows_for_split(_split_batch(76))
    n = out.batch.batch_size[0]
    assert n % 16 == 0 and n % 24 == 0 and n % 4 == 0, n
    assert n == 96  # lcm(4, 16, 24) = 48 -> 96


def test_the_mini_batch_is_scaled_by_rollout_n_before_padding():
    """verl multiplies ppo_mini_batch_size by rollout.n to get rows, then divides by the
    DP size. Padding against the unscaled number divides evenly at the driver and still
    leaves each worker's shard indivisible -- the real run died on `340 % 16 != 0` while
    its 1360-row driver batch was a clean multiple of the unscaled 16."""
    t = _Trainer(_pad_cfg(actor_mini=16, rollout_n=4))
    t.actor_rollout_wg = types.SimpleNamespace(world_size=4)
    out = t._vagen_pad_rows_for_split(_split_batch(1360))
    n = out.batch.batch_size[0]
    assert n == 1408, n            # lcm(4, 16*4) = 64 -> next multiple above 1360
    assert (n // 4) % ((16 * 4) // 4) == 0  # per-worker shard divides its per-gpu mini


def test_the_rescope_runs_after_verl_computes_its_data_metrics():
    """critic/score/* is produced inside _fit_collect_metrics, several hooks after
    _fit_compute_advantage. Rescoping from the earlier hook found no key to rewrite and
    silently did nothing -- the metric fix shipped inert until a live no_concat run showed
    episode_score/mean 0.456 beside critic/score/mean 0.027."""
    t = _Trainer(_pad_cfg())
    t.metrics = {
        "custom_metrics/train/episode_score/mean": 0.456,
        "critic/score/mean": 0.027,
        "critic/score/max": 0.9,
        "critic/score/min": 0.0,
    }
    t._fit_collect_metrics(_batch())

    assert t.metrics["critic/score/mean"] == 0.456
    assert t.metrics["critic/score/by_row/mean"] == 0.027
    assert t.metrics["critic/score/by_row/max"] == 0.9
    assert "critic/score/max" not in t.metrics


def test_a_turn_level_loss_refuses_an_estimator_with_no_turn_id():
    """★ The guard asked `spans_rows`, which is a different question. `trajectory_grpo`
    stitches an episode's rows together (so spans_rows is True) but returns a bare
    AdvantageOutputs rather than going through `_Packed.emit`, so it publishes no
    `turn_id` -- and the turn-level losses read that column. Keyed off the wrong predicate
    the pairing passed every startup check and raised inside the first backward pass,
    which is exactly what this guard exists to pre-empt."""
    from vagen.custom_advantage import publishes_turn_id, spans_rows

    assert spans_rows("trajectory_grpo") is True
    assert publishes_turn_id("trajectory_grpo") is False
    for other in ("default_gae", "token_level_gae", "turn_level_gae",
                  "bi_level_gae", "bi_level_gae_varlam"):
        assert publishes_turn_id(other) is True, other


def test_padding_rows_carry_no_gradient():
    """★ Behavioural, not a grep. `_vagen_filter` padded with `pad_dataproto_to_divisor`,
    which repeats real rows verbatim -- and it runs AFTER advantage, so up to world_size-1
    real episodes carried their advantages and response_mask twice: double gradient weight,
    and double weight in critic/score. The padder thirty lines above says exactly that in
    its docstring; the filter's call site had not caught up. Both share the neutralised
    filler now, so this asserts what the filler contains."""
    import torch
    from verl import DataProto

    from vagen.trainer.mixin import VagenLogicMixin

    n, width = 3, 4
    batch = DataProto.from_dict({
        "attention_mask": torch.ones(n, width, dtype=torch.long),
        "response_mask": torch.ones(n, width, dtype=torch.long),
        "advantages": torch.full((n, width), 2.0),
        "token_level_scores": torch.full((n, width), 3.0),
    })
    padded = VagenLogicMixin._vagen_pad_to_multiple(VagenLogicMixin, batch, 4)

    assert len(padded.batch["attention_mask"]) == 4, "did not pad to the multiple"
    for key in ("response_mask", "advantages", "token_level_scores"):
        real, filler = padded.batch[key][:n], padded.batch[key][n:]
        assert real.abs().sum() > 0, f"{key} lost its real rows"
        assert filler.abs().sum() == 0, (
            f"{key} in the filler row is non-zero -- that row will contribute gradient "
            f"and weight one episode twice")
