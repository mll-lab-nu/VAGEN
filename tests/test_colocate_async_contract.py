"""Contracts at the VAGEN <-> V1 colocate_async boundary."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
from omegaconf import OmegaConf

from vagen.rollout.trajectory import Conversation
from vagen.training.tq_utils import (
    inflight_is_stale,
    token_level_reward_tensor,
    trajectory_columns,
    trajectory_identity,
)


def test_inflight_limit_is_version_based_and_never_applies_to_validation():
    assert not inflight_is_stale(
        current_step=2, start_step=1, max_inflight_steps=2, partition_id="train"
    )
    assert inflight_is_stale(
        current_step=3, start_step=1, max_inflight_steps=2, partition_id="train"
    )
    assert not inflight_is_stale(
        current_step=99, start_step=1, max_inflight_steps=2, partition_id="val"
    )


def test_v1_identity_maps_uid_and_session_without_position_assumptions():
    assert trajectory_identity({"uid": "prompt-b", "session_id": 7}) == ("prompt-b", 7)
    assert trajectory_identity(
        {"uid": "new", "session_id": 9, "group_idx": "legacy", "traj_idx": 2}
    ) == ("legacy", 2)


def test_token_reward_is_transported_without_reencoding_or_relocation():
    output = SimpleNamespace(
        response_ids=[11, 12, 13],
        reward_score=0.75,
        extra_fields={"per_token_reward": [0.0, 0.25, 0.5]},
    )
    assert token_level_reward_tensor(output).tolist() == [0.0, 0.25, 0.5]


@pytest.mark.parametrize(
    "response_ids,rewards,score,match",
    [
        ([1, 2], [1.0], 1.0, "2 tokens"),
        ([1, 2], [0.2, 0.3], 1.0, "must equal"),
    ],
)
def test_tq_boundary_rejects_misaligned_token_rewards(response_ids, rewards, score, match):
    output = SimpleNamespace(
        response_ids=response_ids,
        reward_score=score,
        extra_fields={"per_token_reward": rewards},
    )
    with pytest.raises(ValueError, match=match):
        token_level_reward_tensor(output)


def test_padding_rows_do_not_inherit_a_real_episode_identity():
    columns = trajectory_columns(
        [
            {
                "episode_id": "episode-real",
                "group_idx": "group",
                "traj_idx": 0,
                "turn_idx": 0,
                "rollout_metadata": {"scene_id": "scene-a"},
            },
            # V1 creates this by copying the first row's extra_fields.
            {
                "episode_id": "episode-real",
                "group_idx": "group",
                "traj_idx": 0,
                "turn_idx": 0,
                "rollout_metadata": {"scene_id": "scene-a"},
            },
        ],
        keys=["group_0_0", "padabc_0_0"],
        uids=["group", "padabc"],
        padding=[False, True],
    )
    assert columns["episode_id"].tolist() == ["episode-real", "padabc_0_0"]
    assert columns["group_idx"].tolist() == ["group", "padabc"]
    assert columns["response_spans"].tolist() == [None, []]
    assert columns["rollout_metadata"].tolist() == [
        {"scene_id": "scene-a"},
        {},
    ]


def test_conversation_tracks_the_policy_version_interval():
    conversation = Conversation()
    conversation.add_context([1])
    conversation.add_response([2], [-0.1])
    conversation.observe_weights_version((3, 3))
    conversation.add_context([4])
    conversation.add_response([5], [-0.2])
    conversation.observe_weights_version((4, 5))

    row = conversation.row()
    assert (row.min_global_steps, row.max_global_steps) == (3, 5)


def test_entrypoint_switches_the_manager_with_the_trainer_generation():
    from vagen.training.main import _V0_MANAGER, _V1_MANAGER, _prepare_trainer_mode

    config = OmegaConf.create(
        {
            "trainer": {"use_v1": True, "v1": {"trainer_mode": "colocate_async"}},
            "actor_rollout_ref": {
                "rollout": {
                    "free_cache_engine": False,
                    "agent": {"agent_loop_manager_class": _V0_MANAGER},
                }
            },
            "transfer_queue": {"enable": False},
        }
    )
    _prepare_trainer_mode(config)
    assert config.transfer_queue.enable is True
    assert config.actor_rollout_ref.rollout.free_cache_engine is True
    assert config.actor_rollout_ref.rollout.agent.agent_loop_manager_class == _V1_MANAGER

    config.trainer.use_v1 = False
    _prepare_trainer_mode(config)
    assert config.transfer_queue.enable is False
    assert config.actor_rollout_ref.rollout.free_cache_engine is False
    assert config.actor_rollout_ref.rollout.agent.agent_loop_manager_class == _V0_MANAGER


def test_v1_side_channels_are_written_back_to_transfer_queue():
    source = Path("vagen/training/trainer/v1.py").read_text(encoding="utf-8")
    assert 'for side_channel in ("value_mask", "turn_id")' in source
    assert 'output["loss_mask"] = output["response_mask"].clone()' in source


def test_default_training_flags_select_colocate_async():
    flags = Path("vagen/configs/training_defaults.flags").read_text(encoding="utf-8")
    assert "trainer.use_v1=True" in flags
    assert "trainer.v1.trainer_mode=colocate_async" in flags
    assert "vagen.training.agent_loop.tq.VagenAgentLoopManagerTQ" in flags
    assert "transfer_queue.enable=True" in flags


def test_vagen_config_supplies_v1_sampling_defaults_missing_from_pinned_verl():
    config = OmegaConf.load("vagen/configs/vagen_multiturn.yaml")
    assert config.actor_rollout_ref.rollout.repetition_penalty == 1.0


def test_v1_inherits_best_actor_validation_hook():
    from vagen.training.trainer.v1 import VagenV1Mixin

    assert hasattr(VagenV1Mixin, "_vagen_maybe_save_best_actor")


def test_v1_shutdown_stops_dataloader_workers_before_ray_teardown():
    from vagen.training.trainer.v1 import VagenV1Mixin

    class Iterator:
        calls = 0

        def _shutdown_workers(self):
            self.calls += 1

    train_iterator, val_iterator = Iterator(), Iterator()
    trainer = VagenV1Mixin.__new__(VagenV1Mixin)
    trainer.train_dataloader_it = object()
    trainer.train_dataloader = SimpleNamespace(_iterator=train_iterator)
    trainer.val_dataloader = SimpleNamespace(_iterator=val_iterator)

    trainer._vagen_shutdown_dataloaders()

    assert trainer.train_dataloader_it is None
    assert train_iterator.calls == val_iterator.calls == 1
    assert trainer.train_dataloader._iterator is None
    assert trainer.val_dataloader._iterator is None


def test_tq_worker_accepts_vagen_dataset_without_an_index_column():
    import torch
    from tensordict import TensorDict

    from vagen.training.agent_loop.tq import _trajectory_indices

    batch = TensorDict({"input_ids": torch.zeros(2, 1)}, batch_size=[2])
    assert _trajectory_indices(batch) == [0, 1]

    indexed = TensorDict(
        {"input_ids": torch.zeros(2, 1), "index": torch.tensor([7, 9])},
        batch_size=[2],
    )
    assert _trajectory_indices(indexed).tolist() == [7, 9]


@pytest.mark.asyncio
async def test_tq_worker_rejects_empty_sessions_and_preserves_each_rows_token_reward(
    monkeypatch,
):
    import torch
    from verl.experimental.agent_loop import AgentLoopOutput

    from vagen.training.agent_loop import tq as adapter

    worker_cls = adapter.VagenAgentLoopWorkerTQ.__ray_metadata__.modified_class
    with pytest.raises(adapter.EmptyAgentLoopOutput):
        await worker_cls._agent_loop_postprocess(
            object(), [], False, uid="uid-a", session_id=0
        )

    class Worker:
        async def _compute_score(self, outputs, kwargs):
            return None

        async def _compute_teacher_logprobs(self, *args, **kwargs):
            return None

        def _compute_multi_modal_inputs(self, output, input_ids):
            return {}

        def _get_mm_processor_kwargs(self, audios=None):
            return {}

        def _compute_position_ids(self, input_ids, attention_mask, *args):
            return torch.arange(input_ids.shape[-1]).unsqueeze(0)

    outputs = [
        AgentLoopOutput(
            prompt_ids=[1],
            response_ids=[10, 11],
            response_mask=[1, 1],
            response_logprobs=[-0.1, -0.2],
            reward_score=0.75,
            metrics={},
            extra_fields={
                "episode_id": "episode-a",
                "per_token_reward": [0.25, 0.5],
                "min_global_steps": 2,
                "max_global_steps": 3,
            },
        ),
        AgentLoopOutput(
            prompt_ids=[1],
            response_ids=[12],
            response_mask=[1],
            response_logprobs=[-0.3],
            reward_score=2.0,
            metrics={},
            extra_fields={
                "episode_id": "episode-a",
                "per_token_reward": [2.0],
                "min_global_steps": 3,
                "max_global_steps": 3,
            },
        ),
    ]
    captured = {}

    async def fake_put(**kwargs):
        captured.update(kwargs)

    monkeypatch.setattr(adapter.tq, "async_kv_batch_put", fake_put)
    await worker_cls._agent_loop_postprocess(
        Worker(), outputs, False, uid="uid-a", session_id=4, global_steps=3
    )

    assert captured["keys"] == ["uid-a_4_0", "uid-a_4_1"]
    assert [row.tolist() for row in captured["fields"]["rm_scores"].unbind()] == [
        [0.25, 0.5],
        [2.0],
    ]
    assert [tag["min_global_steps"] for tag in captured["tags"]] == [2, 3]
    assert [tag["max_global_steps"] for tag in captured["tags"]] == [3, 3]


def test_v1_advantage_persists_value_mask_and_turn_id(monkeypatch):
    pytest.importorskip("transfer_queue")
    import torch
    from transfer_queue import KVBatchMeta
    from verl.utils.tensordict_utils import list_of_dict_to_tensordict

    from vagen.training.trainer import v1

    rows = []
    for turn, width in enumerate((2, 1)):
        rows.append(
            {
                "uid": "uid-a",
                "session_id": 0,
                "response_mask": torch.ones(width, dtype=torch.long),
                "rm_scores": torch.tensor(([0.0, 0.5] if width == 2 else [0.5])),
                "prompts": torch.tensor([1, 2]),
                "responses": torch.arange(10, 10 + width),
                "rollout_log_probs": torch.full((width,), -0.1),
                "old_log_probs": torch.full((width,), -0.1),
                "ref_log_prob": torch.full((width,), -0.1),
                "values": torch.full((width,), 0.1),
                "num_turns": 1,
                "reward_model": {},
                "data_source": "stub",
                "extra_fields": {
                    "episode_id": "episode-a",
                    "group_idx": "uid-a",
                    "traj_idx": 0,
                    "turn_idx": turn,
                    "conversation_id": turn,
                    "response_spans": [(0, width)],
                    "episode_turns": 2,
                    "rollout_metadata": {"scene_id": "scene-a"},
                    "reward_extra_info": {},
                },
            }
        )
    stored = list_of_dict_to_tensordict(rows)
    writes = {}
    written_meta = object()

    def fake_get(*, select_fields, **_kwargs):
        return stored.select(*select_fields).clone(recurse=True)

    def fake_put(*, fields, **_kwargs):
        writes.update(dict(fields.items()))
        return written_meta

    monkeypatch.setattr(v1.tq, "kv_batch_get", fake_get)
    monkeypatch.setattr(v1.tq, "kv_batch_put", fake_put)

    trainer = v1.VagenV1Mixin.__new__(v1.VagenV1Mixin)
    trainer.timing_raw = {}
    trainer.tokenizer = SimpleNamespace(
        pad_token_id=0,
        decode=lambda ids, **_kwargs: " ".join(str(int(token)) for token in ids),
    )
    trainer.processor = None
    trainer.config = OmegaConf.create(
        {
            "algorithm": {
                "adv_estimator": "turn_level_gae",
                "gamma": 1.0,
                "lam": 1.0,
                "use_kl_in_reward": False,
                "norm_adv_by_std_in_grpo": True,
            },
            "actor_rollout_ref": {"rollout": {"n": 1}},
        }
    )
    meta = KVBatchMeta(
        keys=["uid-a_0_0", "uid-a_0_1"],
        tags=[{}, {}],
        partition_id="train",
    )

    result = trainer._compute_advantage(meta, {})

    assert result is written_meta
    assert {"advantages", "returns", "value_mask", "turn_id"} <= writes.keys()
    assert [row.tolist() for row in writes["value_mask"].unbind()] == [[1, 0], [1]]
    assert [row.tolist() for row in writes["turn_id"].unbind()] == [[0, 0], [1]]

    materialized = trainer._vagen_materialize_validation_rows(meta)
    from verl import DataProto
    from tensordict import TensorDict
    from vagen.utils.concat_val_multi_turn import concat_val_multi_turn

    target = DataProto(
        batch=TensorDict({}, batch_size=1),
        non_tensor_batch={"uid": __import__("numpy").array(["uid-a"], dtype=object)},
    )
    merged = concat_val_multi_turn(materialized, target, trainer.tokenizer)
    assert merged.batch["rm_scores"].sum().item() == pytest.approx(1.0)
    assert merged.non_tensor_batch["episode_turns"].tolist() == [2]

    dumped = {}
    trainer._dump_generations = lambda **kwargs: dumped.update(kwargs)
    trainer._log_rollout_data(meta, {}, "/unused")
    assert dumped["scores"] == pytest.approx([0.5, 0.5])
    assert dumped["reward_extra_infos_dict"]["episode_id"] == [
        "episode-a",
        "episode-a",
    ]
    assert dumped["reward_extra_infos_dict"]["scene_id"] == [
        "scene-a",
        "scene-a",
    ]
