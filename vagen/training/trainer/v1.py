"""VAGEN bindings for verl's V1 colocated asynchronous PPO trainer."""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Iterable

import numpy as np
import torch
import transfer_queue as tq
from tensordict import TensorDict
from transfer_queue import KVBatchMeta

from verl import DataProto
from verl.trainer.ppo.ray_trainer import apply_kl_penalty
from verl.trainer.ppo.rollout_corr_helper import compute_rollout_correction_and_add_to_batch
from verl.trainer.ppo.v1.trainer_colocate_async import PPOTrainerColocateAsync
from verl.trainer.ppo.v1.utils import compute_advantage_for_multi_trajectories
from verl.utils import tensordict_utils as tu
from verl.utils.debug import marked_timer
from verl.workers.utils.padding import response_to_nested

from vagen.models import replace_image_tokens_for_logging
from vagen.training.filters import FILTER_REGISTRY
from vagen.training.metrics import METRIC_REGISTRY
from vagen.training.tq_utils import (
    parse_tq_trajectory_key,
    trajectory_columns,
    unwrap_non_tensor,
)
from vagen.training.trainer.logic import collect_registry_metrics
from vagen.training.trainer.mixin import VagenLogicMixin, rollout_metadata_columns
from vagen.utils.concat_val_multi_turn import concat_val_multi_turn
from vagen.utils.episode_log import describe_columns, rows_from_validation
from vagen.utils.wandb_episodes import EpisodeTableLogger


def _object_array(values: Iterable[Any]) -> np.ndarray:
    values = list(values)
    result = np.empty(len(values), dtype=object)
    result[:] = values
    return result


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if hasattr(value, "unbind") and callable(value.unbind):
        try:
            return list(value.unbind())
        except (RuntimeError, TypeError):
            pass
    if hasattr(value, "tolist") and not isinstance(value, list):
        try:
            converted = value.tolist()
            return converted if isinstance(converted, list) else [converted]
        except (RuntimeError, TypeError):
            pass
    return list(value)


def _pad_1d(rows: list[torch.Tensor], *, value: int | float, left: bool = False) -> torch.Tensor:
    if not rows:
        return torch.empty((0, 0))
    width = max(int(row.numel()) for row in rows)
    result = rows[0].new_full((len(rows), width), value)
    for index, row in enumerate(rows):
        row = row.reshape(-1)
        if row.numel() == 0:
            continue
        if left:
            result[index, -row.numel() :] = row
        else:
            result[index, : row.numel()] = row
    return result


class VagenV1Mixin(VagenLogicMixin):
    """Attach VAGEN's trajectory semantics to V1's TransferQueue lifecycle."""

    _SUPPORTED_LEGACY_FILTERS = {"reward_variance", "reward_variance_top_p"}

    def on_step_begin(self):
        super().on_step_begin()
        max_inflight_steps = int(
            self.config.trainer.v1.colocate_async.get("max_inflight_steps", 0) or 0
        )
        if max_inflight_steps <= 0:
            return
        expired = self.agent_loop_manager.cancel_stale(
            self.global_steps, max_inflight_steps
        )
        if expired:
            # ReplayBufferAsync observes the resulting failure status, clears every
            # sibling row for the UID, and submits exactly one replacement prompt.
            print(
                f"[vagen] cancelled {expired} rollout group(s) older than "
                f"{max_inflight_steps} policy versions"
            )

    def _vagen_advantage_data(self, batch: KVBatchMeta) -> tuple[DataProto, torch.Tensor]:
        fields = [
            "uid",
            "response_mask",
            "rm_scores",
            "rollout_log_probs",
            "old_log_probs",
            "ref_log_prob",
            "values",
            "extra_fields",
        ]
        raw = tq.kv_batch_get(keys=batch.keys, partition_id=batch.partition_id, select_fields=fields)
        response_mask = raw["response_mask"]
        extras = _as_list(raw.pop("extra_fields"))
        uids = [unwrap_non_tensor(value) for value in _as_list(raw.pop("uid"))]
        data = DataProto(batch=raw.to_padded_tensor())
        padding = [bool(tag.get("is_padding", False)) for tag in batch.tags]
        data.non_tensor_batch.update(
            trajectory_columns(extras, keys=batch.keys, uids=uids, padding=padding)
        )
        return data, response_mask

    def _vagen_filter_data(self, batch: KVBatchMeta) -> DataProto:
        raw = tq.kv_batch_get(
            keys=batch.keys,
            partition_id=batch.partition_id,
            select_fields=["uid", "response_mask", "rm_scores", "extra_fields"],
        )
        extras = _as_list(raw.pop("extra_fields"))
        uids = [unwrap_non_tensor(value) for value in _as_list(raw.pop("uid"))]
        data = DataProto(batch=raw.to_padded_tensor())
        data.batch["token_level_scores"] = data.batch["rm_scores"]
        data.non_tensor_batch.update(trajectory_columns(extras, keys=batch.keys, uids=uids))
        data.non_tensor_batch["__vagen_tq_row__"] = np.arange(len(batch), dtype=np.int64)
        return data

    def _vagen_collect_metrics(self, data: DataProto, metrics: dict) -> None:
        with marked_timer("custom_metrics", self.timing_raw, color="magenta"):
            metrics.update(
                collect_registry_metrics(METRIC_REGISTRY, data, prefix="custom_metrics/train")
            )

    def _vagen_filter_tq(self, batch: KVBatchMeta, metrics: dict) -> KVBatchMeta:
        """Run the shipped reward-variance filters before V1 pads the row batch."""
        cfg = self.config.get("filter", None)
        data = self._vagen_filter_data(batch)
        self._vagen_collect_metrics(data, metrics)
        if not (cfg and cfg.get("enable", False)):
            if batch.extra_info is None:
                batch.extra_info = {}
            batch.extra_info["vagen_metrics_collected"] = True
            return batch
        if cfg.name not in self._SUPPORTED_LEGACY_FILTERS:
            raise ValueError(
                f"V1 supports the built-in reward filters {sorted(self._SUPPORTED_LEGACY_FILTERS)}, "
                f"not filter.name={cfg.name!r}. A custom V1 filter should be implemented as a "
                "ReplayBuffer sampler so it can select TQ keys before model work starts."
            )

        filtered, _ = FILTER_REGISTRY[cfg.name](data, metrics, **cfg.filter_kwargs)
        indices = [int(index) for index in filtered.non_tensor_batch["__vagen_tq_row__"]]
        keep = set(indices)
        dropped_keys = [key for index, key in enumerate(batch.keys) if index not in keep]
        if dropped_keys:
            tq.kv_clear(keys=dropped_keys, partition_id=batch.partition_id)

        extra_info = dict(batch.extra_info or {})
        extra_info["vagen_metrics_collected"] = True
        extra_info["vagen_advantage_scale"] = float(
            metrics.get("filter/train/top_p/loss_scale", 1.0)
        )
        return KVBatchMeta(
            partition_id=batch.partition_id,
            keys=[batch.keys[index] for index in indices],
            tags=[batch.tags[index] for index in indices],
            fields=batch.fields,
            extra_info=extra_info,
        )

    def _balance_batch(self, batch: KVBatchMeta, metrics, logging_prefix="global_seqlen", keep_minibatch=False):
        # Reward-variance filters need neither values nor log-probs.  Applying them here
        # avoids doing model work for rows that will be discarded and lets V1's own
        # padding utility restore all downstream divisibility constraints afterwards.
        if not (batch.extra_info or {}).get("vagen_filter_applied", False):
            batch = self._vagen_filter_tq(batch, metrics)
            batch.extra_info["vagen_filter_applied"] = True
        return super()._balance_batch(
            batch,
            metrics,
            logging_prefix=logging_prefix,
            keep_minibatch=keep_minibatch,
        )

    def _compute_advantage(self, batch: KVBatchMeta, metrics: dict) -> KVBatchMeta:
        """Compute advantages with VAGEN's trajectory columns and persist side channels."""
        data, response_mask = self._vagen_advantage_data(batch)
        data.batch["token_level_scores"] = data.batch["rm_scores"]

        if self.config.algorithm.use_kl_in_reward:
            data, kl_metrics = apply_kl_penalty(
                data,
                kl_ctrl=self.kl_ctrl_in_reward,
                kl_penalty=self.config.algorithm.kl_penalty,
            )
            metrics.update(kl_metrics)
        else:
            data.batch["token_level_rewards"] = data.batch["token_level_scores"]

        rollout_corr_config = self.config.algorithm.get("rollout_correction", None)
        bypass_recomputing = rollout_corr_config and rollout_corr_config.get("bypass_mode", False)
        rollout_correction = (
            rollout_corr_config is not None
            and "rollout_log_probs" in data.batch
            and not bypass_recomputing
        )
        if rollout_correction:
            data, correction_metrics = compute_rollout_correction_and_add_to_batch(
                data, rollout_corr_config
            )
            metrics.update(correction_metrics)

        data = compute_advantage_for_multi_trajectories(
            data,
            batch_keys=batch.keys,
            adv_estimator=self.config.algorithm.adv_estimator,
            gamma=self.config.algorithm.gamma,
            lam=self.config.algorithm.lam,
            num_repeat=self.config.actor_rollout_ref.rollout.n,
            norm_adv_by_std_in_grpo=self.config.algorithm.get("norm_adv_by_std_in_grpo", True),
            config=self.config.algorithm,
        )
        data = self._vagen_write_value_mask(data)

        scale = float((batch.extra_info or {}).get("vagen_advantage_scale", 1.0))
        if scale != 1.0:
            data.batch["advantages"] = data.batch["advantages"] * scale

        if not (batch.extra_info or {}).get("vagen_metrics_collected", False):
            real_indices = [
                index for index, tag in enumerate(batch.tags) if not tag.get("is_padding", False)
            ]
            metric_data = data.select_idxs(real_indices) if real_indices else data[:0]
            self._vagen_collect_metrics(metric_data, metrics)

        write_fields = ["advantages", "returns"]
        if self.config.algorithm.use_kl_in_reward:
            write_fields.append("token_level_rewards")
        if rollout_correction:
            write_fields.append("response_mask")
            if "rollout_is_weights" in data.batch:
                write_fields.append("rollout_is_weights")
        for side_channel in ("value_mask", "turn_id"):
            if side_channel in data.batch:
                write_fields.append(side_channel)

        output = {
            field: response_to_nested(data.batch[field], response_mask)
            for field in write_fields
        }
        if rollout_correction:
            output["loss_mask"] = output["response_mask"].clone()
        batch = tq.kv_batch_put(
            keys=batch.keys,
            partition_id=batch.partition_id,
            fields=TensorDict(output, batch_size=len(batch)),
        )
        return batch

    def _compute_metrics(self, batch, metrics, timing_raw, global_steps, epoch):
        super()._compute_metrics(batch, metrics, timing_raw, global_steps, epoch)
        self.metrics = metrics
        self._vagen_rescope_row_metrics_to_episodes()

    def _vagen_materialize_validation_rows(self, batch: KVBatchMeta) -> DataProto:
        fields = [
            "uid",
            "session_id",
            "prompts",
            "responses",
            "response_mask",
            "rm_scores",
            "num_turns",
            "reward_model",
            "data_source",
            "extra_fields",
        ]
        raw = tq.kv_batch_get(keys=batch.keys, partition_id=batch.partition_id, select_fields=fields)
        extras = _as_list(raw.pop("extra_fields"))
        uids = [unwrap_non_tensor(value) for value in _as_list(raw.pop("uid"))]
        session_ids = [unwrap_non_tensor(value) for value in _as_list(raw.pop("session_id", None))]

        prompt_rows = [torch.as_tensor(row) for row in _as_list(raw.pop("prompts"))]
        response_rows = [torch.as_tensor(row) for row in _as_list(raw.pop("responses"))]
        mask_rows = [torch.as_tensor(row) for row in _as_list(raw.pop("response_mask"))]
        score_rows = [torch.as_tensor(row) for row in _as_list(raw.pop("rm_scores"))]
        prompts = _pad_1d(prompt_rows, value=self.tokenizer.pad_token_id, left=True)
        responses = _pad_1d(response_rows, value=self.tokenizer.pad_token_id)
        response_masks = _pad_1d(mask_rows, value=0)
        rm_scores = _pad_1d(score_rows, value=0.0)
        prompt_mask = torch.zeros_like(prompts, dtype=torch.long)
        response_attention = torch.zeros_like(responses, dtype=torch.long)
        for index, (prompt, response) in enumerate(zip(prompt_rows, response_rows, strict=True)):
            if prompt.numel():
                prompt_mask[index, -prompt.numel() :] = 1
            if response.numel():
                response_attention[index, : response.numel()] = 1

        tensor_batch = TensorDict(
            {
                "prompts": prompts,
                "responses": responses,
                "response_mask": response_masks,
                "rm_scores": rm_scores,
                "attention_mask": torch.cat([prompt_mask, response_attention], dim=1),
            },
            batch_size=len(batch),
        )
        non_tensor = trajectory_columns(extras, keys=batch.keys, uids=uids)
        non_tensor["uid"] = _object_array(uids)
        if session_ids:
            non_tensor["session_id"] = _object_array(session_ids)
        for name in ("num_turns", "reward_model", "data_source"):
            value = raw.pop(name, None)
            if value is not None:
                non_tensor[name] = _object_array(unwrap_non_tensor(item) for item in _as_list(value))
        return DataProto(batch=tensor_batch, non_tensor_batch=non_tensor)

    @staticmethod
    def _vagen_reward_columns(scores: list[float], infos: list[dict[str, Any]]) -> dict[str, list[Any]]:
        columns: dict[str, list[Any]] = {"reward": list(scores)}
        names = sorted({key for info in infos for key in info})
        for name in names:
            columns[name] = [info.get(name) for info in infos]
        return columns

    def _vagen_log_validation_episodes(
        self,
        inputs: list[str],
        outputs: list[str],
        scores: list[float],
        extras: dict[str, list[Any]],
    ) -> None:
        n = self.config.trainer.get("log_val_generations", 0)
        if not n:
            return
        print(f"[vagen] val episodes <- {describe_columns(extras, len(outputs))}")
        if "wandb" not in self.config.trainer.logger:
            return super()._maybe_log_val_generations(inputs, outputs, scores)
        if getattr(self, "_vagen_val_logger", None) is None:
            self._vagen_val_logger = EpisodeTableLogger()
        self._vagen_val_logger.submit(
            rows_from_validation(inputs, outputs, scores, extras.get("image_data"), extras),
            n,
            self.global_steps,
            self.config.trainer.get("val_log_select", "balanced"),
            float(self.config.trainer.get("val_log_success_ratio", 0.5)),
        )

    def _validate(self) -> dict[str, float]:
        """Validate at episode scope, including every row of a split VAGEN rollout."""
        sample_uids: list[str] = []
        sample_inputs: list[str] = []
        sample_outputs: list[str] = []
        sample_gts: list[Any] = []
        sample_scores: list[float] = []
        sample_turns: list[int] = []
        data_sources: list[str] = []
        merged_extras: defaultdict[str, list[Any]] = defaultdict(list)

        import uuid

        for batch_dict in self.val_dataloader:
            batch_dict["uid"] = np.array(
                [str(uuid.uuid4()) for _ in range(len(batch_dict["raw_prompt"]))], dtype=object
            )
            prompt_batch = tu.get_tensordict(batch_dict)
            tu.assign_non_tensor_data(prompt_batch, "global_steps", self.global_steps)
            tu.assign_non_tensor_data(prompt_batch, "validate", True)
            tags = [
                {"is_prompt": True, "status": "pending", "global_steps": self.global_steps}
                for _ in range(len(prompt_batch))
            ]
            tq.kv_batch_put(keys=list(prompt_batch["uid"]), partition_id="val", tags=tags)
            self.agent_loop_manager.generate_sequences(prompt_batch)
            batch, _ = self.replay_buffer.sample(
                global_steps=self.global_steps,
                partition_id="val",
                batch_size=len(prompt_batch),
            )

            if self.reward_loop_manager.reward_loop_worker_handles is None:
                self.checkpoint_manager.sleep_replicas()
                batch = self._compute_reward_colocate(batch)
                self.checkpoint_manager.update_weights()

            rows = self._vagen_materialize_validation_rows(batch)
            session_pairs = sorted(
                {
                    (str(rows.non_tensor_batch["group_idx"][index]), int(rows.non_tensor_batch["traj_idx"][index]))
                    for index in range(len(rows))
                }
            )
            target = DataProto(
                batch=TensorDict({}, batch_size=len(session_pairs)),
                non_tensor_batch={"uid": _object_array(group for group, _ in session_pairs)},
            )
            merged = concat_val_multi_turn(rows, target, self.tokenizer, self.processor)

            inputs = [
                self.tokenizer.decode(ids, skip_special_tokens=True)
                for ids in merged.batch["prompts"]
            ]
            outputs = [
                self.tokenizer.decode(ids, skip_special_tokens=True)
                for ids in merged.batch["responses"]
            ]
            scores = merged.batch["rm_scores"].sum(dim=1).tolist()
            infos = [dict(info or {}) for info in merged.non_tensor_batch["reward_extra_info"]]

            row_sources = rows.non_tensor_batch.get("data_source")
            row_rewards = rows.non_tensor_batch.get("reward_model")
            source_by_session: dict[tuple[str, int], Any] = {}
            reward_by_session: dict[tuple[str, int], Any] = {}
            for index in range(len(rows)):
                pair = (
                    str(rows.non_tensor_batch["group_idx"][index]),
                    int(rows.non_tensor_batch["traj_idx"][index]),
                )
                if row_sources is not None:
                    source_by_session.setdefault(pair, row_sources[index])
                if row_rewards is not None:
                    reward_by_session.setdefault(pair, row_rewards[index])

            sample_uids.extend(group for group, _ in session_pairs)
            sample_inputs.extend(inputs)
            sample_outputs.extend(outputs)
            sample_scores.extend(float(score) for score in scores)
            sample_turns.extend(int(value) for value in merged.non_tensor_batch["episode_turns"])
            data_sources.extend(str(source_by_session.get(pair, "unknown")) for pair in session_pairs)
            sample_gts.extend(
                (reward_by_session.get(pair) or {}).get("ground_truth")
                if isinstance(reward_by_session.get(pair), dict)
                else None
                for pair in session_pairs
            )
            for name, values in merged.non_tensor_batch.items():
                merged_extras[name].extend(list(values))

            tq.kv_clear(keys=batch.keys, partition_id=batch.partition_id)

        reward_columns = self._vagen_reward_columns(
            sample_scores,
            [dict(info or {}) for info in merged_extras.get("reward_extra_info", [])],
        )
        extras = {name: list(values) for name, values in merged_extras.items()}
        self._vagen_log_validation_episodes(sample_inputs, sample_outputs, sample_scores, extras)

        val_data_dir = self.config.trainer.get("validation_data_dir", None)
        if val_data_dir:
            dump_columns = reward_columns | {"uid": sample_uids}
            for name, values in rollout_metadata_columns(
                merged_extras.get("rollout_metadata", [])
            ).items():
                dump_columns.setdefault(name, values)
            self._dump_generations(
                inputs=sample_inputs,
                outputs=sample_outputs,
                gts=sample_gts,
                scores=sample_scores,
                reward_extra_infos_dict=dump_columns,
                dump_path=val_data_dir,
            )

        result = self._val_metrics_update(
            data_sources, sample_uids, reward_columns, sample_turns
        )
        self._vagen_last_val_metrics = result
        self.metrics = result
        return result

    def _dump_generations(self, inputs, outputs, *args, **kwargs):
        if self.config.trainer.get("replace_image_tokens_for_logging", True):
            inputs = replace_image_tokens_for_logging(inputs, self.processor)
            outputs = replace_image_tokens_for_logging(outputs, self.processor)
        return super()._dump_generations(inputs, outputs, *args, **kwargs)

    def _log_rollout_data(self, batch: KVBatchMeta, timing_raw: dict, rollout_data_dir: str):
        """Dump real VAGEN rows without relying on TQ non-tensor ``.tolist()``.

        TransferQueue 0.1.9 represents string/object columns with a LinkedList-backed
        ``NonTensorStack``. Upstream's generic dump calls ``.tolist()`` on that object,
        which fails; materializing through the same adapter as validation also preserves
        VAGEN's episode metadata and excludes synthetic padding rows.
        """
        with marked_timer("dump_rollout_generations", timing_raw, color="green"):
            rows = self._vagen_materialize_validation_rows(batch)
            real_indices = [
                index
                for index, tag in enumerate(batch.tags)
                if not tag.get("is_padding", False)
            ]
            ordered = sorted(
                real_indices,
                key=lambda index: parse_tq_trajectory_key(batch.keys[index]),
            )
            if not ordered:
                return

            inputs = [
                self.tokenizer.decode(rows.batch["prompts"][index], skip_special_tokens=True)
                for index in ordered
            ]
            outputs = [
                self.tokenizer.decode(rows.batch["responses"][index], skip_special_tokens=True)
                for index in ordered
            ]
            scores = [
                float(rows.batch["rm_scores"][index].sum().item())
                for index in ordered
            ]

            reward_models = rows.non_tensor_batch.get("reward_model")
            gts = [
                (
                    reward_models[index].get("ground_truth")
                    if reward_models is not None and isinstance(reward_models[index], dict)
                    else None
                )
                for index in ordered
            ]
            reward_infos = [
                dict(rows.non_tensor_batch["reward_extra_info"][index] or {})
                for index in ordered
            ]
            dump_columns = self._vagen_reward_columns(scores, reward_infos)
            dump_columns["uid"] = [batch.keys[index] for index in ordered]
            for name in ("episode_id", "group_idx", "traj_idx", "turn_idx"):
                values = rows.non_tensor_batch.get(name)
                if values is not None:
                    dump_columns[name] = [values[index] for index in ordered]
            metadata = rows.non_tensor_batch.get("rollout_metadata")
            if metadata is not None:
                for name, values in rollout_metadata_columns(
                    metadata[index] for index in ordered
                ).items():
                    dump_columns.setdefault(name, values)

            self._dump_generations(
                inputs=inputs,
                outputs=outputs,
                gts=gts,
                scores=scores,
                reward_extra_infos_dict=dump_columns,
                dump_path=rollout_data_dir,
            )

            images = rows.non_tensor_batch.get("image_data")
            if images is not None:
                shell = DataProto(
                    batch=TensorDict({}, batch_size=len(ordered)),
                    non_tensor_batch={
                        "image_data": _object_array(images[index] for index in ordered)
                    },
                )
                self._vagen_dump_images(shell)

    def _save_checkpoint(self):
        self._vagen_flush_images()
        upload = self._vagen_should_upload_hf()
        if upload:
            self._vagen_flush_hf()
        super()._save_checkpoint()
        if upload:
            self._vagen_upload_hf()

    def on_validate_end(self):
        super().on_validate_end()
        logger_ = getattr(self, "_vagen_val_logger", None)
        if logger_ is not None:
            logger_.flush()
        self.metrics = getattr(self, "_vagen_last_val_metrics", {})
        self._vagen_maybe_save_best_actor()

    def on_train_end(self):
        self._vagen_flush_images()
        self._vagen_flush_hf()
        logger_ = getattr(self, "_vagen_val_logger", None)
        if logger_ is not None:
            logger_.flush()
        return super().on_train_end()

    def _vagen_shutdown_dataloaders(self) -> None:
        """Stop multiprocessing loader children before Ray tears down this actor."""
        self.train_dataloader_it = None
        for name in ("train_dataloader", "val_dataloader"):
            loader = getattr(self, name, None)
            iterator = getattr(loader, "_iterator", None)
            shutdown = getattr(iterator, "_shutdown_workers", None)
            if shutdown is not None:
                shutdown()
            if loader is not None and hasattr(loader, "_iterator"):
                loader._iterator = None

    def _shutdown_dump_executor(self):
        """Drain every VAGEN background sink on all V1 exit paths.

        V1 returns directly after its normal final step and therefore does not call
        ``on_train_end`` on that path. Its dump-executor shutdown is the one cleanup
        hook shared by final-step, val-only, and loop-exhaustion exits.
        """
        self._vagen_flush_images()
        self._vagen_flush_hf()
        logger_ = getattr(self, "_vagen_val_logger", None)
        if logger_ is not None:
            logger_.flush()
        self._vagen_shutdown_dataloaders()
        return super()._shutdown_dump_executor()


class VagenPPOTrainerColocateAsync(VagenV1Mixin, PPOTrainerColocateAsync):
    """The VAGEN trainer on verl V1's colocated asynchronous scheduler."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.trainer_mode != "colocate_async":
            raise ValueError(
                "VagenPPOTrainerColocateAsync requires "
                "trainer.v1.trainer_mode=colocate_async"
            )
        self._vagen_init()
        self._vagen_last_val_metrics: dict[str, float] = {}


__all__ = ["VagenPPOTrainerColocateAsync", "VagenV1Mixin"]
