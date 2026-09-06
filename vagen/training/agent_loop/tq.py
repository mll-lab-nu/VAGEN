"""TransferQueue adapter for VAGEN's multi-output gym agent loop.

The harness still performs an OpenAI-like rollout and the environment remains the only
component allowed to produce token-level rewards.  This module is solely the V1
scheduling/transport boundary: it maps V1 identities, validates reward alignment, and
stores every conversation row in TransferQueue.
"""

from __future__ import annotations

import asyncio
import logging
import os
from typing import Any

import ray
import torch
import transfer_queue as tq
from tensordict import NonTensorData, NonTensorStack, TensorDict

from verl.experimental.agent_loop import AgentLoopManager, AgentLoopOutput, AgentLoopWorker, get_trajectory_info
from verl.trainer.ppo.v1.agent_loop_tq import apply_greedy_sampling_params
from verl.utils.ray_utils import auto_await
from verl.utils.tensordict_utils import list_of_dict_to_tensordict
from verl.workers.rollout.sampling import build_agent_loop_sampling_params

from vagen.training.tq_utils import (
    ROLLOUT_SOURCE,
    inflight_is_stale,
    token_level_reward_tensor,
    trajectory_identity,
)

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "INFO"))


class EmptyAgentLoopOutput(RuntimeError):
    """A session completed without a trajectory that can be trained on."""


async def _settle(tasks: list[asyncio.Task[Any]]) -> list[BaseException]:
    results = await asyncio.gather(*tasks, return_exceptions=True)
    return [result for result in results if isinstance(result, BaseException)]


@ray.remote
class VagenAgentLoopWorkerTQ(AgentLoopWorker):
    """V1 worker that preserves VAGEN's row and reward semantics in TQ."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        tq.init()
        self.background_tasks: set[asyncio.Task[Any]] = set()
        self.prompt_tasks: dict[Any, tuple[asyncio.Task[Any], int, str]] = {}

    async def generate_sequences(self, batch: TensorDict) -> None:
        """Start each prompt without waiting for its episode to finish."""
        validate = batch["validate"] if "validate" in batch else False
        batch.pop("validate", None)
        config = self.config.actor_rollout_ref.rollout
        sampling_params = build_agent_loop_sampling_params(config, validate=validate)

        if "agent_name" not in batch:
            batch["agent_name"] = NonTensorData(config.agent.default_agent_loop)

        trajectory_info = await get_trajectory_info(batch["global_steps"], batch["index"], validate)
        for i in range(len(batch)):
            prompt = {}
            for key, value in batch.items():
                if isinstance(value, torch.Tensor):
                    prompt[key] = value[i]
                elif isinstance(value, NonTensorStack):
                    prompt[key] = value[i].data
                elif isinstance(value, NonTensorData):
                    prompt[key] = value.data
                else:
                    raise TypeError(f"unsupported prompt field {key!r}: {type(value)!r}")

            task = asyncio.create_task(
                self._run_prompt(prompt, sampling_params, trajectory=trajectory_info[i])
            )
            self.background_tasks.add(task)
            task.add_done_callback(self.background_tasks.discard)
            uid = prompt["uid"]
            start_step = int(prompt["global_steps"])
            partition_id = "val" if trajectory_info[i]["validate"] else "train"
            self.prompt_tasks[uid] = (task, start_step, partition_id)

            def forget(done_task, *, prompt_uid=uid):
                current = self.prompt_tasks.get(prompt_uid)
                if current is not None and current[0] is done_task:
                    self.prompt_tasks.pop(prompt_uid, None)

            task.add_done_callback(forget)

    async def cancel_stale(self, current_step: int, max_inflight_steps: int) -> list[str]:
        """Cancel training prompt groups that have crossed the configured version span."""
        if max_inflight_steps <= 0:
            return []
        stale = [
            (uid, task)
            for uid, (task, start_step, partition_id) in list(self.prompt_tasks.items())
            if inflight_is_stale(
                current_step=current_step,
                start_step=start_step,
                max_inflight_steps=max_inflight_steps,
                partition_id=partition_id,
            )
        ]
        for _, task in stale:
            task.cancel()
        if stale:
            await asyncio.gather(*(task for _, task in stale), return_exceptions=True)
        return [str(uid) for uid, _ in stale]

    async def cancel_all(self) -> int:
        """Settle outstanding prompt tasks before the owning trainer closes TQ."""
        tasks = list({task for task, _, _ in self.prompt_tasks.values() if not task.done()})
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        return len(tasks)

    async def _run_prompt(self, prompt: dict, sampling_params: dict, trajectory: dict) -> None:
        """Run all sessions for a prompt and publish one atomic terminal status."""
        uid = prompt["uid"]
        partition_id = "val" if trajectory["validate"] else "train"
        tasks: list[asyncio.Task[Any]] = []
        try:
            await tq.async_kv_put(key=uid, partition_id=partition_id, tag={"status": "running"})
            config = self.config.actor_rollout_ref.rollout
            n = prompt.pop("__rollout_n__", config.n if not trajectory["validate"] else config.val_kwargs.n)
            do_sample = prompt.pop("__do_sample__", True)
            run_sampling_params = dict(sampling_params)
            if not trajectory["validate"] and not do_sample:
                apply_greedy_sampling_params(run_sampling_params)

            for session_id in range(n):
                # The VAGEN loop consumes the legacy names; publish both spellings so
                # every output also remains self-describing outside the TQ key.
                session_prompt = dict(prompt)
                session_prompt.update(
                    session_id=session_id,
                    group_idx=uid,
                    traj_idx=session_id,
                )
                session_prompt.setdefault(ROLLOUT_SOURCE, trajectory["sample_index"])
                tasks.append(
                    asyncio.create_task(
                        self._run_agent_loop(
                            run_sampling_params,
                            trajectory=trajectory,
                            trace=False,
                            **session_prompt,
                        )
                    )
                )

            errors = await _settle(tasks)
            if errors:
                for error in errors:
                    logger.error(
                        "VAGEN rollout failed for uid=%s",
                        uid,
                        exc_info=(type(error), error, error.__traceback__),
                    )
                status = "failure"
            else:
                status = "finished"
            await tq.async_kv_put(key=uid, partition_id=partition_id, tag={"status": status})
        except asyncio.CancelledError:
            for task in tasks:
                task.cancel()
            if tasks:
                await _settle(tasks)
            await tq.async_kv_put(key=uid, partition_id=partition_id, tag={"status": "failure"})
        except Exception as exc:  # noqa: BLE001 - the group must transition to a terminal state
            logger.exception("VAGEN rollout group failed for uid=%s: %s", uid, exc)
            if tasks:
                await _settle(tasks)
            await tq.async_kv_put(key=uid, partition_id=partition_id, tag={"status": "failure"})

    async def _agent_loop_postprocess(
        self,
        output: AgentLoopOutput | list[AgentLoopOutput],
        validate: bool,
        **kwargs,
    ) -> None:
        """Write every conversation row without changing its reward semantics."""
        uid, session_id = kwargs["uid"], kwargs["session_id"]
        outputs = output if isinstance(output, list) else [output]
        if not outputs:
            # Returning normally would let _run_prompt mark the group ``finished`` even
            # though it has no materializable trajectories.  Raising makes the whole UID
            # group fail atomically, which ReplayBufferAsync evicts and replaces.
            raise EmptyAgentLoopOutput(f"empty output for prompt {uid}_{session_id}")

        had_row_scores = [item.reward_score is not None for item in outputs]
        await self._compute_score(outputs, kwargs=kwargs)

        final_output = outputs[-1]
        await self._compute_teacher_logprobs(
            final_output,
            prompt_ids=final_output.prompt_ids,
            response_ids=final_output.response_ids,
            validate=validate,
            sample_kwargs=kwargs,
        )

        # Preserve the upstream streaming-RM behaviour when none of the rows scored
        # itself.  VAGEN envs do score each row; copying the final row over those values
        # would erase token rewards from every earlier conversation.
        if not any(had_row_scores) and final_output.reward_score is not None:
            final_reward_info = final_output.extra_fields.get("reward_extra_info", {})
            for item in outputs[:-1]:
                item.reward_score = final_output.reward_score
                item.extra_fields["reward_extra_info"] = final_reward_info

        keys, fields, tags = [], [], []
        for index, item in enumerate(outputs):
            prompts = torch.tensor(item.prompt_ids, dtype=torch.int64)
            responses = torch.tensor(item.response_ids, dtype=torch.int64)
            input_ids = torch.cat([prompts, responses], dim=0)
            attention_mask = torch.ones_like(input_ids, dtype=torch.int64)
            multi_modal_inputs = self._compute_multi_modal_inputs(item, input_ids)
            position_ids = self._compute_position_ids(
                input_ids.unsqueeze(0),
                attention_mask.unsqueeze(0),
                multi_modal_inputs,
                item.mm_processor_kwargs
                if item.mm_processor_kwargs is not None
                else self._get_mm_processor_kwargs(
                    item.multi_modal_data.get("audios") if item.multi_modal_data else None
                ),
            ).squeeze(0)

            field = item.as_dict()
            field.update(kwargs)
            field.pop("multi_modal_data", None)

            token_rewards = token_level_reward_tensor(item)
            if token_rewards is not None:
                field["rm_scores"] = token_rewards

            group_idx, traj_idx = trajectory_identity(field)
            field["group_idx"] = group_idx
            field["traj_idx"] = traj_idx
            field["loss_mask"] = field["response_mask"]
            field["input_ids"] = input_ids
            field["position_ids"] = position_ids
            field["multi_modal_inputs"] = multi_modal_inputs

            key = f"{uid}_{session_id}_{index}"
            keys.append(key)
            fields.append(field)
            min_global_steps = field["extra_fields"].get("min_global_steps")
            max_global_steps = field["extra_fields"].get("max_global_steps")
            if min_global_steps is None:
                min_global_steps = kwargs["global_steps"]
            if max_global_steps is None:
                max_global_steps = kwargs["global_steps"]
            tags.append(
                {
                    "status": "success",
                    "prompt_len": int(prompts.size(0)),
                    "response_len": int(responses.size(0)),
                    "seq_len": int(input_ids.size(0)),
                    "global_steps": kwargs["global_steps"],
                    "min_global_steps": min_global_steps,
                    "max_global_steps": max_global_steps,
                }
            )

        await tq.async_kv_batch_put(
            keys=keys,
            fields=list_of_dict_to_tensordict(fields),
            tags=tags,
            partition_id="val" if validate else "train",
        )


class VagenAgentLoopManagerTQ(AgentLoopManager):
    """Fire-and-forget manager backed by :class:`VagenAgentLoopWorkerTQ`."""

    def __init__(self, *args, **kwargs):
        self.agent_loop_workers_class = VagenAgentLoopWorkerTQ
        super().__init__(*args, **kwargs)

    @classmethod
    @auto_await
    async def create(cls, *args, **kwargs):
        instance = cls(*args, **kwargs)
        await instance._init_agent_loop_workers()
        return instance

    def generate_sequences(self, prompts: TensorDict) -> None:
        chunks = prompts.chunk(len(self.agent_loop_workers))
        ray.get(
            [
                worker.generate_sequences.remote(chunk)
                for worker, chunk in zip(self.agent_loop_workers, chunks, strict=False)
            ]
        )

    def cancel_stale(self, current_step: int, max_inflight_steps: int) -> int:
        """Cooperatively cancel over-age training groups on every loop worker."""
        expired = ray.get(
            [
                worker.cancel_stale.remote(current_step, max_inflight_steps)
                for worker in self.agent_loop_workers
            ]
        )
        return sum(len(uids) for uids in expired)

    def cancel_all(self) -> int:
        """Cancel and settle every outstanding prompt on every loop worker."""
        return sum(ray.get([worker.cancel_all.remote() for worker in self.agent_loop_workers]))


__all__ = ["EmptyAgentLoopOutput", "VagenAgentLoopManagerTQ", "VagenAgentLoopWorkerTQ"]
