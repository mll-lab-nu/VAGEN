"""Multi-output agent loops: one rollout produces *several* training rows.

The no-concat formulation trains on one row per environment turn rather than one row
per episode, so a single ``AgentLoopBase.run`` yields ``list[AgentLoopOutput]``. verl
assumes one output per input row.

Supporting that needs two method overrides and no copied upstream code, because of
where the fan-out is intercepted:

    _run_agent_loop:  output = await agent_loop.run(...)          # <- returns a list
                      return await self._agent_loop_postprocess(output, ...)  # <- hook 1
    generate_sequences: outputs = await asyncio.gather(*tasks)
                      return self._postprocess(outputs, ...)      # <- hook 2

Nothing between ``run()`` and ``_agent_loop_postprocess`` inspects the value, and
``generate_sequences`` passes the gathered list straight through, so the list can travel
unnoticed from the agent loop to a single flattening point. Everything else -- the
registry lookup, hydra instantiation, rollout tracing, padding, multimodal inputs,
scoring -- stays upstream's.

``group_idx`` / ``traj_idx`` / ``turn_idx`` need no handling here: the gym loop puts them
at the top level of ``extra_fields``, and verl's ``_postprocess`` turns every
``extra_fields`` key into a ``non_tensor_batch`` column. It applies them *after*
``input_non_tensor_batch``, so the per-segment values correctly shadow the per-row ones
the dataset supplied.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import ray
from verl.experimental.agent_loop.agent_loop import (
    AgentLoopManager,
    AgentLoopOutput,
    AgentLoopWorker,
    _InternalAgentLoopOutput,
)
from verl.protocol import DataProto


class MultiOutputAgentLoopWorker(AgentLoopWorker):
    """An ``AgentLoopWorker`` whose agent loops may return a list of outputs.

    Single outputs still work unchanged, so concat and no-concat agent loops can share
    one worker class and a run can mix them.
    """

    async def _agent_loop_postprocess(
        self, output: AgentLoopOutput | list[AgentLoopOutput], validate: bool, **kwargs
    ) -> _InternalAgentLoopOutput | list[_InternalAgentLoopOutput]:
        """Pad/tokenize each output of the rollout instead of assuming exactly one."""
        # Bind before the comprehension: zero-arg super() cannot resolve inside one,
        # because the comprehension's frame does not have `self` as its first local.
        base = super()._agent_loop_postprocess

        if not isinstance(output, list):
            return await base(output, validate, **kwargs)
        return [await base(item, validate, **kwargs) for item in output]

    def _postprocess(
        self,
        inputs: list[_InternalAgentLoopOutput | list[_InternalAgentLoopOutput]],
        input_non_tensor_batch: dict[str, Any] | None = None,
        validate: bool = False,
    ) -> DataProto:
        """Flatten the per-rollout groups, then let verl build the batch.

        ``input_non_tensor_batch`` is the *input* batch's columns, one entry per rollout,
        so it has to be stretched to match the flattened rows. Skipping this is not a
        quiet inconsistency: verl builds the TensorDict with ``batch_size=len(inputs)``
        and the mismatched non-tensor columns fail downstream.
        """
        groups = [group if isinstance(group, list) else [group] for group in inputs]
        counts = [len(group) for group in groups]
        flat = [output for group in groups for output in group]

        if not flat:
            # verl indexes inputs[0]; fail with the cause rather than an IndexError.
            raise RuntimeError(
                f"all {len(groups)} rollouts returned zero outputs, so there is nothing to "
                "train on -- check that the agent loop appends an AgentLoopOutput per turn"
            )

        expanded = None
        if input_non_tensor_batch:
            # np.repeat raises if a column's length disagrees with the rollout count,
            # which is the loud failure we want if the two ever drift apart.
            expanded = {key: np.repeat(val, counts, axis=0) for key, val in input_non_tensor_batch.items()}

        return super()._postprocess(flat, input_non_tensor_batch=expanded, validate=validate)


class MultiOutputAgentLoopManager(AgentLoopManager):
    """Runs :class:`MultiOutputAgentLoopWorker` instead of verl's ``AgentLoopWorker``.

    Selected via config, no entrypoint fork needed::

        actor_rollout_ref.rollout.agent.agent_loop_manager_class=\\
            vagen.agent_loop.multi_output.MultiOutputAgentLoopManager
    """

    def __init__(self, *args, **kwargs):
        # AgentLoopManager.__init__ assigns verl's own worker class under
        # `if not hasattr(...)`, an explicit subclass hook. Claiming it first is the
        # intended use and avoids building an ActorClass that is then discarded;
        # the value is only read later, by _init_agent_loop_workers.
        self.agent_loop_workers_class = ray.remote(MultiOutputAgentLoopWorker)
        super().__init__(*args, **kwargs)

    def generate_sequences(self, prompts: DataProto):
        """Label rows with their prompt group and trajectory, then generate.

        This is the only point where the batch is both already repeated and not yet
        split: ``_fit_generate`` repeats inside itself with no hook afterwards, and the
        base's next act is ``prompts.chunk(...)``. Doing it per worker instead would
        restart ``traj_idx`` at 0 whenever a chunk boundary fell inside a group.

        Left as a plain ``def``: the base is decorated with ``auto_await``, so it
        returns a result to sync callers and an awaitable to async ones, and passing
        that through unchanged keeps both working.
        """
        self._vagen_assign_indices(prompts)
        return super().generate_sequences(prompts)

    def _vagen_assign_indices(self, prompts: DataProto) -> None:
        from vagen.trainer.logic import traj_idx_for_interleaved_repeat

        uid = prompts.non_tensor_batch["uid"]
        prompts.non_tensor_batch["group_idx"] = uid
        prompts.non_tensor_batch["traj_idx"] = traj_idx_for_interleaved_repeat(
            len(uid), self.rollout_config.n
        )
