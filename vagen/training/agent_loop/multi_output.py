"""VERL multi-output loops: one rollout produces *several* training rows.

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

import inspect
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

#: Which input rollout each output row came from, as an index into the batch the trainer
#: dispatched. Stamped on the batch *before* it is chunked across workers, so it rides
#: through ``input_non_tensor_batch`` and gets expanded per row by the same
#: ``np.repeat(val, counts)`` that expands every other per-rollout column.
#:
#: ★ This column is what stops the trainer from silently discarding rows. ``_fit_generate``
#: assumes one output row per input row; when a harness splits an episode across rows it
#: has to expand its own per-rollout columns to match, and it cannot work out the mapping
#: on its own -- by the time it sees the batch the workers' outputs have been
#: concatenated and the chunk boundaries are gone. Deriving it here rather than
#: reconstructing it there also handles a rollout that produced *no* rows: its index
#: simply never appears, so the trainer drops it instead of misaligning everything after
#: it. See ``SeparateRayPPOTrainer._align_generated_rows`` for the consuming end.
ROLLOUT_SOURCE = "__vagen_rollout_index__"


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
                f"all {len(groups)} rollouts returned zero outputs, so there is nothing "
                f"to train on. Read the per-episode warnings above before suspecting the "
                f"agent loop: the common cause is a trainer.compact_budget too small to "
                f"hold a summary plus one generation, which closes every conversation "
                f"after one turn and raises CompactionMakesNoProgress on every episode. "
                f"The static checks cannot catch that -- the threshold depends on the "
                f"system prompt, which they cannot see. Otherwise, check that the agent "
                f"loop appends an AgentLoopOutput per turn."
            )

        # How many rows each rollout produced is the one number that says whether the
        # split layout is doing anything, and nothing downstream reveals it: every row
        # reports num_turns=1 by construction, and the batch dimensions look the same
        # as a single-turn run. Cheap enough to always emit -- one line per worker step.
        spread = np.bincount(np.asarray(counts))
        print(
            f"[vagen] rollout -> rows: {len(groups)} rollouts produced {len(flat)} rows "
            f"(counts {dict(enumerate(spread.tolist()))} as rows->rollouts)"
        )

        expanded = None
        if input_non_tensor_batch:
            # np.repeat raises if a column's length disagrees with the rollout count,
            # which is the loud failure we want if the two ever drift apart.
            expanded = {key: np.repeat(val, counts, axis=0) for key, val in input_non_tensor_batch.items()}

        output = super()._postprocess(flat, input_non_tensor_batch=expanded, validate=validate)
        before = set(output.non_tensor_batch)
        output = self._vagen_restore_indices(output, expanded)
        output = self._vagen_restore_row_columns(output, flat)
        # What the batch leaves here carrying. Everything downstream depends on these
        # surviving, and their absence is silent -- the episode log just groups every
        # row on its own and reports one turn each.
        want = (*self.INDEX_COLUMNS, *self.ROW_COLUMNS)
        print(
            f"[vagen] postprocess(validate={validate}) rows={len(flat)} "
            + " ".join(
                f"{k}={'y' if k in output.non_tensor_batch else 'N'}"
                f"{'*' if k in output.non_tensor_batch and k not in before else ''}"
                f"u{len({str(v) for v in output.non_tensor_batch[k]}) if k in output.non_tensor_batch else 0}"
                for k in want
            )
        )
        return output

    # Columns the trajectory estimators group on. The no-concat loop happens to emit
    # them per turn via extra_fields, but the concat loop does not, and verl drops
    # `input_non_tensor_batch` entirely when streaming reward is enabled -- so relying
    # on either route makes the estimators work under one layout and not the other.
    INDEX_COLUMNS = ("group_idx", "traj_idx")

    # Per-row facts only the loop knows, restored from the outputs themselves. These
    # cannot come from `input_non_tensor_batch`: that is per rollout, and these differ
    # between the rows of one rollout. Losing turn_idx does not fail -- it makes the
    # episode log sort every turn equal, so a transcript reads as a coherent episode
    # that never happened.
    ROW_COLUMNS = ("episode_id", "turn_idx", "conversation_id", "episode_turns",
                   "response_spans", "ends_with_summary")

    def _vagen_restore_indices(self, output: DataProto, expanded: dict[str, Any] | None) -> DataProto:
        """Put the trajectory index columns back if verl dropped them.

        ★ verl drops `input_non_tensor_batch` *wholesale* when streaming reward is on
        (`agent_loop.py`: `if self.reward_loop_worker_handles is None and ...`), and VAGEN
        scores inside the environment, so `use_rm` is False, the handles exist, and that
        branch is the normal one. Every per-rollout column arrives here already deleted --
        including ROLLOUT_SOURCE, without which the trainer cannot align the rows it just
        produced and refuses the whole step.
        """
        if not expanded:
            return output
        for key in (*self.INDEX_COLUMNS, ROLLOUT_SOURCE):
            if key not in output.non_tensor_batch and key in expanded:
                output.non_tensor_batch[key] = expanded[key]
        return output

    def _vagen_restore_row_columns(self, output: DataProto, flat: list) -> DataProto:
        """Put back the per-row columns, reading them off the outputs."""
        for key in self.ROW_COLUMNS:
            if key in output.non_tensor_batch:
                continue
            values = [getattr(o, "extra_fields", {}).get(key) for o in flat]
            if all(v is None for v in values):
                continue
            arr = np.empty(len(flat), dtype=object)
            arr[:] = values
            output.non_tensor_batch[key] = arr
        return output


class MultiOutputAgentLoopManager(AgentLoopManager):
    """Runs :class:`MultiOutputAgentLoopWorker` instead of verl's ``AgentLoopWorker``.

    Selected via config, no entrypoint fork needed::

        actor_rollout_ref.rollout.agent.agent_loop_manager_class=\\
            vagen.training.agent_loop.multi_output.MultiOutputAgentLoopManager
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

        Validation additionally gets the per-turn rows folded back into one row per
        trajectory -- see :meth:`_vagen_merge_for_validation`.

        Left as a plain ``def``: the base is decorated with ``auto_await``, so it
        returns a result to sync callers and an awaitable to async ones. Both are
        handled rather than assumed.
        """
        self._vagen_assign_indices(prompts)
        validating = bool(prompts.meta_info.get("validate", False))
        result = super().generate_sequences(prompts)

        if not validating:
            return result
        if inspect.isawaitable(result):

            async def _merged():
                return self._vagen_merge_for_validation(await result, prompts)

            return _merged()
        return self._vagen_merge_for_validation(result, prompts)

    def _vagen_assign_indices(self, prompts: DataProto) -> None:
        from vagen.training.trainer.logic import traj_idx_for_interleaved_repeat

        uid = prompts.non_tensor_batch["uid"]
        prompts.non_tensor_batch["group_idx"] = uid
        n = self.rollout_config.val_kwargs.n if prompts.meta_info.get("validate") else self.rollout_config.n
        prompts.non_tensor_batch["traj_idx"] = traj_idx_for_interleaved_repeat(len(uid), n)
        # Stamped here because here is the last point where a row's position *is* its
        # identity: the base's next act is `prompts.chunk(...)`, after which a worker
        # knows only its own slice. Riding out through `input_non_tensor_batch` costs
        # nothing -- the worker already repeats every such column by its row counts.
        prompts.non_tensor_batch[ROLLOUT_SOURCE] = np.arange(len(uid), dtype=np.int64)

    def _vagen_merge_for_validation(self, output: DataProto, prompts: DataProto) -> DataProto:
        """Fold each trajectory's per-turn rows back into a single row.

        Validation, unlike training, wants whole episodes: verl unpads the generated
        batch positionally against the padded input and then unions the two, both of
        which require one output row per input row. Merging here keeps that contract,
        so ``_validate`` needs no changes to work with the split layout.
        """
        from vagen.utils.concat_val_multi_turn import concat_val_multi_turn

        return concat_val_multi_turn(output, prompts, self._vagen_tokenizer(), self._vagen_processor())

    def _vagen_processor(self):
        """The processor, which is what declares the image placeholder ids.

        Cached alongside the tokenizer. Absent for a text-only model, and then there are
        no placeholders to find either.
        """
        if getattr(self, "_vagen_processor_cache", "unset") == "unset":
            from verl.utils import hf_processor
            from verl.utils.fs import copy_to_local

            model_cfg = self.config.actor_rollout_ref.model
            path = model_cfg.get("processor_path") or model_cfg.path
            try:
                self._vagen_processor_cache = hf_processor(
                    copy_to_local(path), trust_remote_code=model_cfg.get("trust_remote_code", False)
                )
            except Exception:  # noqa: BLE001 - a text-only model has none
                self._vagen_processor_cache = None
        return self._vagen_processor_cache

    def _vagen_tokenizer(self):
        """Cached; building it re-reads the model directory."""
        if getattr(self, "_vagen_tokenizer_cache", None) is None:
            from verl.utils import hf_tokenizer
            from verl.utils.fs import copy_to_local

            model_cfg = self.config.actor_rollout_ref.model
            path = model_cfg.get("tokenizer_path") or model_cfg.path
            self._vagen_tokenizer_cache = hf_tokenizer(
                copy_to_local(path), trust_remote_code=model_cfg.get("trust_remote_code", False)
            )
        return self._vagen_tokenizer_cache
