"""One rollout, several training rows: the rows have to survive into the batch.

``_fit_generate`` was written for one output row per input row. It sized the generated
batch with ``slice(0, num_sampled_prompts)`` -- the *input* count -- so a harness that
splits an episode across rows had everything past the first few rollouts thrown away.
Measured on real runs: 8 of ~155 rows kept under ``no_concat``, 8 of ~40 under
``compact``. Nothing failed, because after the slice every shape is self-consistent and
the batch looks exactly like an ordinary single-turn one.

These tests pin the alignment and, just as importantly, pin that the *absence* of the
alignment column is now an error rather than a silent truncation.
"""

from __future__ import annotations

import types

import numpy as np
import pytest
import torch
from verl import DataProto
from verl.experimental.separation.ray_trainer import SeparateRayPPOTrainer

from vagen.training.agent_loop.multi_output import ROLLOUT_SOURCE

ALIGN = SeparateRayPPOTrainer._align_generated_rows
SOURCE = SeparateRayPPOTrainer.ROLLOUT_SOURCE_COLUMN


def _trainer(estimator="token_level_gae"):
    """The two attributes `_align_generated_rows` actually reads."""
    return types.SimpleNamespace(
        config=types.SimpleNamespace(algorithm=types.SimpleNamespace(adv_estimator=estimator)),
        ROLLOUT_SOURCE_COLUMN=SOURCE,
    )


def _gen_output(source=None, n=None, uid=None):
    """A generated batch: `n` rows, optionally stamped with their source rollout."""
    n = len(source) if source is not None else n
    columns = {
        "responses": torch.arange(n * 3).reshape(n, 3),
        "uid": np.array(uid if uid is not None else [f"u{i}" for i in range(n)], dtype=object),
    }
    if source is not None:
        columns[SOURCE] = np.asarray(source, dtype=np.int64)
    return DataProto.from_single_dict(columns)


def _prompt_batch(n, uid=None):
    """What the trainer holds: no tensors of its own, just the per-rollout columns."""
    return DataProto.from_single_dict(
        {
            "placeholder": torch.zeros(n, 1),
            "uid": np.array(uid if uid is not None else [f"u{i}" for i in range(n)], dtype=object),
        }
    )


# ------------------------------------------------------------------ the split layouts


def test_every_generated_row_survives():
    """★ The regression. 4 rollouts producing 2/1/3/1 rows must yield 7 training rows."""
    source = [0, 0, 1, 2, 2, 2, 3]
    out, index = ALIGN(_trainer(), _gen_output(source), num_sampled_prompts=4)

    assert len(out) == 7, "rows were dropped"
    assert index.tolist() == source


def test_the_batch_is_expanded_to_match_and_unions_cleanly():
    """The other half: the trainer's own per-rollout columns have to be stretched the
    same way, or `union` compares a 4-row column against a 7-row one."""
    source = [0, 0, 1, 2, 2, 2, 3]
    uid_per_rollout = ["a", "b", "c", "d"]
    gen = _gen_output(source, uid=[uid_per_rollout[i] for i in source])

    out, index = ALIGN(_trainer(), gen, num_sampled_prompts=4)
    batch = _prompt_batch(4, uid=uid_per_rollout).select_idxs(index)

    assert len(batch) == 7
    assert list(batch.non_tensor_batch["uid"]) == ["a", "a", "b", "c", "c", "c", "d"]
    merged = batch.union(out)
    assert len(merged) == 7
    # ★ Each row's uid is its own episode's, not a positional coincidence.
    assert list(merged.non_tensor_batch["uid"]) == ["a", "a", "b", "c", "c", "c", "d"]


def test_a_rollout_that_produced_nothing_is_dropped_not_shifted():
    """An unusable episode returns no rows. Its prompt has to leave the batch with it --
    otherwise every later row is paired with the wrong prompt. This case used to abort
    the step with a bare size assertion from `union`."""
    source = [0, 0, 2, 3]  # rollout 1 produced nothing
    uid_per_rollout = ["a", "b", "c", "d"]
    gen = _gen_output(source, uid=[uid_per_rollout[i] for i in source])

    out, index = ALIGN(_trainer(), gen, num_sampled_prompts=4)
    batch = _prompt_batch(4, uid=uid_per_rollout).select_idxs(index)

    assert list(batch.non_tensor_batch["uid"]) == ["a", "a", "c", "d"], "b must be gone, not shifted"
    assert len(batch.union(out)) == 4


# ------------------------------------------------------- the one-row-per-rollout case


def test_concat_is_untouched():
    """One row per rollout: no column, counts already agree, behaviour identical to
    before this method existed -- including returning None so the caller skips the
    expansion entirely."""
    gen = _gen_output(n=4)
    out, index = ALIGN(_trainer(), gen, num_sampled_prompts=4)

    assert index is None
    assert len(out) == 4
    assert list(out.non_tensor_batch["uid"]) == ["u0", "u1", "u2", "u3"]


def test_an_unclaimed_row_count_mismatch_is_an_error_now():
    """★ The silent failure, made loud. A loop that emits several rows but does not stamp
    the column would otherwise have them dropped without a word."""
    with pytest.raises(ValueError, match="rows for 4 prompts"):
        ALIGN(_trainer(), _gen_output(n=7), num_sampled_prompts=4)


def test_too_few_rows_is_also_an_error():
    with pytest.raises(ValueError, match="rows for 4 prompts"):
        ALIGN(_trainer(), _gen_output(n=3), num_sampled_prompts=4)


# --------------------------------------------------------------------- bad index data


def test_out_of_range_source_is_rejected():
    with pytest.raises(ValueError, match="ranges over"):
        ALIGN(_trainer(), _gen_output([0, 1, 9]), num_sampled_prompts=4)


def test_reordered_workers_are_rejected():
    """Alignment relies on chunks being contiguous and concatenated in worker order. If
    that ever stops holding, the indices arrive out of order and matching rows to prompts
    is no longer possible -- which must fail rather than mis-pair them."""
    with pytest.raises(ValueError, match="non-decreasing"):
        ALIGN(_trainer(), _gen_output([0, 2, 1]), num_sampled_prompts=4)


def test_remax_is_refused_rather_than_mis_split():
    """REMAX slices the generated batch positionally into a sampled and a greedy half,
    which multi-row output invalidates."""
    with pytest.raises(ValueError, match="REMAX"):
        ALIGN(_trainer("remax"), _gen_output([0, 0, 1]), num_sampled_prompts=2)


# ------------------------------------------------------- the column reaches the batch


class _MergeTrainer:
    """Borrows the three methods under test; supplies only what they read."""

    _merge_generated_rows = SeparateRayPPOTrainer._merge_generated_rows
    _rows_divisor = SeparateRayPPOTrainer._rows_divisor
    _pad_rows_to_divisor = SeparateRayPPOTrainer._pad_rows_to_divisor

    def __init__(self, rollout_n=1, world_size=1, mini=1, use_critic=False):
        self.config = types.SimpleNamespace(
            actor_rollout_ref=types.SimpleNamespace(
                rollout=types.SimpleNamespace(n=rollout_n),
                actor=types.SimpleNamespace(ppo_mini_batch_size=mini),
            ),
            critic=types.SimpleNamespace(ppo_mini_batch_size=mini),
        )
        self.actor_rollout_wg = types.SimpleNamespace(world_size=world_size)
        self.use_critic = use_critic


def test_the_index_is_applied_not_merely_computed():
    """★ Everything above checks that the right index is *computed*. Disabling the line
    that applies it left all of them green -- the index was correct and unused, and the
    rows went straight back to being dropped. This exercises the application itself."""
    source = [0, 0, 1, 2, 2, 2, 3]
    uid_per_rollout = ["a", "b", "c", "d"]
    gen = _gen_output(source, uid=[uid_per_rollout[i] for i in source])
    _, index = ALIGN(_trainer(), gen, num_sampled_prompts=4)

    merged = _MergeTrainer()._merge_generated_rows(
        _prompt_batch(4, uid=uid_per_rollout), gen, index)

    assert len(merged) == 7, "the generated rows did not survive the merge"
    assert list(merged.non_tensor_batch["uid"]) == ["a", "a", "b", "c", "c", "c", "d"]


def test_the_merge_is_a_plain_repeat_when_there_is_no_index():
    """One row per rollout, rollout.n=2: the historical path, unchanged."""
    gen = _gen_output(n=4, uid=["a", "a", "b", "b"])
    merged = _MergeTrainer(rollout_n=2)._merge_generated_rows(
        _prompt_batch(2, uid=["a", "b"]), gen, None)

    assert len(merged) == 4
    assert list(merged.non_tensor_batch["uid"]) == ["a", "a", "b", "b"]


def test_the_merge_refuses_a_size_mismatch():
    """The invariant that would have caught the original bug on its first step."""
    gen = _gen_output(n=7)
    with pytest.raises(ValueError, match="rows and the rollout returned"):
        _MergeTrainer()._merge_generated_rows(_prompt_batch(4), gen, None)


# ------------------------------------------------------------------------- padding
#
# Restoring the rows is what makes the row count arbitrary. Before the fix it was always
# train_batch_size * rollout.n, so every divisor divided it by construction.


def test_a_variable_row_count_is_padded_for_the_dp_split():
    """★ 7 rows across 8 ranks: `_balance_batch` asserts `len % world_size == 0` and the
    step dies with a bare AssertionError naming neither the cause nor the harness."""
    source = [0, 0, 1, 2, 2, 2, 3]
    uid = ["a", "b", "c", "d"]
    gen = _gen_output(source, uid=[uid[i] for i in source])
    _, index = ALIGN(_trainer(), gen, num_sampled_prompts=4)

    merged = _MergeTrainer(world_size=8)._merge_generated_rows(
        _prompt_batch(4, uid=uid), gen, index)
    assert len(merged) % 8 == 0
    assert len(merged) == 8, "padded to the next multiple, not further"


def test_padding_also_satisfies_the_mini_batch_size():
    """The actor update asserts `batch_size % ppo_mini_batch_size == 0` separately, so the
    divisor is the lcm of the two -- padding for one and tripping the other would just
    move the crash."""
    source = [0, 0, 1, 2, 2, 2, 3]
    uid = ["a", "b", "c", "d"]
    gen = _gen_output(source, uid=[uid[i] for i in source])
    _, index = ALIGN(_trainer(), gen, num_sampled_prompts=4)

    merged = _MergeTrainer(world_size=4, mini=6)._merge_generated_rows(
        _prompt_batch(4, uid=uid), gen, index)
    assert len(merged) % 4 == 0 and len(merged) % 6 == 0
    assert len(merged) == 12


def test_an_already_divisible_count_is_left_alone():
    source = [0, 0, 1, 2]
    uid = ["a", "b", "c"]
    gen = _gen_output(source, uid=[uid[i] for i in source])
    _, index = ALIGN(_trainer(), gen, num_sampled_prompts=3)
    merged = _MergeTrainer(world_size=4)._merge_generated_rows(
        _prompt_batch(3, uid=uid), gen, index)
    assert len(merged) == 4


def test_the_one_row_per_rollout_path_is_never_padded():
    """★ Concat was always divisible by construction, and padding it would change a
    behaviour that has always been correct. The guard is `batch_row_index is not None`."""
    gen = _gen_output(n=3)
    merged = _MergeTrainer(world_size=8)._merge_generated_rows(_prompt_batch(3), gen, None)
    assert len(merged) == 3, "concat must not acquire padding it never needed"


def test_padding_duplicates_rows_the_estimators_deduplicate():
    """The copies are byte-identical to the rows they duplicate, which is the premise
    TrajectoryView's dedup rests on."""
    source = [0, 0, 1]
    gen = _gen_output(source, uid=["a", "a", "b"])
    _, index = ALIGN(_trainer(), gen, num_sampled_prompts=2)
    merged = _MergeTrainer(world_size=4)._merge_generated_rows(
        _prompt_batch(2, uid=["a", "b"]), gen, index)

    assert len(merged) == 4
    uid = list(merged.non_tensor_batch["uid"])
    assert uid == ["a", "a", "b", "a"], "padding copies from the front"


def test_the_loop_manager_stamps_the_source_column():
    """The index is stamped before chunking, because that is the last point where a row's
    position is its identity. Pinned here so the two ends keep the same name."""
    assert ROLLOUT_SOURCE == SOURCE

    import inspect

    from vagen.training.agent_loop.multi_output import MultiOutputAgentLoopManager

    src = inspect.getsource(MultiOutputAgentLoopManager._vagen_assign_indices)
    assert "ROLLOUT_SOURCE" in src and "np.arange" in src


def test_the_worker_expands_the_column_per_row():
    """It rides out through `input_non_tensor_batch`, which `_postprocess` repeats by the
    per-rollout row counts -- the same mechanism that expands uid and data_source."""
    import inspect

    from vagen.training.agent_loop.multi_output import MultiOutputAgentLoopWorker

    src = inspect.getsource(MultiOutputAgentLoopWorker._postprocess)
    assert "np.repeat(val, counts" in src, (
        "the source column reaches the rows only because every input column is repeated "
        "by the row counts; if that goes, alignment goes with it"
    )
