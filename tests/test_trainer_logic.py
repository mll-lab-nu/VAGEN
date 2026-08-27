"""Unit tests for vagen/training/trainer/logic.py.

Pure tensor logic -- no verl, no ray, no GPU. Should run in well under a second.
"""

import pytest
import torch

from vagen.training.trainer.logic import (
    IGNORE_RETURN,
    collect_registry_metrics,
    default_eps,
    kl_penalty_term,
    pad_to_multiple,
    value_mask_from_returns,
)


# --------------------------------------------------------------------------- eps


def test_default_eps_is_looser_for_half_precision():
    """bf16/fp16 cannot round-trip -100.0 exactly, so the sentinel needs slack."""
    assert default_eps(torch.bfloat16) > default_eps(torch.float32)
    assert default_eps(torch.float16) == default_eps(torch.bfloat16)


# -------------------------------------------------------------------- value mask


def test_value_mask_marks_only_non_sentinel_positions():
    returns = torch.tensor([[1.0, IGNORE_RETURN, 3.0], [IGNORE_RETURN, IGNORE_RETURN, 0.0]])
    resp = torch.ones(2, 3, dtype=torch.long)

    mask = value_mask_from_returns(returns, resp)

    assert mask.tolist() == [[1, 0, 1], [0, 0, 1]]


def test_value_mask_follows_response_mask_dtype():
    """It gets ANDed with response_mask downstream; matching dtype avoids a cast."""
    returns = torch.zeros(1, 3)
    for dtype in (torch.long, torch.bool, torch.float32):
        assert value_mask_from_returns(returns, torch.ones(1, 3, dtype=dtype)).dtype == dtype


def test_value_mask_zero_return_is_supervised_not_sentinel():
    """A genuine return of 0.0 must not be mistaken for the sentinel."""
    returns = torch.tensor([[0.0, IGNORE_RETURN]])
    mask = value_mask_from_returns(returns, torch.ones(1, 2, dtype=torch.long))

    assert mask.tolist() == [[1, 0]]


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_sentinel_is_exactly_representable(dtype):
    """★ This is the invariant the whole scheme rests on, and it is why the
    dtype-dependent tolerance is *not* actually load-bearing today.

    -100.0 round-trips exactly through fp32/fp16/bf16, so sentinel detection is exact.
    The tolerance cannot in fact rescue a drifted sentinel: near -100 the bf16 ulp is
    0.5, fifty times `default_eps(bfloat16) == 1e-2`. It is kept only as cheap
    insurance for a future IGNORE_RETURN that is *not* exactly representable.

    If someone changes IGNORE_RETURN to such a value, this test fails first and
    points at the right place -- rather than the mask silently going wrong in bf16.
    """
    assert torch.tensor([IGNORE_RETURN], dtype=dtype).item() == IGNORE_RETURN

    returns = torch.tensor([[IGNORE_RETURN, 5.0]], dtype=dtype)
    mask = value_mask_from_returns(returns, torch.ones(1, 2, dtype=torch.long))
    assert mask.tolist() == [[0, 1]]


def test_tolerance_is_narrower_than_one_bf16_ulp_near_the_sentinel():
    """Documents the limitation rather than pretending it away: a sentinel perturbed
    by a single bf16 ulp (0.5) is NOT recognised. Detection is exact-match in
    practice; do not rely on the tolerance for robustness."""
    one_ulp_off = torch.tensor([[IGNORE_RETURN + 0.5]], dtype=torch.bfloat16)
    mask = value_mask_from_returns(one_ulp_off, torch.ones(1, 1, dtype=torch.long))

    assert mask.tolist() == [[1]], "a one-ulp drift reads as a real return, by design"


def test_value_mask_all_sentinel_is_allowed():
    """Degenerate but legal (e.g. a padding row). The empty-mask consequence is
    verl's, not ours -- see tests/workers/test_vagen_value_mask_on_cpu.py in the verl
    fork."""
    returns = torch.full((2, 4), IGNORE_RETURN)
    assert value_mask_from_returns(returns, torch.ones(2, 4, dtype=torch.long)).sum() == 0


# ------------------------------------------------------------------- metrics


def test_collect_registry_metrics_prefixes_names():
    out = collect_registry_metrics({"a": lambda d: 1.0, "b": lambda d: 2.0}, None, prefix="m")
    assert out == {"m/a": 1.0, "m/b": 2.0}


def test_failing_metric_does_not_kill_the_step_but_is_visible():
    """A broken metric must not abort training -- but it must not vanish silently
    either, which printing alone would allow."""

    def boom(_):
        raise RuntimeError("nope")

    out = collect_registry_metrics({"ok": lambda d: 1.0, "bad": boom}, None, prefix="m")

    assert out["m/ok"] == 1.0
    assert out["m/_failed/bad"] == 1.0, "failures must surface as a metric, not just stdout"
    assert "m/bad" not in out


def test_strict_mode_reraises():
    def boom(_):
        raise RuntimeError("nope")

    with pytest.raises(RuntimeError):
        collect_registry_metrics({"bad": boom}, None, strict=True)


def test_empty_registry_is_a_no_op():
    assert collect_registry_metrics({}, None) == {}


# ------------------------------------------------------------------- misc


def test_kl_penalty_term_is_negative_beta_times_kld():
    kld = torch.tensor([[1.0, -2.0]])
    torch.testing.assert_close(kl_penalty_term(kld, 0.5), torch.tensor([[-0.5, 1.0]]))


@pytest.mark.parametrize(
    "size,multiple,expected",
    [(10, 4, 2), (12, 4, 0), (1, 8, 7), (0, 4, 0), (9, 1, 0)],
)
def test_pad_to_multiple(size, multiple, expected):
    assert pad_to_multiple(size, multiple) == expected
    assert (size + pad_to_multiple(size, multiple)) % multiple == 0


def test_pad_to_multiple_rejects_nonpositive():
    with pytest.raises(ValueError):
        pad_to_multiple(10, 0)


# ------------------------------------------------------- no-concat index bookkeeping

import numpy as np

from vagen.training.trainer.logic import traj_idx_for_interleaved_repeat


def test_traj_idx_cycles_within_each_prompt_group():
    """★ interleave=True lays rows out [A A A B B B], so the index cycles rather than
    running 0..n-1 across the batch. Getting this backwards silently merges the returns
    of unrelated trajectories instead of raising."""
    assert traj_idx_for_interleaved_repeat(9, 3).tolist() == [0, 1, 2, 0, 1, 2, 0, 1, 2]


def test_traj_idx_with_a_single_trajectory_is_all_zeros():
    assert traj_idx_for_interleaved_repeat(4, 1).tolist() == [0, 0, 0, 0]


def test_traj_idx_rejects_a_ragged_batch():
    """np.tile would return a short array and the assignment would misalign."""
    with pytest.raises(ValueError, match="whole number of groups"):
        traj_idx_for_interleaved_repeat(10, 3)


def test_traj_idx_rejects_nonpositive_repeat():
    with pytest.raises(ValueError):
        traj_idx_for_interleaved_repeat(6, 0)

