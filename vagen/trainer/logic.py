# All comments are in English.
"""Pure trainer-side logic, extracted from the vendored ``vagen/ray_trainer.py``.

Nothing here imports verl, ray or hydra: every function takes plain tensors or a
registry dict. That is deliberate -- it is the layer we want unit-testable without a
cluster, and the layer that survives when the verl trainer underneath is swapped
(``SeparateRayPPOTrainer`` today, verl main's V1 ``PPOTrainer`` later).

``vagen/trainer/mixin.py`` is the thin adapter that binds these to verl's ``_fit_*``
hooks and unwraps ``DataProto``.
"""

from __future__ import annotations

from typing import Any, Callable, Mapping

import torch

# Sentinel written into `returns` at positions that carry no value supervision.
# Turn-level / bi-level GAE writes a real return at one anchor token per turn and
# leaves everything else at this value; mirrors CrossEntropyLoss's ignore_index.
IGNORE_RETURN = -100.0


def default_eps(dtype: torch.dtype, small_eps: float = 1e-2, large_eps: float = 1e-6) -> float:
    """Comparison tolerance for spotting the sentinel, scaled to the dtype.

    bf16/fp16 cannot represent -100.0 exactly after a round trip through the batch
    machinery, hence the far looser tolerance there.
    """
    return small_eps if dtype in (torch.float16, torch.bfloat16) else large_eps


def value_mask_from_returns(
    returns: torch.Tensor,
    response_mask: torch.Tensor,
    ignore_value: float = IGNORE_RETURN,
    eps: float | None = None,
) -> torch.Tensor:
    """Positions where the critic actually has a return to regress towards.

    Consumed by verl's patched ``workers/utils/losses.py::value_loss``, which ANDs it
    into ``response_mask``. Returned in ``response_mask``'s dtype so the two compose
    without a cast.

    Note this only looks at ``returns``; it does not itself exclude padding. Padding is
    already handled by ``response_mask``, and the AND happens downstream.
    """
    if eps is None:
        eps = default_eps(returns.dtype)
    is_ignored = (returns - ignore_value).abs() < eps
    return (~is_ignored).to(dtype=response_mask.dtype)


def collect_registry_metrics(
    registry: Mapping[str, Callable[[Any], float]],
    data: Any,
    prefix: str = "custom_metrics",
    strict: bool = False,
) -> dict[str, float]:
    """Run every metric in ``registry`` over ``data``.

    A misbehaving metric must not take down a training step, so by default failures
    are caught. But a metric that silently vanishes from the dashboard is its own kind
    of bug, so each failure also emits ``{prefix}/_failed/{name} = 1.0`` -- visible in
    wandb rather than only in stdout, which is what the original implementation did.

    Set ``strict=True`` in tests to turn failures back into exceptions.
    """
    out: dict[str, float] = {}
    for name, fn in registry.items():
        try:
            out[f"{prefix}/{name}"] = fn(data)
        except Exception as exc:  # noqa: BLE001 - one bad metric must not kill the step
            if strict:
                raise
            print(f"[vagen] custom metric {name!r} failed: {type(exc).__name__}: {exc}")
            out[f"{prefix}/_failed/{name}"] = 1.0
    return out


def kl_penalty_term(kld: torch.Tensor, beta: float) -> torch.Tensor:
    """The signed KL penalty actually subtracted from token-level scores.

    verl folds this into ``token_level_rewards`` and never surfaces it; VAGEN logs it
    separately so the reward and the penalty can be read apart on the dashboard.
    """
    return -beta * kld


def pad_to_multiple(size: int, multiple: int) -> int:
    """Number of rows to append so ``size`` becomes divisible by ``multiple``.

    In no-concat mode one prompt expands into a variable number of samples, so the
    batch is no longer a clean multiple of the DP world size and has to be padded
    before ``_balance_batch``.
    """
    if multiple <= 0:
        raise ValueError(f"multiple must be positive, got {multiple}")
    return (-size) % multiple
