"""Two-timescale credit assignment for multi-turn trajectories.

The estimator uses token time within an action and turn time across environment
transitions. The implementation lives here, rather than in ``_common``; shared code
contains contracts, not a concrete advantage estimator.
"""

from __future__ import annotations

import math

import torch

from vagen.algorithms._common import (
    AdvantageInputs,
    advantage_estimator,
    register_algorithm,
)
from vagen.algorithms._common.packing import _pack


def _turn_boundaries(packed):
    """Mark the final valid token of each environment action."""
    return packed.boundary() & packed.valid


def _variable_clock_gae(
    packed,
    boundary: torch.Tensor,
    gamma_turn: float,
    lambda_turn: float,
    lambda_token: float,
) -> torch.Tensor:
    """One GAE recursion with token-time inside actions and turn-time across actions."""
    n_traj, max_len = packed.valid.shape
    advantages = torch.zeros_like(packed.seq_v)
    next_value = torch.zeros(
        n_traj, dtype=packed.seq_v.dtype, device=packed.seq_v.device
    )
    next_adv = torch.zeros_like(next_value)
    for i in reversed(range(max_len)):
        live = packed.valid[:, i]
        crosses_turn = boundary[:, i]
        gamma = torch.where(crosses_turn, gamma_turn, 1.0).to(packed.seq_v.dtype)
        lam = torch.where(crosses_turn, lambda_turn, lambda_token).to(
            packed.seq_v.dtype
        )
        delta = packed.seq_r[:, i] + gamma * next_value - packed.seq_v[:, i]
        current = delta + gamma * lam * next_adv
        advantages[:, i] = torch.where(live, current, torch.zeros_like(current))
        next_value = torch.where(live, packed.seq_v[:, i], next_value)
        next_adv = torch.where(live, current, next_adv)
    return advantages


@advantage_estimator("bi_level_gae", needs_critic=True, undiscounted=True)
def compute_bi_level_gae(inputs: AdvantageInputs):
    """Compute variable-clock bi-level GAE.

    ``algorithm.gamma`` is the token clock and must remain 1.0 (enforced by the trainer),
    so verbosity cannot change the effective inter-turn discount.  ``gamma_turn`` is the
    only temporal discount between environment actions.
    """
    gamma_turn = float(inputs.param("gamma_turn", 0.95))
    lambda_turn = float(inputs.param("lambda_turn", 0.95))
    lambda_token = float(inputs.param("lambda_token", 1.0))
    bi_level_mix = float(inputs.param("bi_level_mix", 0.75))
    for name, value in (
        ("gamma_turn", gamma_turn),
        ("lambda_turn", lambda_turn),
        ("lambda_token", lambda_token),
        ("bi_level_mix", bi_level_mix),
    ):
        if not math.isfinite(value) or not 0.0 <= value <= 1.0:
            raise ValueError(f"algorithm.{name} must be finite and in [0, 1], got {value}")

    with torch.no_grad():
        packed = _pack(inputs)

        # A single variable-clock return: token transitions are undiscounted, while
        # crossing an environment transition spends gamma_turn and lambda_turn once.
        # Unlike applying gamma_turn per token, this horizon does not depend on how
        # verbose the policy happened to be inside the action.
        boundary = _turn_boundaries(packed)
        bi_level_adv = _variable_clock_gae(
            packed, boundary, gamma_turn, lambda_turn, lambda_token
        )
        if bi_level_mix == 1.0:
            token_adv = bi_level_adv
        else:
            # A continuous path from ordinary token GAE to the variable-clock
            # estimator. At alpha=0 every transition uses the token clock; at
            # alpha=1 turn boundaries use gamma_turn/lambda_turn.
            token_adv_default = _variable_clock_gae(
                packed,
                boundary,
                gamma_turn=1.0,
                lambda_turn=lambda_token,
                lambda_token=lambda_token,
            )
            token_adv = token_adv_default + bi_level_mix * (
                bi_level_adv - token_adv_default
            )

        return packed.emit(
            advantages=packed.scatter(token_adv),
            returns=packed.scatter(token_adv + packed.seq_v),
        )


SPEC = register_algorithm("bi_level_gae", compute_bi_level_gae)

__all__ = ["SPEC", "compute_bi_level_gae"]
