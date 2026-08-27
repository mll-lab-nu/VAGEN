"""Token-level GAE over complete trajectories."""

import torch

from vagen.algorithms._common import AdvantageInputs, advantage_estimator, register_algorithm
from vagen.algorithms._common.packing import _backward_gae, _pack


@advantage_estimator("token_level_gae", needs_critic=True)
def compute_token_level_gae(inputs: AdvantageInputs):
    """Token-level GAE over a trajectory, across whatever rows it occupies."""
    gamma, lam = float(inputs.config.gamma), float(inputs.config.lam)

    with torch.no_grad():
        packed = _pack(inputs)
        seq_adv = _backward_gae(packed.seq_r, packed.seq_v, packed.valid, gamma, lam)
        return packed.emit(
            advantages=packed.scatter(seq_adv),
            returns=packed.scatter(seq_adv + packed.seq_v),
        )


SPEC = register_algorithm("token_level_gae", compute_token_level_gae)

__all__ = ["SPEC", "compute_token_level_gae"]
