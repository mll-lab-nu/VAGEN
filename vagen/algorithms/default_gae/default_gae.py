"""Default episode-level GAE implementation."""

import torch

from vagen.algorithms._common import AdvantageInputs, advantage_estimator, register_algorithm
from vagen.algorithms._common.packing import _backward_gae, _last_valid, _pack


def _lump_rewards_at_trajectory_end(packed):
    """Move the trajectory's full reward to its final model-output token."""
    total = (packed.seq_r * packed.valid).sum(dim=1, keepdim=True)
    return torch.where(
        _last_valid(packed.valid),
        total.expand_as(packed.seq_r),
        torch.zeros_like(packed.seq_r),
    )


@advantage_estimator("default_gae", needs_critic=True)
def compute_default_gae(inputs: AdvantageInputs):
    """**The baseline the others are measured against.** Ordinary single-turn GAE.

    Identical to ``token_level_gae`` in every respect but one: before the recursion runs,
    every reward the episode collected is summed and moved to its **last** model-output
    token. That is the standard RLHF setup -- one scalar per sequence, delivered at the
    end -- applied to a multi-turn episode by pretending it is one long single-turn one.

    ★ What that costs, exactly. Undiscounted, a token's return is everything credited at
    or after it, so lumping leaves the *first* token's return unchanged and raises every
    later one to the same episode total::

        per-token placement:  G_j = sum of rewards from j onward   (differs by position)
        lumped at the end:    G_j = the episode total              (identical everywhere)

    So at ``lam = 1`` every token in the episode is handed the same return and the only
    thing distinguishing their advantages is ``V(s_j)`` -- the critic alone has to work
    out which of four thousand tokens earned the reward, from a scalar that says only
    whether the episode as a whole went well. A turn that scored is not distinguished
    from the turn after it that did not. This is the credit-assignment problem the other
    estimators here exist to address, and it is the honest baseline for showing whether
    they do: the comparison is only meaningful because everything else -- the critic, the
    whitening, the lambda, the row stitching -- is held identical.

    ★ Why not verl's own ``gae``. Under ``concat`` this *is* verl's ``gae`` and
    ``tests/test_default_gae.py`` pins that they agree token for token. Under
    ``no_concat`` and ``compact`` it is not: verl scores one row at a time and opens each
    with ``nextvalues = 0``, so the episode total would be lumped onto the last token of
    *every row* -- each turn separately credited with the whole episode's reward, several
    times over. Running the baseline through the same packing as everything else is what
    keeps the three context policies comparable, and is the reason this is a registered
    estimator rather than a config flag.

    ★ No seam handling, deliberately. A compaction seam is not an environment transition,
    so an estimator may want ``lambda = 1`` there; this one does not distinguish it, for the
    same reason it does not distinguish turns -- a baseline that quietly imports the fixes
    gets credit for them. At the recommended ``gamma = 1, lam = 1`` the question is moot,
    the recursion telescopes everywhere, and at ``lam < 1`` the seam penalty is part of what
    the baseline is being asked to demonstrate. ``_Packed.seam`` is how an estimator
    that does want it gets at the flag.
    """
    gamma, lam = float(inputs.config.gamma), float(inputs.config.lam)

    with torch.no_grad():
        packed = _pack(inputs)
        seq_adv = _backward_gae(
            _lump_rewards_at_trajectory_end(packed),
            packed.seq_v,
            packed.valid,
            gamma,
            lam,
        )
        return packed.emit(
            advantages=packed.scatter(seq_adv),
            returns=packed.scatter(seq_adv + packed.seq_v),
        )


SPEC = register_algorithm("default_gae", compute_default_gae)

__all__ = ["SPEC", "compute_default_gae"]
