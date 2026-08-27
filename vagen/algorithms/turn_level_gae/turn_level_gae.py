"""Turn-level GAE implementation."""

import torch

from vagen.algorithms._common import AdvantageInputs, advantage_estimator, register_algorithm
from vagen.algorithms._common.packing import _pack
from vagen.training.trainer.logic import IGNORE_RETURN


@advantage_estimator("turn_level_gae", needs_critic=True, sentinel_returns=True)
def compute_turn_level_gae(inputs: AdvantageInputs):
    """Turn-level GAE: one decision per turn, whatever rows the turn occupies.

    The recursion runs over turns rather than tokens -- a turn's reward is everything it
    collected, its value is the critic at its *first* model-output token -- and the
    resulting advantage is broadcast to every token of that turn.

    ★ First, not last. ``V(s_t)`` is the value of the state the turn acts from, which is
    the position before any of its tokens have been emitted. The critic at the turn's
    last token has already seen nearly the whole turn and is answering a different
    question. That broadcast is not
    an approximation: an autoregressive policy factorises as
    ``log pi(turn) = sum_j log pi(token_j)``, so the per-token coefficient in the
    turn-level policy gradient is exactly the turn's advantage.

    Returns are written only at each turn's first token, the rest left at the sentinel,
    because the critic is being asked for a turn-level value and only that position
    carries one. ``value_mask`` is what stops it training on the rest.

    Turns are found from the token stream rather than assumed to be rows, so this works
    under every context policy. The estimator it replaced assumed one row per turn, which
    made it a no-concat-only algorithm wearing an algorithm's name.
    """
    gamma, lam = float(inputs.config.gamma), float(inputs.config.lam)
    # Was a keyword argument, which the adapter has no way to supply -- it calls the
    # estimator with the inputs object and nothing else, so the parameter was unreachable
    # and its default was the only value it could ever take.
    ignore_value = float(inputs.param("ignore_return", IGNORE_RETURN))

    with torch.no_grad():
        packed = _pack(inputs)
        index, valid = packed.index, packed.valid
        seq_r, seq_v = packed.seq_r, packed.seq_v
        boundary = packed.boundary() & valid

        # Turn index of every token: how many turns ended strictly before it.
        turn_of = (boundary.cumsum(dim=1) - boundary.long()).clamp(min=0)
        # A turn's first token: the one after a boundary, plus the sequence's own start.
        start = torch.zeros_like(boundary)
        start[:, 1:] = boundary[:, :-1]
        start[:, 0] = valid[:, 0]
        start = start & valid
        n_turns = int(boundary.sum(dim=1).max().item()) or 1
        n_traj = index.shape[0]

        turn_r = torch.zeros((n_traj, n_turns), dtype=seq_v.dtype, device=seq_v.device)
        turn_r.scatter_add_(1, turn_of.clamp(max=n_turns - 1), seq_r)
        # scatter_add_, not scatter_: with one index per token, several tokens land in
        # the same turn slot and a plain scatter keeps whichever is written last -- a
        # zero from a non-start token, overwriting the value that was wanted.
        turn_v = torch.zeros_like(turn_r)
        turn_v.scatter_add_(1, turn_of.clamp(max=n_turns - 1), torch.where(start, seq_v, torch.zeros_like(seq_v)))
        turn_count = torch.zeros_like(turn_r)
        turn_count.scatter_add_(1, turn_of.clamp(max=n_turns - 1), boundary.to(turn_r.dtype))
        turn_alive = turn_count > 0

        turn_adv = torch.zeros_like(turn_r)
        nextvalue = torch.zeros(n_traj, dtype=seq_v.dtype, device=seq_v.device)
        lastgaelam = torch.zeros_like(nextvalue)
        for t in reversed(range(n_turns)):
            live = turn_alive[:, t]
            delta = turn_r[:, t] + gamma * nextvalue - turn_v[:, t]
            lastgaelam = torch.where(live, delta + gamma * lam * lastgaelam, lastgaelam)
            turn_adv[:, t] = torch.where(live, lastgaelam, torch.zeros_like(lastgaelam))
            nextvalue = torch.where(live, turn_v[:, t], nextvalue)

        seq_adv = torch.gather(turn_adv, 1, turn_of.clamp(max=n_turns - 1)) * valid
        seq_ret = torch.gather(turn_adv + turn_v, 1, turn_of.clamp(max=n_turns - 1))

        return packed.emit(
            advantages=packed.scatter(seq_adv),
            # Only the turn's first token carries a turn-level return -- that is the
            # state the value was asked about. The rest stay at the sentinel and are
            # excluded from the critic loss by value_mask.
            returns=packed.scatter(seq_ret, where=start, fill=ignore_value),
            # The critic's supervision, stated by the estimator that knows it rather
            # than reverse-engineered downstream by looking for the sentinel. Same
            # positions, but one source instead of two that can disagree.
            value_mask=packed.scatter_flag(start),
        )


SPEC = register_algorithm("turn_level_gae", compute_turn_level_gae)

__all__ = ["SPEC", "compute_turn_level_gae"]
