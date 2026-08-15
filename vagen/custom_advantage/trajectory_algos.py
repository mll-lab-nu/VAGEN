"""Advantage estimators that score a trajectory however its rows are laid out.

All of them work under all three context policies. :class:`TrajectoryView` presents a
trajectory as an ordered list of rows, and concat is simply the case where that list has
length one -- so no estimator here needs a per-policy branch.

* ``default_gae`` -- **the baseline.** Ordinary GAE with the episode's whole reward
  lumped onto its last token, which is what single-turn RLHF does. Every token is then
  handed the same return and the critic alone has to apportion it.
* ``token_level_gae`` -- the same recursion, but each reward left at the token that
  earned it. One MDP step per model-emitted token: the
  state is everything the model had seen before emitting it, the action is the token.
  Anything the model did not emit -- observations, the chat template's own scaffolding --
  is part of the state, never an action, and the recursion steps over it. The recursion
  also carries across row boundaries, which is the only thing that makes one estimator
  cover all three policies (see below).
* ``turn_level_gae`` -- one decision per turn instead of per token.
* ``trajectory_grpo`` -- one advantage for the whole trajectory, normalised against the other
  trajectories of its prompt group and broadcast to all of its tokens.

★ Why crossing rows is the whole point. A trajectory is one episode, but only concat
puts an episode in one row; no-concat gives each turn its own row and compact starts a
new one whenever it compacts. verl's own ``gae`` opens every row with ``nextvalues=0``,
so under those two policies it asserts that nothing after the row boundary is worth
anything -- an episode's later turns simply stop being credited to its earlier ones.
That is not a different algorithm, it is the same algorithm applied to a truncated
trajectory. Stitching the rows back together is what these estimators add; on concat
they reduce to verl's ``gae`` exactly, which ``tests/test_trajectory_algos.py`` pins.

Only ``turn_level_gae`` writes sentinel returns -- the others supervise every
model-output token, and GRPO needs no critic. See ``registry.py`` for ``value_mask``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import torch
import verl.utils.torch_functional as verl_F

from vagen.custom_advantage.inputs import (
    AdvantageInputs,
    AdvantageOutputs,
    advantage_estimator,
)
from vagen.custom_advantage.trajectory import TrajectoryView, to_int64_codes
from vagen.trainer.logic import IGNORE_RETURN

logger = logging.getLogger(__name__)


def _sequence_index(view: TrajectoryView, width: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Flatten each trajectory's model-output tokens into one padded sequence.

    Returns ``(index, valid)`` of shape ``(n_traj, max_len)``. ``index`` addresses a
    flattened ``(n_rows, width)`` tensor, so a single gather collects a trajectory's
    tokens across all of its rows in turn order. Padding sits on the right and is
    excluded by ``valid``.
    """
    device = view.mask.device
    sequences = []
    for rows in view.trajectories:
        # Rows already arrive in turn order, so concatenating their token positions
        # yields the trajectory's tokens in the order the model produced them.
        positions = [r * width + view.mask[r].nonzero(as_tuple=True)[0] for r in rows]
        sequences.append(torch.cat(positions) if positions else torch.empty(0, dtype=torch.long, device=device))

    max_len = max((len(s) for s in sequences), default=0)
    index = torch.zeros((len(sequences), max_len), dtype=torch.long, device=device)
    valid = torch.zeros((len(sequences), max_len), dtype=torch.bool, device=device)
    for i, seq in enumerate(sequences):
        index[i, : len(seq)] = seq
        valid[i, : len(seq)] = True
    return index, valid


def _last_valid(valid: torch.Tensor) -> torch.Tensor:
    """True at each sequence's final valid position; all False for a sequence with none.

    Padding is contiguous and on the right, so the last valid position is the one whose
    successor is padding, plus the final column. Three callers want this and each used to
    spell it out; the version they spelled used ``torch.roll``, which wraps the first
    column onto the last and then had to undo it -- correct, but only because of a repair
    line that is easy to drop when the expression is copied.
    """
    last = torch.zeros_like(valid)
    if valid.shape[1] == 0:
        return last
    last[:, :-1] = valid[:, :-1] & ~valid[:, 1:]
    last[:, -1] = valid[:, -1]
    return last


@dataclass
class _Packed:
    """A batch's trajectories, each flattened to one padded sequence of its own tokens.

    Every estimator here needs the same three things -- gather the trajectory's rows into
    a single ordered sequence, run a recursion backward over it, scatter the result back
    to the rows it came from -- and differ only in the recursion. Keeping the gather and
    the scatter in one place is what lets the differences be read side by side, and stops
    a fix to one estimator's padding or dtype handling from missing the others.
    """

    view: TrajectoryView
    index: torch.Tensor  # (n_traj, max_len), addressing a flattened (n_rows, width)
    valid: torch.Tensor  # (n_traj, max_len) bool -- False on the right-hand padding
    seq_r: torch.Tensor
    seq_v: torch.Tensor
    mask_f: torch.Tensor  # (n_rows, width), the model-output mask as a float
    rows_values: torch.Tensor
    width: int

    def boundary(self) -> torch.Tensor:
        """True at the last model-output token of each turn."""
        return _is_turn_boundary(self.index, self.valid, self.width)

    def rewards_lumped_to_turn_end(self) -> torch.Tensor:
        """``seq_r`` with each turn's rewards summed onto that turn's final token.

        For estimators whose outer chain has one reward slot per turn and therefore read
        a turn's reward only at its last token -- see ``registry.wants_turn_lumped_reward``
        and the ★ note on :func:`bi_level_gae`. Anything left mid-turn is invisible to
        that chain, so an estimator that needs this shape has to produce it.

        ★ It reduces here rather than at the source. The environment pays a reward on the
        tokens that earned it, because that is what it knows; which estimator will read it
        is not the environment's business and is not knowable from inside a rollout. This
        direction is also the only one available: summing spans onto a boundary is
        well-defined, recovering spans from a boundary total is not.

        The per-turn total is preserved exactly, so this opens no length-hacking channel;
        it only moves where within a turn the reward sits.
        """
        bnd = self.boundary() & self.valid
        # A trajectory's trailing tokens need a home. Every turn should end at a boundary
        # and the last valid token should be one, but a reward summed into a turn whose
        # boundary is missing would be dropped without a trace -- so the final valid token
        # counts as a boundary here whether or not it is flagged as one.
        last_valid = self.valid & ~torch.cat(
            [self.valid[:, 1:], torch.zeros_like(self.valid[:, :1])], dim=1
        )
        bnd = bnd | last_valid

        # Turn id per position: how many boundaries lie strictly before it. At a boundary
        # that is the turn the boundary closes, so a boundary and its own turn agree.
        seg = bnd.long().cumsum(dim=1) - bnd.long()
        masked = torch.where(self.valid, self.seq_r, torch.zeros_like(self.seq_r))
        totals = torch.zeros_like(masked).scatter_add_(1, seg, masked)
        return torch.where(bnd, totals.gather(1, seg), torch.zeros_like(masked))

    def seam(self, ends_with_summary) -> torch.Tensor:
        """True at the last model-output token of a compaction summary.

        Those tokens are turn boundaries by :meth:`boundary` -- a summary is a model
        emission and an action, so it ends a turn in the token stream -- but no
        environment step follows one. Marking them lets an estimator charge the seam
        differently from a real transition; see the module docstring of
        ``vagen/core/harness.py`` for why the difference matters.
        """
        seam = torch.zeros_like(self.valid)
        if ends_with_summary is None or self.valid.shape[1] == 0:
            return seam
        flags = torch.as_tensor(
            np.asarray([bool(x) for x in ends_with_summary]), device=self.valid.device
        )
        on_flagged_row = flags[self.view.rows[self.index // self.width]] & self.valid

        # The summary is the row's *last* emission, so only that one token is the seam.
        # A conversation holds several turns, so `boundary()` is true several times on a
        # flagged row and intersecting with it would mark every turn in the conversation
        # as a seam -- turning the whole row's inter-turn discounting off.
        row_of = self.index // self.width
        last_on_row = torch.zeros_like(self.valid)
        last_on_row[:, :-1] = self.valid[:, 1:] & (row_of[:, 1:] != row_of[:, :-1])
        return on_flagged_row & (last_on_row | _last_valid(self.valid))

    def lump_at_end(self) -> torch.Tensor:
        """``seq_r`` with every reward moved to the trajectory's final model-output token.

        The trajectory's total is unchanged -- only where it is credited moves -- so the
        undiscounted return of the *first* token is identical either way. What changes is
        every token after it. See ``default_gae``.
        """
        total = (self.seq_r * self.valid).sum(dim=1, keepdim=True)
        return torch.where(
            _last_valid(self.valid), total.expand_as(self.seq_r), torch.zeros_like(self.seq_r)
        )

    def scatter(self, seq: torch.Tensor, where: torch.Tensor | None = None, fill: float = 0.0):
        """Sequence-shaped -> row-shaped, writing only at ``where`` (default: everywhere)."""
        where = self.valid if where is None else where
        out = torch.full_like(self.rows_values, float(fill)).reshape(-1)
        out[self.index[where]] = seq[where]
        return out.view_as(self.rows_values)

    def scatter_flag(self, where: torch.Tensor) -> torch.Tensor:
        """A 0/1 long tensor marking the row positions named by ``where``."""
        return self.scatter_int(torch.ones_like(self.index), where=where)

    def scatter_int(self, seq, where: torch.Tensor | None = None, fill: int = 0):
        """Sequence-shaped -> row-shaped, as integers. ``scatter`` but not float."""
        where = self.valid if where is None else where
        out = torch.full_like(self.rows_values, float(fill), dtype=torch.long).reshape(-1)
        out[self.index[where]] = seq[where].to(torch.long)
        return out.view_as(self.rows_values)

    def turn_ids(self) -> torch.Tensor:
        """Which turn each model-output token belongs to; ``-1`` where the model was silent.

        Numbered per trajectory rather than per row, which costs nothing -- a row belongs
        to exactly one trajectory, so ids never collide within a row -- and makes the
        column readable on its own.

        ★ This is the only channel a turn-level *loss* has. ``PolicyLossFn``'s signature
        is fixed and carries no notion of a turn, and the loss sees padded tensors rather
        than the ``response_spans`` the loop published. Deriving turn boundaries a second
        time inside the loss would mean two implementations that can disagree while both
        look right, so the estimator that already computed them publishes them.

        ``-1`` at the silent positions rather than ``0``, which is a real turn. This is
        defensive, not load-bearing: ``turn_gspo`` weights every sum by ``response_mask``,
        so folding the observation tokens into turn 0 would not change its answer -- the
        mutation is a genuine no-op there, and no test catches it because there is nothing
        to catch. The convention is for any *other* consumer, which would otherwise take
        ``0`` at face value and inflate turn 0's length -- hence shrink its ``1 / L_t`` --
        by however verbose the environment happened to be.
        """
        boundary = self.boundary() & self.valid
        turn_of = (boundary.cumsum(dim=1) - boundary.long()).clamp(min=0)
        return self.scatter_int(turn_of, fill=-1)

    def emit(self, advantages, returns, value_mask=None) -> AdvantageOutputs:
        """Whiten the advantages over the model-output mask and broadcast back to rows.

        Every estimator that packs also publishes ``turn_id``, whether or not its own
        recursion needed it: it is the actor's only way to know where a turn starts, and
        which advantage estimator is in use should not decide whether a turn-level loss
        can run. ``turn_gspo`` refuses to start without it rather than inventing one.
        """
        advantages = verl_F.masked_whiten(advantages, self.mask_f) * self.mask_f
        return AdvantageOutputs(
            advantages=self.view.broadcast(advantages),
            returns=self.view.broadcast(returns),
            value_mask=None if value_mask is None else self.view.broadcast(value_mask),
            extra={"turn_id": self.view.broadcast(self.turn_ids())},
        )


def _pack(inputs: AdvantageInputs) -> _Packed:
    view = inputs.view
    scores, values = inputs.rewards, inputs.values
    width = scores.shape[1]
    rows_scores, rows_values = view.gather(scores), view.gather(values)
    index, valid = _sequence_index(view, width)
    flat_scores, flat_values = rows_scores.reshape(-1), rows_values.reshape(-1)
    zeros = torch.zeros_like(flat_values[index])
    return _Packed(
        view=view,
        index=index,
        valid=valid,
        seq_r=torch.where(valid, flat_scores[index], zeros),
        seq_v=torch.where(valid, flat_values[index], zeros),
        mask_f=view.mask.to(rows_scores.dtype),
        rows_values=rows_values,
        width=width,
    )


def _backward_gae(seq_r, seq_v, valid, gamma: float, lam) -> torch.Tensor:
    """GAE run backward over the packed sequences, vectorised across trajectories.

    ``lam`` is either a scalar -- plain token-level GAE -- or a tensor the shape of the
    sequences, which is Sutton & Barto's variable-lambda return. Nothing in the tree passes
    a tensor today; it is what a per-turn lambda would use.

    Padding sits on the right, so the first steps of the loop must leave the recursion
    untouched rather than folding zeros into it.
    """
    n_traj, max_len = valid.shape
    seq_adv = torch.zeros_like(seq_v)
    nextvalues = torch.zeros(n_traj, dtype=seq_v.dtype, device=seq_v.device)
    lastgaelam = torch.zeros_like(nextvalues)
    lam_t = lam if torch.is_tensor(lam) else None

    for t in reversed(range(max_len)):
        live = valid[:, t]
        lam_t_now = lam_t[:, t] if lam_t is not None else lam
        delta = seq_r[:, t] + gamma * nextvalues - seq_v[:, t]
        lastgaelam = torch.where(live, delta + gamma * lam_t_now * lastgaelam, lastgaelam)
        seq_adv[:, t] = torch.where(live, lastgaelam, torch.zeros_like(lastgaelam))
        nextvalues = torch.where(live, seq_v[:, t], nextvalues)
    return seq_adv


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
    the baseline is being asked to demonstrate. ``TrajectoryView.seam`` is how an estimator
    that does want it gets at the flag.
    """
    gamma, lam = float(inputs.config.gamma), float(inputs.config.lam)

    with torch.no_grad():
        packed = _pack(inputs)
        seq_adv = _backward_gae(packed.lump_at_end(), packed.seq_v, packed.valid, gamma, lam)
        return packed.emit(
            advantages=packed.scatter(seq_adv),
            returns=packed.scatter(seq_adv + packed.seq_v),
        )


# publishes_turn_id=False: this one returns a bare AdvantageOutputs rather than
# going through _Packed.emit, so there is no turn_id column for a turn-level loss.
@advantage_estimator("trajectory_grpo", publishes_turn_id=False)
def compute_trajectory_grpo(inputs: AdvantageInputs):
    """One advantage per trajectory, normalised within its prompt group.

    Needs no critic, so ``returns`` mirrors ``advantages`` -- verl's own GRPO does the
    same, and nothing reads ``returns`` when the critic is disabled.
    """
    scores = inputs.rewards
    norm_by_std = inputs.param("norm_adv_by_std_in_grpo", True)

    with torch.no_grad():
        view = inputs.view
        rows_scores = view.gather(scores)
        mask_f = view.mask.to(rows_scores.dtype)

        # A trajectory's return is every reward it collected, wherever those rows sit.
        row_totals = (rows_scores * mask_f).sum(dim=1)
        traj_return = torch.stack([row_totals[rows].sum() for rows in view.trajectories])

        group_codes = to_int64_codes(inputs.group_idx, factorize_if_non_numeric=True)
        # Every row of a trajectory shares its group, so the first row identifies it.
        traj_group = torch.as_tensor(
            np.asarray([group_codes[view.rows[rows[0]].item()] for rows in view.trajectories]),
            device=traj_return.device,
        )

        traj_adv = torch.zeros_like(traj_return)
        for g in torch.unique(traj_group).tolist():
            sel = traj_group == g
            rewards = traj_return[sel]
            centred = rewards - rewards.mean()
            if norm_by_std:
                # A group whose trajectories all scored the same carries no signal;
                # dividing by its zero std would produce NaNs rather than zeros.
                std = rewards.std(unbiased=False)
                centred = centred / (std + 1e-6) if std > 0 else torch.zeros_like(centred)
            traj_adv[sel] = centred

        advantages = torch.zeros_like(rows_scores)
        for j, rows in enumerate(view.trajectories):
            for r in rows:
                advantages[r] = traj_adv[j]
        advantages = advantages * mask_f

        return AdvantageOutputs(
            advantages=view.broadcast(advantages),
            returns=view.broadcast(advantages.clone()),
        )


def _is_turn_boundary(index: torch.Tensor, valid: torch.Tensor, width: int) -> torch.Tensor:
    """True at the last model-output token of each turn.

    A turn ends either because the environment interrupts -- leaving a gap in the
    positions within a row -- or because the trajectory continues in the next row. The
    row change has to be tested explicitly: ``row * width + position`` runs on unbroken
    across a row boundary whenever a row's model output reaches its end, so a gap test
    alone silently merges two turns into one.
    """
    boundary = torch.zeros_like(valid)
    if valid.shape[1] == 0:
        # No row in the batch has a single model-output token, so there are no turns to
        # bracket. Without this the `valid[:, -1]` below indexes a zero-width dimension
        # and raises IndexError instead of returning "no boundaries".
        return boundary
    gap = index[:, 1:] != index[:, :-1] + 1
    new_row = index[:, 1:] // width != index[:, :-1] // width
    boundary[:, :-1] = valid[:, 1:] & (gap | new_row)
    # The last valid token of a trajectory ends the last turn.
    return boundary | _last_valid(valid)


@advantage_estimator("bi_level_gae", needs_critic=True, turn_lumped_reward=True)
def compute_bi_level_gae(inputs: AdvantageInputs):
    """The **published** VAGEN Bi-Level GAE, reproduced as released rather than as fixed.

    Ported from the released implementation
    (``compute_bi_level_gae_advantage_return``, commit 4076507 in this repo's own
    history), not from the paper's prose -- §4.2 and Algorithm 2 disagree about whether
    the turn advantage is composed with the last token's delta or replaces it, and the
    code is what produced the published numbers. It replaces.

    Two nested passes:

    1. **Turn level.** GAE over the turn-final tokens only, anchored at each turn's
       *last* model-output token -- ``V(tau_{<=a_t})``, the value of a prefix that already
       contains the action -- using ``high_level_gamma``::

           delta_t = r_t + high_level_gamma * V(eos_{t+1}) - V(eos_t)
           A_t     = delta_t + high_level_gamma * lam * A_{t+1}

    2. **Token level.** The turn's *return* ``A_t + V(eos_t)`` is written back as the
       reward at that token, then token-level GAE runs backward with ``gamma``, and at
       every turn-final token both the bootstrap and the accumulator are zeroed::

           at a turn end:  nextvalue = 0, lastgaelam = 0
                           delta     = (A_t + V) - V = A_t     ->  advantage = A_t
           inside a turn:  the ordinary token recursion, seeded from A_t

    So the turn's last token receives exactly ``A_t`` -- its own delta is discarded, not
    added -- and each turn's inner chain is independent of every other's.

    ★ The afterstate anchor is real but **self-cancelling here**, and this is the most
    important thing to know about the estimator. Pass 1 telescopes to ``A_t = G_t - V(e_t)``
    and pass 2 inside the turn telescopes to ``V(e_t) - V(j)``; the ``V(e_t)`` terms cancel
    and every token receives ``G_t - V(j)``, the correct advantage. Measured on a tabular
    multi-turn MDP with an exact critic and an exact policy gradient:

    * ``high_level_gamma == gamma`` and ``lam == 1``: **identical to token_level_gae**, to
      4e-16. Not approximately -- the same numbers.
    * the outer chain *in isolation* (i.e. lifted out to be a turn-level estimator) is
      catastrophic: relative error 0.949, and **exactly zero gradient** on every
      non-final token of a turn, because ``G_t - E[G_t | s_t, a_t^{1..L-1}]`` is
      conditionally zero-mean given all but the last token. Only the final token -- usually
      a near-deterministic EOS -- keeps any signal.
    * "fixing" the anchor to the turn's first token *without* moving the write-back
      position makes it **worse than the bug**: relative error 0.156 and a gradient 1.92x
      too large, because the value is then subtracted twice. Anchor and deposit position
      must move together, which is what ``turn_level_gae`` does and why the corrected
      first-anchor estimator already exists under that name.

    So do not "correct" the anchor here. The defect is in how the outer chain is
    *described*, not in what the composite computes.

    ★ Therefore ``high_level_gamma`` is the only thing separating this from
    ``token_level_gae``, and at ``high_level_gamma == gamma`` there is nothing left. The
    released config ships ``gamma 1.0 / lam 1.0 / high_level_gamma 0.99`` and the released
    sokoban script uses ``0.9``; measured divergence from token-level GAE on the same
    fixture is 4e-16 at 1.0, 1.8e-2 at 0.99 and 2.1e-1 at 0.9. Running this at
    ``high_level_gamma = gamma`` reproduces nothing.

    The zeroed accumulator is separately why intra-turn reward cannot propagate across a
    turn boundary, and the overwrite is why the token carrying the outcome reward learns
    from the turn advantage rather than its own delta.

    ★ Pass 1 reads a turn's reward only at the turn's last token, so this estimator lumps
    it there itself, via ``_Packed.rewards_lumped_to_turn_end``. It does not ask the
    environment to pay it that way: where a reward is earned is the environment's business,
    what shape this recursion needs is this function's, and a reward left mid-turn would
    otherwise be invisible to the outer chain and credited by the inner one alone.

    ★ Two clocks, two gammas -- ``gamma`` for tokens, ``+algorithm.high_level_gamma`` for
    turns. Two explicit gammas is what makes this well-defined away from 1.0, and why it
    carries no ``undiscounted`` guard: the released code takes the two separately
    and the paper's Table 23 sets the token one to 1.0. Left unset, ``high_level_gamma``
    follows ``gamma``.
    """
    gamma = float(inputs.config.gamma)
    lam = float(inputs.config.lam)
    high_gamma = float(inputs.param("high_level_gamma", gamma))

    if high_gamma == gamma and lam == 1.0:
        # Not an error -- it is what the released code does with those numbers -- but a run
        # started this way is token_level_gae wearing the baseline's name, and the curves
        # give no hint of it.
        logger.warning(
            "bi_level_gae: high_level_gamma == gamma == %s and lam == 1.0, at which "
            "the two passes telescope and this estimator is IDENTICAL to token_level_gae "
            "(verified to 4e-16). The outer chain's V(afterstate) cancels against the "
            "inner chain exactly. Reproducing the published setting means "
            "high_level_gamma < gamma -- the released config uses 0.99 and the released "
            "sokoban script 0.9.", gamma,
        )

    with torch.no_grad():
        packed = _pack(inputs)
        valid, seq_v = packed.valid, packed.seq_v
        # Pass 1 steps only over turn-final tokens, so fold each turn's reward onto its own
        # boundary first. The environment pays per span; this is where that becomes the
        # one-slot-per-turn shape the outer chain requires.
        seq_r = packed.rewards_lumped_to_turn_end()
        # The released code's `reward_mask`: one position per turn, at its last token.
        turn_end = packed.boundary() & valid
        n_traj, max_len = valid.shape
        zeros = torch.zeros(n_traj, dtype=seq_v.dtype, device=seq_v.device)

        # -- pass 1: turn level, stepping only over turn-final tokens.
        turn_adv = torch.zeros_like(seq_v)
        nextvalue, lastgaelam = zeros.clone(), zeros.clone()
        for t in reversed(range(max_len)):
            live = turn_end[:, t]
            delta = seq_r[:, t] + high_gamma * nextvalue - seq_v[:, t]
            lastgaelam = torch.where(live, delta + high_gamma * lam * lastgaelam, lastgaelam)
            turn_adv[:, t] = torch.where(live, lastgaelam, torch.zeros_like(lastgaelam))
            nextvalue = torch.where(live, seq_v[:, t], nextvalue)

        # The turn's return becomes the reward at its last token; every other position
        # keeps whatever token-level reward it had.
        upd_r = torch.where(turn_end, turn_adv + seq_v, seq_r)

        # -- pass 2: token level, restarting at every turn end.
        seq_adv = torch.zeros_like(seq_v)
        nextvalues, lastgaelam = zeros.clone(), zeros.clone()
        for t in reversed(range(max_len)):
            live = valid[:, t]
            ends = turn_end[:, t]
            nv = torch.where(ends, torch.zeros_like(nextvalues), nextvalues)
            lg = torch.where(ends, torch.zeros_like(lastgaelam), lastgaelam)
            delta = upd_r[:, t] + gamma * nv - seq_v[:, t]
            lastgaelam = torch.where(live, delta + gamma * lam * lg, lastgaelam)
            seq_adv[:, t] = torch.where(live, lastgaelam, torch.zeros_like(lastgaelam))
            nextvalues = torch.where(live, seq_v[:, t], nextvalues)

        return packed.emit(
            advantages=packed.scatter(seq_adv),
            returns=packed.scatter(seq_adv + packed.seq_v),
        )


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
