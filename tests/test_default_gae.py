"""``default_gae`` -- the vanilla baseline the multi-turn estimators are measured against.

It is ``token_level_gae`` with one thing changed: the episode's rewards are summed and
moved to its last model-output token before the recursion runs. Everything here pins
either that difference or the fact that nothing *else* differs, because the comparison in
the sweep is only worth anything if the baseline is the same algorithm with one knob
turned.

Two claims from the docstring get their own tests because they are the reason this
estimator exists at all rather than being ``algorithm.adv_estimator=gae``:
under concat it must agree with verl's own GAE exactly, and under no-concat verl's must
disagree -- by crediting *every row's* last token with the whole episode's reward.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from tensordict import TensorDict
from verl.trainer.ppo.core_algos import compute_gae_advantage_return, get_adv_estimator_fn

from vagen.algorithms import (
    TRAJECTORY_ESTIMATORS,
    UNDISCOUNTED_ESTIMATORS,
    needs_critic,
    needs_value_mask,
    requires_undiscounted,
    spans_rows,
)


class _Cfg(dict):
    """config.algorithm, which is both attribute- and get-accessed."""

    def __init__(self, gamma=1.0, lam=1.0, **extra):
        super().__init__()
        self.gamma, self.lam = gamma, lam
        for key, value in extra.items():
            setattr(self, key, value)

    def get(self, key, default=None):
        return getattr(self, key, default)


# One episode, two turns of two model-output tokens. ★ Reward in *both* turns: with
# reward only at the very end, lumping is a no-op and every test here would pass against
# `token_level_gae` too.
VALUES = [0.10, 0.20, 0.30, 0.40]  # per model-output token, in emission order
MID, END = 0.5, 1.0                # turn 0's reward, turn 1's
TOTAL = MID + END


def _concat(scores_at_tokens=(0.0, MID, 0.0, END), values=VALUES):
    """One row per episode: ``[a0 a1 <obs> a2 a3]``, the observation masked out."""
    s = list(scores_at_tokens)
    return (
        [[s[0], s[1], 0.0, s[2], s[3]]],
        [[1, 1, 0, 1, 1]],
        [[values[0], values[1], 9.9, values[2], values[3]]],
        ["g"], [0], [0],
    )


def _split(scores_at_tokens=(0.0, MID, 0.0, END), values=VALUES):
    """One row per turn: ``[a0 a1]`` then ``[a2 a3]``. Same tokens, same values."""
    s = list(scores_at_tokens)
    return (
        [[s[0], s[1]], [s[2], s[3]]],
        [[1, 1], [1, 1]],
        [[values[0], values[1]], [values[2], values[3]]],
        ["g", "g"], [0, 0], [0, 1],
    )


def _tensors(layout):
    scores, masks, values, group, traj, turn = layout
    batch = TensorDict(
        {
            "token_level_scores": torch.tensor(scores, dtype=torch.float32),
            "response_mask": torch.tensor(masks, dtype=torch.long),
            "values": torch.tensor(values, dtype=torch.float32),
        },
        batch_size=[len(scores)],
    )
    non_tensor = {
        "group_idx": np.array(group, dtype=object),
        "traj_idx": np.array(traj),
        "turn_idx": np.array(turn),
    }
    return batch, non_tensor, torch.tensor(masks, dtype=torch.bool)


def _call(name, layout, **cfg):
    batch, non_tensor, mask = _tensors(layout)
    adv, ret = get_adv_estimator_fn(name)(batch=batch, non_tensor_batch=non_tensor, config=_Cfg(**cfg))
    return adv, ret, mask


def _at_tokens(tensor, mask):
    """Values at model-output positions in emission order -- the one layout-free view."""
    return [float(v) for row, m in zip(tensor, mask) for v in row[m]]


# ------------------------------------------------------------------- the defining claim


def test_every_token_is_handed_the_whole_episode_total():
    """★ The property that makes this the baseline.

    Undiscounted with ``lam = 1``, a token's return is every reward credited at or after
    it. Lumping puts them all at the end, so *every* token's return is the episode total
    -- the turn that scored is not distinguished from the turn that did not, and only the
    critic can tell them apart.
    """
    _, ret, mask = _call("default_gae", _concat(), gamma=1.0, lam=1.0)
    assert _at_tokens(ret, mask) == pytest.approx([TOTAL] * 4)


def test_token_level_gae_on_the_same_fixture_does_not():
    """The contrast, on identical inputs. Per-token placement gives a token before the
    mid-episode reward a larger return than one after it; that difference is exactly the
    credit signal the baseline throws away."""
    _, ret, mask = _call("token_level_gae", _concat(), gamma=1.0, lam=1.0)
    assert _at_tokens(ret, mask) == pytest.approx([TOTAL, TOTAL, END, END])


def test_where_the_reward_sat_makes_no_difference():
    """★ The strongest statement of what "lumped" means: three placements of the same
    1.5, byte-identical outputs. If any of the reward were being credited in place, the
    front-loaded layout would not match the back-loaded one."""
    spread = _call("default_gae", _concat((0.0, MID, 0.0, END)), lam=0.7)
    up_front = _call("default_gae", _concat((TOTAL, 0.0, 0.0, 0.0)), lam=0.7)
    at_the_end = _call("default_gae", _concat((0.0, 0.0, 0.0, TOTAL)), lam=0.7)

    for other in (up_front, at_the_end):
        assert _at_tokens(other[0], other[2]) == pytest.approx(_at_tokens(spread[0], spread[2]))
        assert _at_tokens(other[1], other[2]) == pytest.approx(_at_tokens(spread[1], spread[2]))


def test_the_episode_total_is_conserved():
    """Lumping moves reward, it does not create or destroy any: the first token's return
    is the same as ``token_level_gae`` would give it, since nothing is credited before it
    either way."""
    _, lumped, mask = _call("default_gae", _concat(), gamma=1.0, lam=1.0)
    _, per_token, _ = _call("token_level_gae", _concat(), gamma=1.0, lam=1.0)
    assert _at_tokens(lumped, mask)[0] == pytest.approx(_at_tokens(per_token, mask)[0])


def test_only_the_critic_distinguishes_two_tokens():
    """With one return for the whole episode, ``A_j = total - V(s_j)``: two tokens whose
    values agree must receive the same advantage, however far apart they are and whatever
    happened between them. Whitening is affine, so it preserves the equality."""
    values = [0.10, 0.25, 0.30, 0.25]  # tokens 1 and 3 share a value
    adv, _, mask = _call("default_gae", _concat(values=values), gamma=1.0, lam=1.0)
    at = _at_tokens(adv, mask)
    assert at[1] == pytest.approx(at[3])
    # ...and the ones that differ in value are genuinely told apart, so the test above is
    # not passing because everything collapsed to a constant.
    assert at[0] != pytest.approx(at[1])


def test_lambda_is_honoured():
    """A guard against the recursion silently running at ``lam = 1``, which would make
    most of this file pass for the wrong reason."""
    strict, _, mask = _call("default_gae", _concat(), lam=1.0)
    bootstrapped, _, _ = _call("default_gae", _concat(), lam=0.3)
    assert _at_tokens(strict, mask) != pytest.approx(_at_tokens(bootstrapped, mask))


# ------------------------------------------------------- why it is not verl's own `gae`


def _verl_gae(layout, gamma=1.0, lam=1.0):
    batch, _, mask = _tensors(layout)
    adv, ret = compute_gae_advantage_return(
        token_level_rewards=batch["token_level_scores"],
        values=batch["values"],
        response_mask=batch["response_mask"],
        gamma=gamma,
        lam=lam,
    )
    return adv, ret, mask


def test_under_concat_it_is_exactly_verls_gae():
    """★ The claim that makes this a fair baseline rather than a new algorithm. Given a
    concat row whose reward already sits at the last token -- which is what lumping
    produces -- the two agree token for token. Returns, not advantages: ours are whitened
    and verl's are not, and whitening is the trainer's business, not the estimator's."""
    already_lumped = _concat((0.0, 0.0, 0.0, TOTAL))
    _, mine, mask = _call("default_gae", already_lumped, gamma=1.0, lam=0.9)
    _, theirs, _ = _verl_gae(already_lumped, gamma=1.0, lam=0.9)
    assert _at_tokens(mine, mask) == pytest.approx(_at_tokens(theirs, mask), rel=1e-5, abs=1e-6)


def test_verls_gae_would_credit_every_row_under_no_concat():
    """★ Documents the bug this estimator exists to avoid -- verl's behaviour, not ours.

    Hand verl's GAE the same episode split one-row-per-turn with the total already at the
    end, and it opens each row with ``nextvalues = 0``: turn 0 is credited with 1.5 of its
    own and turn 1 with another 1.5. The episode's reward is counted once per row, and
    nothing about the run looks wrong.
    """
    split_and_lumped = ([[0.0, TOTAL], [0.0, TOTAL]], [[1, 1], [1, 1]],
                        [[VALUES[0], VALUES[1]], [VALUES[2], VALUES[3]]], ["g", "g"], [0, 0], [0, 1])
    _, theirs, mask = _verl_gae(split_and_lumped, gamma=1.0, lam=1.0)
    assert _at_tokens(theirs, mask) == pytest.approx([TOTAL, TOTAL, TOTAL, TOTAL])

    # Ours, given the un-lumped episode in the same layout, credits 1.5 once.
    _, mine, mask = _call("default_gae", _split(), gamma=1.0, lam=1.0)
    assert _at_tokens(mine, mask) == pytest.approx([TOTAL] * 4)
    rewards_seen = sum(sum(r) for r in split_and_lumped[0])
    assert rewards_seen == pytest.approx(2 * TOTAL), "verl was handed the episode twice over"


def test_the_two_layouts_agree():
    """Already covered by the contract battery, restated here because for *this*
    estimator it is a stronger claim than usual: the lump has to be found across rows,
    so a per-row sum would pass every other test in this file and fail this one."""
    a_concat, r_concat, m_concat = _call("default_gae", _concat(), lam=0.8)
    a_split, r_split, m_split = _call("default_gae", _split(), lam=0.8)
    assert _at_tokens(a_concat, m_concat) == pytest.approx(_at_tokens(a_split, m_split))
    assert _at_tokens(r_concat, m_concat) == pytest.approx(_at_tokens(r_split, m_split))


# ------------------------------------------------------------------ registry and edges


def test_declares_what_the_trainer_needs_to_know():
    assert "default_gae" in TRAJECTORY_ESTIMATORS, "must stitch an episode's rows together"
    assert spans_rows("default_gae") is True
    assert needs_critic("default_gae") is True
    # It supervises every model-output token, so a value_mask would be wrong for it.
    assert needs_value_mask("default_gae") is False


def test_is_not_a_two_clock_estimator():
    """★ This runs one recursion on one clock, so ``gamma < 1``
    is merely a choice rather than undefined -- and the startup assertion must not fire.
    (It is a bad choice: at gamma 0.99 a 4000-token episode delivers 2e-18 of its reward
    to the first token. That is a documented caveat, not something to refuse.)"""
    assert "default_gae" not in UNDISCOUNTED_ESTIMATORS
    assert requires_undiscounted("default_gae") is False


def test_publishes_turn_id_like_every_other_packing_estimator():
    """The actor's only channel for turn boundaries. Which advantage estimator is in use
    must not decide whether a turn-level loss can run."""
    batch, non_tensor, _ = _tensors(_concat())
    get_adv_estimator_fn("default_gae")(batch=batch, non_tensor_batch=non_tensor, config=_Cfg())
    assert "turn_id" in batch.keys()
    assert batch["turn_id"].tolist() == [[0, 0, -1, 1, 1]]


def test_a_shorter_trajectory_still_gets_its_own_total():
    """★ The ragged case, which every batch in training is and no fixture in this repo
    was. Two trajectories of different lengths pack into one padded block, so the shorter
    one's last token is in the middle of its row rather than at the end -- the branch of
    ``_last_valid`` that the equal-length fixtures never reach.

    A mutation that marks the *first* valid position instead of the last passes every
    other test in this suite, because with equal lengths the final-column repair covers
    for it. Here it does not: the short trajectory's reward is dropped on the floor and it
    trains on a return of zero, while the long one looks fine.
    """
    long_reward, short_reward = TOTAL, 3.0
    layout = (
        [[0.0, MID, 0.0, 0.0, END], [0.0, short_reward, 0.0, 0.0, 0.0]],
        [[1, 1, 0, 1, 1], [1, 1, 0, 0, 0]],                     # 4 tokens, then 2
        [[0.1, 0.2, 9.9, 0.3, 0.4], [0.5, 0.6, 9.9, 9.9, 9.9]],
        ["g", "g"], [0, 1], [0, 0],
    )
    _, ret, mask = _call("default_gae", layout, gamma=1.0, lam=1.0)
    per_row = [[float(v) for v in row[m]] for row, m in zip(ret, mask)]
    assert per_row[0] == pytest.approx([long_reward] * 4)
    assert per_row[1] == pytest.approx([short_reward] * 2), "the short trajectory lost its reward"


def test_the_short_trajectorys_last_turn_closes():
    """The same ragged block through ``_last_valid``'s *other* caller, and the reason
    that helper is shared rather than inlined three times.

    ``_last_valid`` is what closes a trajectory's final turn. Turn *numbering* cannot
    detect a missing final boundary -- ``turn_of`` is a cumsum taken before it -- so this
    goes through ``turn_level_gae``, where an unclosed turn is counted as not existing and
    the whole trajectory silently receives zero advantage. Here the short trajectory has
    exactly one turn, so losing it loses everything: reward 1.0 from a state valued 0.5
    must give a return of 1.0 at the turn's anchor, not the bare 0.5 of a turn that never
    ran.
    """
    from vagen.training.trainer.logic import IGNORE_RETURN

    layout = (
        [[0.0, 0.0, 0.0, 0.0, END], [0.0, END, 0.0, 0.0, 0.0]],
        [[1, 1, 0, 1, 1], [1, 1, 0, 0, 0]],
        [[0.1, 0.2, 9.9, 0.3, 0.4], [0.5, 0.6, 9.9, 9.9, 9.9]],
        ["g", "g"], [0, 1], [0, 0],
    )
    adv, ret, mask = _call("turn_level_gae", layout, gamma=1.0, lam=1.0)
    assert float(ret[1][0]) == pytest.approx(END), "the short trajectory's only turn did not close"
    assert float(ret[1][1]) == pytest.approx(IGNORE_RETURN)
    assert not torch.all(adv[1][mask[1]] == 0), "and so it learned nothing from the episode"


def test_a_trajectory_with_no_model_output_does_not_blow_up():
    """A fully-masked row reaches ``_last_valid`` with nothing to mark. It must produce
    zeros rather than an index error or a NaN from summing an empty slice."""
    layout = (
        [[0.0, 0.0], [0.0, END]],
        [[0, 0], [1, 1]],
        [[9.9, 9.9], [VALUES[0], VALUES[1]]],
        ["g", "g"], [0, 1], [0, 0],
    )
    adv, ret, mask = _call("default_gae", layout)
    assert torch.isfinite(adv).all() and torch.isfinite(ret).all()
    assert torch.all(adv[~mask] == 0)
