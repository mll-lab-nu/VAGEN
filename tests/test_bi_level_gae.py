"""``bi_level_gae`` -- the published VAGEN Bi-Level GAE, reproduced as released.

The point of this estimator is fidelity, so the load-bearing test is differential:
``_released`` below is ``compute_bi_level_gae_advantage_return`` copied verbatim out of
this repo's own history (commit 4076507), and the vectorised implementation has to agree
with it token for token. Everything else here pins a property that would otherwise only
be visible by reading that loop.

Why a reproduction at all: this is the algorithm as released, three corrections behind
(anchor at the turn's first token, add rather than overwrite, no intra-turn reset). Those
corrections are worth what they measure, and they can only be measured against the thing
they were made to.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from tensordict import TensorDict
from verl.trainer.ppo.core_algos import get_adv_estimator_fn

import verl.utils.torch_functional as verl_F

import vagen.custom_advantage  # noqa: F401  importing is what registers the estimators


# --------------------------------------------------------------------- the original
#
# Verbatim from commit 4076507 `vagen/trainer/ppo/core_algos.py`, reformatted but not
# altered. Do not "clean this up": its value is that it is not our code.


def _released(token_level_rewards, reward_mask, values, loss_mask, gamma, lam, high_level_gamma):
    with torch.no_grad():
        batch_size, gen_len = token_level_rewards.shape
        advantages = torch.zeros_like(token_level_rewards)
        returns = torch.zeros_like(token_level_rewards)
        updated_reward = token_level_rewards.clone()

        for b in range(batch_size):
            eos_positions = reward_mask[b].nonzero(as_tuple=True)[0]
            lastgaelam = 0.0
            for i in range(len(eos_positions) - 1, -1, -1):
                curr_pos = eos_positions[i]
                if i < len(eos_positions) - 1:
                    next_pos = eos_positions[i + 1]
                    nextvalue = values[b, next_pos]
                else:
                    nextvalue = 0.0
                delta = updated_reward[b, curr_pos] + high_level_gamma * nextvalue - values[b, curr_pos]
                lastgaelam = delta + high_level_gamma * lam * lastgaelam
                advantages[b, curr_pos] = lastgaelam

            for i, pos in enumerate(eos_positions):
                returns[b, pos] = advantages[b, pos] + values[b, pos]
                updated_reward[b, pos] = advantages[b, pos] + values[b, pos]

            lastgaelam = 0.0
            valid_positions = loss_mask[b].nonzero(as_tuple=True)[0]
            for i in range(len(valid_positions) - 1, -1, -1):
                curr_pos = valid_positions[i]
                if curr_pos not in eos_positions:
                    next_pos = valid_positions[i + 1]
                    nextvalue = values[b, next_pos]
                else:
                    nextvalue = 0.0
                    lastgaelam = 0.0
                delta = updated_reward[b, curr_pos] + gamma * nextvalue - values[b, curr_pos]
                lastgaelam = delta + gamma * lam * lastgaelam
                advantages[b, curr_pos] = lastgaelam
                returns[b, curr_pos] = lastgaelam + values[b, curr_pos]

        advantages = verl_F.masked_whiten(advantages, loss_mask)
    return advantages, returns


# ------------------------------------------------------------------------- fixtures


class _Cfg(dict):
    def __init__(self, gamma=1.0, lam=0.95, **extra):
        super().__init__()
        self.gamma, self.lam = gamma, lam
        for k, v in extra.items():
            setattr(self, k, v)

    def get(self, key, default=None):
        return getattr(self, key, default)


#: One episode, three turns of unequal length, an observation between each.
#: ``.`` = model token, ``o`` = observation. This fixture is already turn-lumped so it can
#: be compared directly with the released implementation; a separate test below pins the
#: new reduction from per-span rewards.
MASK = [1, 1, 1, 0, 1, 1, 0, 1]           # turns: [0,1,2] [4,5] [7]
SCORES = [0.0, 0.0, 0.3, 0.0, 0.0, -0.2, 0.0, 1.0]
VALUES = [0.10, 0.25, 0.40, 9.9, 0.55, 0.70, 9.9, 0.85]
REWARD_MASK = [0, 0, 1, 0, 0, 1, 0, 1]    # each turn's last model token


def _concat(n_traj=2):
    scale = [1.0, 2.0][:n_traj]
    return (
        [[s * k for s in SCORES] for k in scale],
        [MASK for _ in scale],
        [VALUES for _ in scale],
        ["g"] * n_traj, list(range(n_traj)), [0] * n_traj,
    )


def _split():
    """The same episode, one row per turn -- same tokens, values and rewards.

    Right-padded to a common width, because rows of a batch are one tensor. The padding
    is masked out, so the model tokens are the same six in the same order as `_concat`.
    """
    return (
        [[0.0, 0.0, 0.3], [0.0, -0.2, 0.0], [1.0, 0.0, 0.0]],
        [[1, 1, 1], [1, 1, 0], [1, 0, 0]],
        [[0.10, 0.25, 0.40], [0.55, 0.70, 9.9], [0.85, 9.9, 9.9]],
        ["g", "g", "g"], [0, 0, 0], [0, 1, 2],
    )


def _tensors(layout):
    scores, masks, values, group, traj, turn = layout
    batch = TensorDict(
        {
            "token_level_scores": torch.tensor(scores, dtype=torch.float64),
            "response_mask": torch.tensor(masks, dtype=torch.long),
            "values": torch.tensor(values, dtype=torch.float64),
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


def _at(tensor, mask):
    return [float(v) for row, m in zip(tensor, mask) for v in row[m]]


# ------------------------------------------------------- the claim: it is the original


@pytest.mark.parametrize("gamma,lam,high", [(1.0, 0.95, 1.0), (1.0, 1.0, 1.0),
                                            (0.99, 0.9, 0.95), (1.0, 0.5, 0.8)])
def test_matches_the_released_implementation_token_for_token(gamma, lam, high):
    """★ The whole reason this estimator exists. A reproduction that is merely
    *inspired by* the released code reproduces nothing; the published numbers came from
    that loop, so the test is a differential against that loop and not against a reading
    of the paper -- whose §4.2 and Algorithm 2 disagree with each other about the
    overwrite."""
    layout = _concat()
    batch, _, mask = _tensors(layout)
    want_adv, want_ret = _released(
        token_level_rewards=batch["token_level_scores"],
        reward_mask=torch.tensor([REWARD_MASK] * len(layout[0]), dtype=torch.long),
        values=batch["values"],
        loss_mask=batch["response_mask"],
        gamma=gamma, lam=lam, high_level_gamma=high,
    )
    got_adv, got_ret, _ = _call("bi_level_gae", layout, gamma=gamma, lam=lam, high_level_gamma=high)

    # Returns, and the advantage *before* whitening (`return - V`, which is what both
    # implementations actually compute). Exact equality, not a tolerance: same recursion,
    # same order, same float64 inputs, so anything but 0.0 is a difference in the
    # algorithm. The whitened advantage is deliberately not compared -- `masked_whiten` is
    # verl's, both call it on identical input, and comparing it only measures how it
    # amplifies float noise.
    v = batch["values"]
    assert _at(got_ret, mask) == _at(want_ret, mask)
    assert _at(got_ret - v, mask) == _at(want_ret - v, mask)
    # `want_adv` (whitened) is intentionally unused: see the comment above.


def test_high_level_gamma_follows_gamma_when_unset():
    """Left unset it must not silently become 1.0 (or 0.0); it follows the token gamma,
    which is what a single-gamma config expects."""
    a, b = (_call("bi_level_gae", _concat(), gamma=0.97, lam=0.9, high_level_gamma=g)
            for g in (0.97, 0.5))
    default, _, mask = _call("bi_level_gae", _concat(), gamma=0.97, lam=0.9)
    assert _at(default, mask) == pytest.approx(_at(a[0], mask))
    assert _at(default, mask) != pytest.approx(_at(b[0], mask))


# --------------------------------------------------------- the properties, stated once


def test_a_turns_last_token_gets_the_turn_advantage_and_not_its_own_delta():
    """★ The overwrite. At a turn end the released code zeroes both the bootstrap and the
    accumulator, so ``delta = (A_turn + V) - V = A_turn`` exactly -- that token's own
    delta is discarded rather than added. Adding it instead is a one-line change, and the
    behaviour pinned here is what makes this the released algorithm rather than that one.

    Checked on `returns`, which is unwhitened: ``return = A_turn + V(eos)`` is the turn's
    own return, so the last turn's must be its reward plus nothing, 1.0, because it
    bootstraps from zero.
    """
    _, ret, mask = _call("bi_level_gae", _concat(1), gamma=1.0, lam=1.0, high_level_gamma=1.0)
    at = _at(ret, mask)
    # Final turn (one token, reward 1.0, bootstrap 0): A = 1.0 - 0.85, return = 1.0.
    assert at[-1] == pytest.approx(1.0)


def test_each_turns_inner_chain_is_independent():
    """★ The reset. Changing a reward in a later turn must not reach an earlier turn's
    *non-final* tokens through the inner chain -- only through the outer chain, which
    enters at the earlier turn's own last token. This is exactly the property the
    corrected estimator drops, and the reason intra-turn reward cannot propagate here.

    So: perturb the last turn's reward, and hold the earlier turns' values fixed. The
    earlier turns' last tokens move (outer chain); the tokens before them move only by
    what the outer chain handed to their turn end.
    """
    base = list(SCORES)
    bumped = list(SCORES)
    bumped[-1] += 5.0

    def run(scores):
        layout = ([scores], [MASK], [VALUES], ["g"], [0], [0])
        _, ret, mask = _call("bi_level_gae", layout, gamma=1.0, lam=0.0, high_level_gamma=0.0)
        return _at(ret, mask)

    # lam=0 and high_level_gamma=0 sever the outer chain entirely: with no discounting of
    # a future turn and no lambda, each turn's advantage is its own delta alone. Then a
    # change in the last turn must not move ANY earlier token.
    before, after = run(base), run(bumped)
    assert before[:-1] == pytest.approx(after[:-1]), "credit crossed a turn boundary with the chains cut"
    assert before[-1] != pytest.approx(after[-1]), "the perturbed turn did not move at all"


# `test_it_is_not_the_corrected_estimator` stood here, asserting that this estimator and
# `bi_level_gae_varlam` differ on the same input. The variable-lambda estimator has been
# removed, so there is nothing left to differ from; what this algorithm IS is pinned by the
# hand-computed recursions above, which is the stronger statement anyway.


def test_the_two_layouts_agree():
    """It must not be a concat-only algorithm. The released code is: it takes one row and
    opens it at ``nextvalues=0``, so under no_concat every turn would be its own episode.
    Running it through the packing is the one thing here that is not a straight port."""
    a_concat, r_concat, m_concat = _call("bi_level_gae", _concat(1), lam=0.9)
    a_split, r_split, m_split = _call("bi_level_gae", _split(), lam=0.9)
    assert _at(r_concat, m_concat) == pytest.approx(_at(r_split, m_split))
    assert _at(a_concat, m_concat) == pytest.approx(_at(a_split, m_split))


def test_per_span_rewards_are_reduced_to_the_same_turn_totals_internally():
    """The environment no longer has to know that this estimator wants one slot/turn."""
    per_span = [0.3, 0.0, 0.0, 0.0, -0.2, 0.0, 0.0, 1.0]
    layout = ([per_span], [MASK], [VALUES], ["g"], [0], [0])

    a_lumped, r_lumped, m_lumped = _call("bi_level_gae", _concat(1), lam=0.9)
    a_span, r_span, m_span = _call("bi_level_gae", layout, lam=0.9)

    assert _at(r_span, m_span) == pytest.approx(_at(r_lumped, m_lumped))
    assert _at(a_span, m_span) == pytest.approx(_at(a_lumped, m_lumped))


def test_registry_declarations():
    from vagen.custom_advantage import (
        TRAJECTORY_ESTIMATORS, UNDISCOUNTED_ESTIMATORS, needs_critic, needs_value_mask,
    )

    assert "bi_level_gae" in TRAJECTORY_ESTIMATORS
    assert needs_critic("bi_level_gae") is True
    assert needs_value_mask("bi_level_gae") is False
    # ★ Two explicit gammas, so it is well-defined away from 1.0 and must NOT be caught by
    # the single-clock startup assertion.
    assert "bi_level_gae" not in UNDISCOUNTED_ESTIMATORS


# ------------------------------------------------- the fact that decides the experiment


def test_it_is_token_level_gae_when_the_two_clocks_agree():
    """★ At ``high_level_gamma == gamma`` and ``lam == 1`` this estimator IS
    ``token_level_gae`` -- not close to it, identical.

    Pass 1 telescopes to ``A_t = G_t - V(e_t)`` and pass 2 inside the turn telescopes to
    ``V(e_t) - V(j)``; the afterstate anchor cancels and every token gets ``G_t - V(j)``.
    Nothing in a training curve would reveal this, so a run configured that way is the
    token-level baseline wearing the paper baseline's name -- which is exactly how the
    first three `bi_level_new` jobs were launched on 2026-08-10.
    """
    paper, _, mask = _call("bi_level_gae", _concat(), gamma=1.0, lam=1.0, high_level_gamma=1.0)
    token, _, _ = _call("token_level_gae", _concat(), gamma=1.0, lam=1.0)
    assert _at(paper, mask) == pytest.approx(_at(token, mask), abs=1e-12)


@pytest.mark.parametrize("high,floor", [(0.99, 1e-2), (0.9, 1e-1)])
def test_the_published_settings_are_genuinely_a_different_algorithm(high, floor):
    """...and away from that point it is not. The released config ships
    ``high_level_gamma=0.99`` and the released sokoban script ``0.9``, which is where the
    estimator earns its name. The floors are loose on purpose: the claim is that the gap
    is real and grows as the clocks separate, not that it has a particular size."""
    paper, _, mask = _call("bi_level_gae", _concat(), gamma=1.0, lam=1.0, high_level_gamma=high)
    token, _, _ = _call("token_level_gae", _concat(), gamma=1.0, lam=1.0)
    gap = max(abs(a - b) for a, b in zip(_at(paper, mask), _at(token, mask)))
    assert gap > floor, f"high_level_gamma={high} is indistinguishable from token_level_gae"


def test_the_degenerate_setting_warns(caplog):
    """A guard nothing says out loud is not a guard. The run has to be able to say, in its
    own log, that it reproduced the wrong thing."""
    import logging

    with caplog.at_level(logging.WARNING):
        _call("bi_level_gae", _concat(), gamma=1.0, lam=1.0, high_level_gamma=1.0)
    assert any("IDENTICAL to token_level_gae" in r.message for r in caplog.records)

    caplog.clear()
    with caplog.at_level(logging.WARNING):
        _call("bi_level_gae", _concat(), gamma=1.0, lam=1.0, high_level_gamma=0.9)
    assert not any("IDENTICAL" in r.message for r in caplog.records)
