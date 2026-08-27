"""The compact seam, and the identity that decides where one trajectory ends.

Nothing covered either before. `test_trajectory_algos.py` is 526 lines and never mentions
compaction; the shape it misses is the one compact actually produces -- a row whose last
model-output run is a *summary*, preceded by a mask-0 summary request, followed by another
row of the same episode.

Both failures these guard against are silent. A broken seam does not raise; it just stops
crediting anything that happened after a compaction to anything before it.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from tensordict import TensorDict
from verl.trainer.ppo.core_algos import get_adv_estimator_fn

from vagen.algorithms._common.trajectory import TrajectoryView

TOKEN_GAE = get_adv_estimator_fn("token_level_gae")


class _Cfg(dict):
    gamma, lam = 1.0, 1.0

    def get(self, key, default=None):
        return getattr(self, key, default)


def _batch(scores, masks, values=None):
    scores = torch.tensor(scores, dtype=torch.float32)
    values = torch.zeros_like(scores) if values is None else torch.tensor(values, dtype=torch.float32)
    return TensorDict(
        {
            "token_level_scores": scores,
            "response_mask": torch.tensor(masks, dtype=torch.long),
            "values": values,
        },
        batch_size=[scores.shape[0]],
    )


def _nt(episode, turn, group=None, traj=None):
    columns = {
        "group_idx": np.array(group or ["g"] * len(episode), dtype=object),
        "traj_idx": np.array(traj if traj is not None else [0] * len(episode)),
        "turn_idx": np.array(turn),
    }
    if episode is not None:
        columns["episode_id"] = np.array(episode, dtype=object)
    return columns


# ------------------------------------------------------------------------- the seam
#
# A two-conversation compact episode, laid out as the harness produces it.
#
#   row 0:  [act act] [obs] [act] [request] [summary summary]
#   row 1:  [act act]                                            reward 1.0 at the end
#
# The summary is model output, so mask 1. The request is a user message into the closing
# conversation, so it sits in the response region at mask 0.

SEAM_MASKS = [[1, 1, 0, 1, 0, 1, 1], [1, 1, 0, 0, 0, 0, 0]]
SEAM_SCORES = [[0.0] * 7, [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]]


def test_credit_crosses_the_compaction_seam():
    """★ The property compact exists for. A reward earned after a compaction has to reach
    the actions taken before it. Under a row-local estimator row 0 gets nothing at all."""
    adv, ret = TOKEN_GAE(
        batch=_batch(SEAM_SCORES, SEAM_MASKS),
        non_tensor_batch=_nt(["ep", "ep"], [0, 1]),
        config=_Cfg(),
    )

    # returns are unwhitened and V is zero here, so they are the raw discounted rewards.
    first_action = float(ret[0, 0])
    assert first_action == pytest.approx(1.0), (
        f"the episode's first action was credited {first_action}, not the 1.0 earned after "
        "the seam -- the recursion stopped at the row boundary"
    )


def test_the_summary_is_an_action_and_is_credited():
    """The summary is model-emitted and determines the next conversation's state, so it is
    an action in the same chain. If it were treated as context it would carry no gradient
    and compaction would be unlearnable."""
    adv, ret = TOKEN_GAE(
        batch=_batch(SEAM_SCORES, SEAM_MASKS),
        non_tensor_batch=_nt(["ep", "ep"], [0, 1]),
        config=_Cfg(),
    )

    assert float(ret[0, 5]) == pytest.approx(1.0), "summary tokens must inherit later reward"
    assert float(ret[0, 6]) == pytest.approx(1.0)


def test_the_summary_request_costs_no_discount():
    """It is mask 0, so the recursion steps over it: with gamma < 1 the discount between
    the last real action and the first summary token must be one step, not one per
    request token."""
    cfg = _Cfg()
    cfg.gamma, cfg.lam = 0.5, 1.0
    _, ret = TOKEN_GAE(
        batch=_batch(SEAM_SCORES, SEAM_MASKS),
        non_tensor_batch=_nt(["ep", "ep"], [0, 1]),
        config=cfg,
    )

    # model tokens in emission order: (0,0) (0,1) (0,3) (0,5) (0,6) (1,0) (1,1)
    # reward 1.0 sits on the last of them, so token k back gets gamma**k.
    order = [(0, 0), (0, 1), (0, 3), (0, 5), (0, 6), (1, 0), (1, 1)]
    got = [float(ret[r, c]) for r, c in order]
    expected = [0.5 ** k for k in range(len(order) - 1, -1, -1)]
    assert got == pytest.approx(expected), (
        "the discount ladder is wrong -- a masked token or a row boundary consumed a step"
    )


def test_observations_between_turns_cost_no_discount_either():
    """Same claim for the ordinary in-conversation observation at (0, 2)."""
    cfg = _Cfg()
    cfg.gamma, cfg.lam = 0.5, 1.0
    _, ret = TOKEN_GAE(
        batch=_batch(SEAM_SCORES, SEAM_MASKS),
        non_tensor_batch=_nt(["ep", "ep"], [0, 1]),
        config=cfg,
    )
    # (0,1) and (0,3) are adjacent actions separated only by the observation at (0,2).
    assert float(ret[0, 1]) == pytest.approx(0.5 * float(ret[0, 3]))


def test_a_dropped_conversation_leaves_a_gap_in_turn_idx():
    """Conversation ordinals are assigned when a conversation opens and are not returned
    when one turns out to carry no model output, so turn_idx can skip. Ordering must come
    from the values, not from their being consecutive."""
    view = TrajectoryView.build(
        torch.tensor([[1, 1], [1, 1], [1, 1]], dtype=torch.long),
        _nt(["ep"] * 3, [0, 2, 5]),
    )
    assert view.trajectories == [[0, 1, 2]]


# -------------------------------------------------------------- what identifies an episode


def test_episode_id_is_preferred_over_the_rollout_pair():
    """★ Two distinct episodes that happen to share (group_idx, traj_idx) must stay
    separate. They already do collide on the validation path, where padding a batch to a
    multiple of the worker count duplicates prompts that then run as different episodes.
    Merging them is silent: the recursion runs through both and the earlier episode is
    credited with the later one's reward."""
    masks = [[1, 1], [1, 1]]
    view = TrajectoryView.build(
        torch.tensor(masks, dtype=torch.long),
        _nt(["A", "B"], [0, 1], group=["g", "g"], traj=[0, 0]),
    )
    assert view.trajectories == [[0], [1]], "two episodes were merged into one chain"


def test_without_episode_id_it_falls_back_to_the_pair():
    """Older loops and the estimator's own unit tests do not emit the column."""
    masks = [[1, 1], [1, 1]]
    nt = _nt(None, [0, 1], group=["g", "g"], traj=[0, 0])
    assert "episode_id" not in nt
    view = TrajectoryView.build(torch.tensor(masks, dtype=torch.long), nt)
    assert view.trajectories == [[0, 1]]


def test_the_merge_would_actually_corrupt_the_advantage():
    """The consequence, so the guard above is not merely structural. Episode A earns
    nothing, episode B earns 1.0; merged, A's tokens inherit B's reward."""
    masks = [[1, 1], [1, 1]]
    scores = [[0.0, 0.0], [0.0, 1.0]]

    separate, _ = TOKEN_GAE(
        batch=_batch(scores, masks),
        non_tensor_batch=_nt(["A", "B"], [0, 1], group=["g", "g"], traj=[0, 0]),
        config=_Cfg(),
    )
    merged, _ = TOKEN_GAE(
        batch=_batch(scores, masks),
        non_tensor_batch=_nt(None, [0, 1], group=["g", "g"], traj=[0, 0]),
        config=_Cfg(),
    )

    # Under the merge, A's tokens are credited; kept apart, they are not.
    assert float(separate[0, 0]) != pytest.approx(float(merged[0, 0]))
