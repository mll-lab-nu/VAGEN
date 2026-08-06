"""What the advantage is built from, and what the critic metrics measure.

Both were wrong in ways that hide themselves. The estimators read the un-penalised
scores, so the KL penalty was computed and never used. And the critic metrics averaged
the -100 sentinel that marks unsupervised positions, which makes
``critic/vf_explained_var`` meaningless -- the one number whose job is to tell you the
critic is broken.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from vagen.custom_advantage.trajectory_algos import rewards_for_advantage  # noqa: E402


def test_the_penalised_reward_is_preferred_when_present():
    """verl writes the KL penalty into token_level_rewards and leaves the scores alone."""
    batch = {
        "token_level_scores": torch.tensor([[1.0, 2.0]]),
        "token_level_rewards": torch.tensor([[0.5, 1.5]]),
    }
    assert torch.equal(rewards_for_advantage(batch), batch["token_level_rewards"]), (
        "use_kl_in_reward is a silent no-op: the penalty is stored and never read"
    )


def test_the_scores_are_used_when_there_is_no_penalty():
    batch = {"token_level_scores": torch.tensor([[1.0, 2.0]])}
    assert torch.equal(rewards_for_advantage(batch), batch["token_level_scores"])


def test_every_estimator_goes_through_the_helper():
    """One of them reading the raw key is the whole bug back again."""
    import inspect

    from vagen.custom_advantage import no_concat_gae, trajectory_algos

    for module in (trajectory_algos, no_concat_gae):
        src = inspect.getsource(module)
        body = src.split("def rewards_for_advantage", 1)
        body = body[1].split("\n\n\n", 1)[1] if len(body) > 1 else src
        assert 'batch["token_level_scores"]' not in body, (
            f"{module.__name__} still reads the un-penalised scores directly"
        )


def test_critic_metrics_ignore_the_unsupervised_sentinel():
    """The sentinel marks positions with no return. Averaging it in gave
    critic/returns/mean = -87 where the supervised mean was 0.75."""
    import inspect

    from verl.trainer.ppo import metric_utils

    src = inspect.getsource(metric_utils.compute_data_metrics)
    assert "value_mask" in src, "the metrics no longer know about the sentinel"
    assert "masked_select(returns, returns_mask)" in src
    assert "masked_select(values, returns_mask)" in src, (
        "values and returns must be masked the same way or explained variance "
        "compares different position sets"
    )


# ------------------------------------------------ colliding trajectory keys
def _view(masks, group, traj, turn=None):
    import numpy as np

    from vagen.custom_advantage.trajectory import TrajectoryView

    nt = {"group_idx": np.array(group, dtype=object),
          "traj_idx": np.array(traj, dtype=object)}
    if turn is not None:
        nt["turn_idx"] = np.array(turn, dtype=object)
    return TrajectoryView.build(torch.tensor(masks, dtype=torch.bool), nt)


def test_padding_copies_are_still_deduplicated_silently():
    """pad_dataproto_to_divisor appends exact copies; folding them is correct."""
    v = _view([[1, 1], [1, 1]], ["g", "g"], [0, 0], [0, 0])
    assert v is not None


def test_rows_that_merely_collide_are_refused():
    """A loop that stopped emitting turn_idx collapses every row onto turn 0. Folding
    those hands one row an advantage computed for a different response."""
    with pytest.raises(ValueError, match="different response"):
        _view([[1, 1, 0, 0], [0, 0, 1, 1]], ["g", "g"], [0, 0], [0, 0])


def test_distinct_turns_are_untouched():
    v = _view([[1, 1, 0, 0], [0, 0, 1, 1]], ["g", "g"], [0, 0], [0, 1])
    assert v is not None
