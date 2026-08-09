"""What the advantage is built from, and what the critic metrics measure.

Both were wrong in ways that hide themselves. The estimators read the un-penalised
scores, so the KL penalty was computed and never used. And the critic metrics averaged
the -100 sentinel that marks unsupervised positions, which makes
``critic/vf_explained_var`` meaningless -- the one number whose job is to tell you the
critic is broken.
"""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from vagen.custom_advantage import AdvantageInputs  # noqa: E402


def _inputs(**columns):
    return AdvantageInputs(columns, {}, None, "probe")


def test_the_penalised_reward_is_preferred_when_present():
    """verl writes the KL penalty into token_level_rewards and leaves the scores alone."""
    inputs = _inputs(
        token_level_scores=torch.tensor([[1.0, 2.0]]),
        token_level_rewards=torch.tensor([[0.5, 1.5]]),
    )
    assert torch.equal(inputs.rewards, torch.tensor([[0.5, 1.5]])), (
        "use_kl_in_reward is a silent no-op: the penalty is stored and never read"
    )


def test_the_scores_are_used_when_there_is_no_penalty():
    inputs = _inputs(token_level_scores=torch.tensor([[1.0, 2.0]]))
    assert torch.equal(inputs.rewards, torch.tensor([[1.0, 2.0]]))


def test_no_estimator_reads_the_raw_scores_directly():
    """★ One of them reaching past `inputs.rewards` is the whole bug back again.

    The previous version of this test split the module on a helper that no longer exists
    and then searched the *whole file* when the split failed -- so it was asserting
    against a string that had moved, not against the estimators.
    """
    import ast
    import inspect

    from vagen.custom_advantage import trajectory_algos

    tree = ast.parse(inspect.getsource(trajectory_algos))
    offenders = [
        node.name
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and any("advantage_estimator" in ast.unparse(d) for d in node.decorator_list)
        and ("token_level_scores" in ast.unparse(node) or "inputs.scores" in ast.unparse(node))
    ]
    assert not offenders, (
        f"{offenders} read the un-penalised scores; use inputs.rewards or "
        "algorithm.use_kl_in_reward silently does nothing"
    )


def test_critic_metrics_ignore_the_unsupervised_sentinel():
    """★ Behavioural, not a source grep. The previous version asserted that the string
    "value_mask" appeared in the function -- which an explanatory *comment* satisfies, so
    deleting the narrowing itself left the test green while critic/returns/mean went back
    to reporting about -87."""
    from verl import DataProto
    from verl.trainer.ppo.metric_utils import compute_data_metrics

    from vagen.trainer.logic import IGNORE_RETURN

    n, width = 2, 4
    returns = torch.full((n, width), IGNORE_RETURN)
    returns[:, 0] = 0.75                       # one supervised anchor per row
    values = torch.zeros(n, width)
    values[:, 0] = 0.5
    value_mask = torch.zeros(n, width, dtype=torch.long)
    value_mask[:, 0] = 1

    batch = DataProto.from_single_dict({
        "returns": returns,
        "values": values,
        "value_mask": value_mask,
        "advantages": torch.zeros(n, width),
        "token_level_scores": torch.zeros(n, width),
        "token_level_rewards": torch.zeros(n, width),
        "response_mask": torch.ones(n, width, dtype=torch.long),
        "attention_mask": torch.ones(n, width * 2, dtype=torch.long),
        "prompts": torch.zeros(n, width, dtype=torch.long),
        "responses": torch.zeros(n, width, dtype=torch.long),
    })
    metrics = compute_data_metrics(batch, use_critic=True)

    assert metrics["critic/returns/mean"] == pytest.approx(0.75), (
        f"critic/returns/mean is {metrics['critic/returns/mean']}, so the -100 sentinel "
        "is being averaged in"
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
