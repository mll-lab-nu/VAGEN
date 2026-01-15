

from verl import DataProto
from typing import Any
from narwhals import Enum


METRIC_REGISTRY: dict[str, Any] = {}


def register_metric(name_or_enum: str) -> Any:
    """Decorator to register a advantage estimator function with a given name.

    Args:
        name_or_enum: `(str)` or `(AdvantageEstimator)`
            The name or enum of the advantage estimator.

    """

    def decorator(fn):
        name = name_or_enum.value if isinstance(name_or_enum, Enum) else name_or_enum
        if name in METRIC_REGISTRY and METRIC_REGISTRY[name] != fn:
            raise ValueError(
                f"Adv estimator {name} has already been registered: {METRIC_REGISTRY[name]} vs {fn}"
            )
        METRIC_REGISTRY[name] = fn
        return fn

    return decorator

@register_metric("reward_variance")
def reward_variance(data: DataProto) -> float:
    """Compute the variance of rewards in the given DataProto.

    Args:
        data: `(DataProto)`
            The data containing rewards.

    Returns:
        float: The variance of total rewards across groups
    """
    import torch
    import numpy as np
    from collections import defaultdict

    token_level_scores = data.batch["token_level_scores"]
    if "group_idx" in data.non_tensor_batch:  # optional
        group_idx = data.non_tensor_batch["group_idx"]
    else:
        group_idx = data.non_tensor_batch["uid"]

    # Calculate total reward per sample (sum over token dimension)
    if isinstance(token_level_scores, torch.Tensor):
        total_rewards = token_level_scores.sum(dim=-1).cpu().numpy()
    else:
        total_rewards = np.array(token_level_scores).sum(axis=-1)

    # Group rewards by group_idx
    group_rewards = defaultdict(list)
    for idx, reward in zip(group_idx, total_rewards):
        group_rewards[str(idx)].append(reward)

    # Calculate mean reward for each group
    group_mean_rewards = [np.mean(rewards) for rewards in group_rewards.values()]

    # Calculate variance across groups
    if len(group_mean_rewards) > 1:
        variance = np.var(group_mean_rewards)
    else:
        variance = 0.0

    return float(variance)