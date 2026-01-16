

from verl import DataProto
from typing import Any
from enum import Enum
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
def reward_variance(data: DataProto,ddof = 0) -> float:
    """Compute mean of within-group reward variances.

    Steps:
      1) total_reward per sample = sum(token_level_scores over token dim)
      2) for each group: compute variance(total_reward within group)
      3) return mean(variance_per_group)

    Returns:
        float: Mean of within-group variances of total rewards.
    """
    import torch
    import numpy as np
    from collections import defaultdict

    token_level_scores = data.batch["token_level_scores"]
    group_idx = (
        data.non_tensor_batch["group_idx"]
        if "group_idx" in data.non_tensor_batch
        else data.non_tensor_batch["uid"]
    )

    # 1) total reward per sample
    if isinstance(token_level_scores, torch.Tensor):
        total_rewards = token_level_scores.sum(dim=-1).detach().cpu().numpy()
    else:
        total_rewards = np.asarray(token_level_scores).sum(axis=-1)

    # 2) group rewards
    group_rewards = defaultdict(list)
    for idx, reward in zip(group_idx, total_rewards):
        group_rewards[str(idx)].append(float(reward))

    # 3) per-group variance, then mean
    # ddof=0 => population variance; change to 1 if you want sample variance
    per_group_vars = []
    for rewards in group_rewards.values():
        if len(rewards) <= 1:
            per_group_vars.append(0.0)
        else:
            per_group_vars.append(float(np.var(rewards, ddof=ddof)))

    if len(per_group_vars) == 0:
        return 0.0

    return float(np.mean(per_group_vars))
