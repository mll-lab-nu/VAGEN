

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

@register_metric("episode_turns")
def episode_turns(data: DataProto) -> dict[str, float]:
    """How many turns the episodes in this batch actually ran.

    verl's own ``num_turns/*`` metric reads ``__num_turns__``, which VAGEN fills with 1
    per row by construction -- so the only turn count on the dashboard reports every
    episode as single-turn no matter how long it was, and a run whose episodes collapse
    to one turn (a broken action parser, say) looks exactly like a healthy one. The agent
    loop already records the real count as ``episode_turns``; this is what puts it where
    it can be read.

    Deduplicated by ``episode_id``: in no_concat and compact mode one episode becomes
    several rows that each carry the same count, and averaging over rows would weight
    long episodes by how many rows they happened to be split into. In concat mode an
    episode is one row and the dedup is a no-op.
    """
    import numpy as np

    turns = data.non_tensor_batch.get("episode_turns")
    if turns is None:
        raise KeyError("episode_turns is not on the batch; the agent loop did not record it")

    episode_ids = data.non_tensor_batch.get("episode_id")
    if episode_ids is not None:
        per_episode: dict[str, float] = {}
        for eid, t in zip(episode_ids, turns):
            per_episode.setdefault(str(eid), float(t))
        values = np.array(list(per_episode.values()), dtype=float)
    else:
        values = np.asarray(turns, dtype=float)

    if values.size == 0:
        return {"min": 0.0, "max": 0.0, "mean": 0.0}
    return {
        "min": float(values.min()),
        "max": float(values.max()),
        "mean": float(values.mean()),
    }


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
