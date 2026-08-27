

from verl import DataProto
from vagen.training.metrics._common.registry import register_metric

@register_metric("episode_score")
def episode_score(data: DataProto) -> dict[str, float]:
    """Reward per *episode*, which is the only form comparable across harnesses.

    verl's ``critic/score/mean`` averages ``token_level_scores.sum(-1)`` over rows. In
    concat an episode is one row and that is the episode reward; in no_concat and compact
    an episode becomes several rows, and whether the reward is spread over them or written
    on the last one, the per-row mean comes out as ``episode_reward / rows_per_episode``.

    Measured on the 2026-08-12 sweep, where this silently inverted the ranking: the two
    concat arms agreed with validation to three decimals (0.571 vs 0.573, 0.843 vs 0.823)
    while compact read 0.539 against a validation 0.776 and no_concat read 0.375 against a
    validation 0.918 -- each low by exactly its rows-per-episode factor (1.44 and 2.45).
    By the row metric no_concat was the worst arm; by episode it is the best.

    Summed within an episode, then reduced over episodes, so it equals what validation
    reports regardless of how the harness split the episode up.
    """
    from collections import defaultdict

    import numpy as np
    import torch

    scores = data.batch.get("token_level_scores")
    if scores is None:
        raise KeyError("token_level_scores is not on the batch")
    per_row = (
        scores.sum(dim=-1).detach().cpu().numpy()
        if isinstance(scores, torch.Tensor)
        else np.asarray(scores).sum(axis=-1)
    )

    episode_ids = data.non_tensor_batch.get("episode_id")
    if episode_ids is None:
        # Concat already has one row per episode, so the row mean is the episode mean.
        totals = np.asarray(per_row, dtype=float)
    else:
        acc: dict[str, float] = defaultdict(float)
        for eid, s in zip(episode_ids, per_row):
            acc[str(eid)] += float(s)
        totals = np.array(list(acc.values()), dtype=float)

    if totals.size == 0:
        return {"min": 0.0, "max": 0.0, "mean": 0.0}
    return {"min": float(totals.min()), "max": float(totals.max()), "mean": float(totals.mean())}


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
