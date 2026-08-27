# Custom Metric

VAGEN supports custom metrics for W&B logging during training.

## Built-in Metrics

Three are registered in `vagen/training/metrics/episode/`:

| name | what it reports |
|---|---|
| `episode_score` | the episode-level reward. Under `no_concat` and `compact` a row is one turn, so the per-row mean reads as `episode_reward / rows_per_episode` — measured 17x low. This is what `critic/score/mean` points at. |
| `episode_turns` | environment steps per episode |
| `reward_variance` | variance of reward within a group |

Two of them return `dict[str, float]` rather than a single float; `collect_registry_metrics`
(`vagen/training/trainer/logic.py`) supports both. The registry calls `fn(data)` with no extra
arguments — there is no `**kwargs` mechanism.

## Creating a Custom Metric

### Step 1: Create Your Metric

Add your metric in its own directory under [`vagen/training/metrics`](https://github.com/mll-lab-nu/VAGEN/tree/main/vagen/training/metrics):

```python
from vagen.training.metrics import register_metric
from verl import DataProto

@register_metric("my_metric")
def my_metric(data: DataProto) -> float:
    """
    Custom metric implementation.

    Args:
        data: DataProto containing training data

    Returns:
        float: Metric value for logging
    """
    import torch
    import numpy as np

    # Access training data
    token_level_scores = data.batch["token_level_scores"]
    group_idx = data.non_tensor_batch["group_idx"]

    # Compute your metric
    if isinstance(token_level_scores, torch.Tensor):
        total_rewards = token_level_scores.sum(dim=-1).detach().cpu().numpy()
    else:
        total_rewards = np.asarray(token_level_scores).sum(axis=-1)

    # Example: return mean reward
    return float(np.mean(total_rewards))
```

### Step 2: Use the Metric

Metrics registered with `@register_metric` are automatically available for use during training. They are logged to W&B.

## Example: Reward Variance Metric

The built-in `reward_variance` metric computes the mean within-group reward variance:

```python
@register_metric("reward_variance")
def reward_variance(data: DataProto, ddof=0) -> float:
    """Compute mean of within-group reward variances."""
    import torch
    import numpy as np
    from collections import defaultdict

    token_level_scores = data.batch["token_level_scores"]
    group_idx = data.non_tensor_batch["group_idx"]

    # 1) Total reward per sample
    if isinstance(token_level_scores, torch.Tensor):
        total_rewards = token_level_scores.sum(dim=-1).detach().cpu().numpy()
    else:
        total_rewards = np.asarray(token_level_scores).sum(axis=-1)

    # 2) Group rewards
    group_rewards = defaultdict(list)
    for idx, reward in zip(group_idx, total_rewards):
        group_rewards[str(idx)].append(float(reward))

    # 3) Per-group variance, then mean
    per_group_vars = []
    for rewards in group_rewards.values():
        if len(rewards) <= 1:
            per_group_vars.append(0.0)
        else:
            per_group_vars.append(float(np.var(rewards, ddof=ddof)))

    return float(np.mean(per_group_vars)) if per_group_vars else 0.0
```
