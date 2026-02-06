# Custom Filter

VAGEN supports custom filters to preprocess training data before optimization. This feature is inspired by [RAGEN](https://github.com/RAGEN-AI/RAGEN).

## Built-in Filters

| Filter | Description |
|--------|-------------|
| `reward_variance` | Keep top-k groups by reward variance |
| `reward_variance_top_p` | Keep groups until cumulative variance reaches top-p |

## Creating a Custom Filter

### Step 1: Create Your Filter

Add your filter in [`vagen/custom_filter/filter.py`](https://github.com/mll-lab-nu/VAGEN/blob/main/vagen/custom_filter/filter.py):

```python
from vagen.custom_filter.filter import register_filter
from verl import DataProto

@register_filter("my_filter")
def my_filter(data_proto: DataProto, metrics: dict, **kwargs) -> tuple[DataProto, dict]:
    """
    Custom filter implementation.

    Args:
        data_proto: Input data containing batch and non_tensor_batch
        metrics: Metrics dict to update for W&B logging
        **kwargs: Additional arguments from filter.filter_kwargs config

    Returns:
        filtered_data: DataProto with filtered samples
        metrics: Updated metrics dict
    """
    # Get parameters from config
    threshold = kwargs.get("threshold", 0.5)

    # Access data
    token_level_scores = data_proto.batch["token_level_scores"]
    group_idx = data_proto.non_tensor_batch["group_idx"]

    # Your filtering logic: determine which indices to keep
    keep_indices = []
    for i, score in enumerate(token_level_scores):
        if score.sum() > threshold:
            keep_indices.append(i)

    # Apply filter
    filtered_data = data_proto.select_idxs(keep_indices)

    # Update metrics for logging (optional)
    metrics["filter/kept_ratio"] = len(keep_indices) / len(data_proto)

    return filtered_data, metrics
```

### Step 2: Enable in Config

Update [`vagen/configs/vagen_multiturn.yaml`](https://github.com/mll-lab-nu/VAGEN/blob/main/vagen/configs/vagen_multiturn.yaml):

```yaml
filter:
  name: my_filter
  filter_kwargs:
    threshold: 0.5
  enable: True
```

## Example: Reward Variance Filter

The built-in `reward_variance` filter keeps only groups with the highest reward variance:

```python
@register_filter("reward_variance")
def reward_variance_filter(data_proto: DataProto, metrics, **kwargs) -> tuple[DataProto, dict]:
    topk_ratio = kwargs.get("topk", 0.2)  # Keep top 20% groups

    # 1) Calculate total reward per sample
    token_level_scores = data_proto.batch["token_level_scores"]
    total_rewards = token_level_scores.sum(dim=-1)

    # 2) Group rewards by group_idx
    group_idx = data_proto.non_tensor_batch["group_idx"]
    group_rewards = defaultdict(list)
    for i, (gid, reward) in enumerate(zip(group_idx, total_rewards)):
        group_rewards[gid].append((i, reward))

    # 3) Calculate variance for each group
    group_variances = {
        gid: np.var([r for _, r in rewards])
        for gid, rewards in group_rewards.items()
    }

    # 4) Select top-k groups by variance
    sorted_groups = sorted(group_variances.items(), key=lambda x: x[1], reverse=True)
    num_keep = max(1, int(len(sorted_groups) * topk_ratio))
    top_groups = set(gid for gid, _ in sorted_groups[:num_keep])

    # 5) Collect indices
    keep_indices = [i for gid, rewards in group_rewards.items()
                    if gid in top_groups for i, _ in rewards]

    return data_proto.select_idxs(keep_indices), metrics
```

## Configuration Reference

```yaml
filter:
  name: reward_variance      # Filter name (must be registered)
  filter_kwargs:             # Passed to filter function as **kwargs
    topk: 0.2                # Example: keep top 20% groups
  enable: False              # Set to True to enable filtering
```
