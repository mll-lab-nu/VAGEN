# Configuration

The main config file is [`vagen/configs/vagen_multiturn.yaml`](https://github.com/RAGEN-AI/VAGEN/blob/main/vagen/configs/vagen_multiturn.yaml).

## Trainer

```yaml
trainer:
  skip_special_tokens_val: False
  skip_special_tokens_train: False
  replace_image_tokens_for_logging: True
  log_image:
    enable: True
    max_pending: 2
    png_compress_level: 0
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `skip_special_tokens_val` | bool | `False` | Skip special tokens when logging validation outputs |
| `skip_special_tokens_train` | bool | `False` | Skip special tokens when logging training outputs |
| `replace_image_tokens_for_logging` | bool | `True` | Replace `<image>` tokens with placeholder for cleaner logs |
| `log_image.enable` | bool | `True` | Enable image logging to local folder |
| `log_image.max_pending` | int | `2` | Maximum pending image uploads |
| `log_image.png_compress_level` | int | `0` | PNG compression level (0=none, 9=max) |


## Filter

```yaml
filter:
  name: reward_variance
  filter_kwargs: {}
  enable: False
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `name` | str | `reward_variance` | Filter name (must be registered via `@register_filter`) |
| `filter_kwargs` | dict | `{}` | Arguments passed to filter function as `**kwargs` |
| `enable` | bool | `False` | Enable/disable filtering |

See [Custom Filter](custom-filter.md) for creating your own filters.

### Built-in Filter Options

**`reward_variance`**
```yaml
filter:
  name: reward_variance
  filter_kwargs:
    topk: 0.2      # Keep top 20% groups by variance
    ddof: 0        # Degrees of freedom for variance calculation
  enable: True
```

**`reward_variance_top_p`**
```yaml
filter:
  name: reward_variance_top_p
  filter_kwargs:
    top_p: 0.9     # Keep groups until 90% cumulative variance
    ddof: 0
  enable: True
```


