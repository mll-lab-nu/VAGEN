# Evaluate

Run LLM agents across multiple environment specs and collect rollout metrics.

## Quick Start

```bash
python -m vagen.evaluate.run_eval --config path/to/config.yaml
```

Override any field via CLI:
```bash
python -m vagen.evaluate.run_eval --config config.yaml run.backend=claude run.max_concurrent_jobs=8
```

## Config Composition

Configs support a `defaults:` key to inherit from base configs, so you only
need to write the fields you want to override.

### How `defaults` paths work

Paths in `defaults:` are **relative to the config file itself**. The `.yaml`
extension is appended automatically if omitted.

```
VAGEN/
├── vagen/configs/eval_default.yaml          # VAGEN base config
├── examples/evaluate/
│   ├── sokoban/config.yaml                  # defaults: [../../../vagen/configs/eval_default]
│   └── frozenlake/config.yaml               # same relative path
```

```
ViewSuite/
├── examples/evaluation/
│   ├── eval_default.yaml                    # ViewSuite base config
│   └── eval_proxy_task/
│       └── gpt_5_4_pro.yaml                # defaults: [../eval_default]
```

### `default_chat_config`

A top-level `default_chat_config` key can be used to set chat parameters for
all envs that don't define their own `chat_config`. This works across files
via the `defaults` merge — the child config can override `default_chat_config`
without redefining all envs.

### Example: Minimal proxy_task config

```yaml
defaults:
  - ../eval_default

default_chat_config:
  temperature: 0.0
  max_tokens: 10000
  extra_body:
    reasoning:
      effort: "medium"

experiment:
  dump_dir: ${fileroot}/rollouts/my_model

backends:
  openai:
    model: "openai/gpt-5.4-pro"
```

### Example: Custom envs with native backend

```yaml
defaults:
  - ../eval_default

envs:
  - name: ScannetTool
    n_envs: 545
    tag_id: 21
    ...
    chat_config:             # per-env chat_config overrides default_chat_config
      max_completion_tokens: 32768
      reasoning_effort: "high"

run:
  backend: "azure"

backends:
  azure:
    azure_endpoint: "https://..."
    deployment: "gpt-5"
```

### Multiple defaults

```yaml
defaults:
  - shared/backends
  - shared/viewsuite_envs
```

Nested `defaults:` are also supported (a base config can reference its own defaults).

## Config Structure

| Section | Description |
|---|---|
| `default_chat_config` | Default chat params applied to envs without inline `chat_config` |
| `envs` | List of environment specs (name, n_envs, seed, config, chat_config) |
| `experiment` | `dump_dir`, `default_max_turns` |
| `run` | `backend`, `max_concurrent_jobs`, `resume`, `live_summary` |
| `backends` | Per-backend settings (openai, azure, sglang, vllm, claude, gemini, etc.) |
