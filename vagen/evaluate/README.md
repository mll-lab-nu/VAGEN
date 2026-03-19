# Evaluate

Run LLM agents across vision-based gym environments and collect rollout metrics.

## 1. Config

```bash
python -m vagen.evaluate.run_eval --config path/to/config.yaml
```

CLI overrides (OmegaConf dotlist):
```bash
python -m vagen.evaluate.run_eval --config config.yaml run.backend=claude backends.claude.model=claude-opus-4-6
```

Below is a complete example (`examples/evaluate/frozenlake/config.yaml`):

```yaml
defaults:
  - ../../../vagen/configs/eval_default   # inherit shared backend definitions

fileroot: ${oc.env:HOME}/projects/vagen

envs:
  - name: FrozenLake                      # registered env class name
    n_envs: 128                           # how many episodes to run
    tag_id: frozenlake_test               # groups rollout outputs under tag_{tag_id}/
    seed: [0,128,1]                       # [start, end, step] → generates seeds 0..127
    max_turns: 5                          # max agent–env interaction turns per episode
    config:                               # passed to the env constructor
      render_mode: vision
      size: 4
      p: 0.8
      is_slippery: false
      slip_prob: 0.0
      max_actions_per_step: 5

experiment:
  dump_dir: ${fileroot}/rollouts/eval_frozenlake   # rollout output root
  default_max_turns: 5                              # fallback if env omits max_turns

run:
  backend: "openai"              # which backend to use (see backends section)
  base_seed: 0                   # global seed offset added to all env seeds
  max_concurrent_jobs: 64        # max episodes running in parallel
  resume: skip_completed         # skip_completed | off | force_rerun
  live_summary: true             # write summary.json after each episode finishes

backends:
  sglang:                                 # for local model serving
    base_url: "http://127.0.0.1:30000/v1"
    api_key: "EMPTY"
    model: ""                             # set by sglang launch script
    max_concurrency: 2
    max_retries: 6
    min_backoff: 0.5
    max_backoff: 8.0

  openai:                                 # for API-based models
    api_key: ""                           # or export OPENAI_API_KEY
    base_url: null
    model: "gpt-4.1-mini"
    max_concurrency: 8                    # max concurrent API requests (rate limit gate)
    max_retries: 6                        # retry count on transient errors
    min_backoff: 0.5                      # exponential backoff lower bound (seconds)
    max_backoff: 8.0                      # exponential backoff upper bound (seconds)
```

### Parameter reference

**`defaults`** — List of base YAML files to inherit from (paths relative to this config file, `.yaml` auto-appended). Deep-merged in order, then this config merges on top.

**`envs[]`** — Each entry defines a batch of episodes:

| Field | Type | Description |
|---|---|---|
| `name` | str | Registered environment class (e.g. `FrozenLake`, `Sokoban`, `RemoteEnv`, `ScannetTool`) |
| `n_envs` | int | Number of episodes to run |
| `tag_id` | int/str | Output subdirectory name: `tag_{tag_id}/` |
| `seed` | list | `[start, end, step]` to generate a range, or explicit list of seeds |
| `max_turns` | int | Max agent–env turns per episode |
| `split` | str | Dataset split identifier (default: `"default"`) |
| `config` | dict | Kwargs passed to the environment constructor |
| `chat_config` | dict | Kwargs passed to the LLM completion call (temperature, max_tokens, etc.) |
| `concat_multi_turn` | bool | `true`: send full message history; `false`: only system + last turn (default: `true`) |

**`default_chat_config`** — Top-level fallback: applied to any env that doesn't define its own `chat_config`.

**`experiment`**:
- `dump_dir` — Root directory for rollout outputs
- `default_max_turns` — Fallback max_turns if env doesn't specify one

**`run`**:
- `backend` — Which backend to use: `openai` | `azure` | `sglang` | `vllm` | `together` | `claude` | `gemini` | `openai_responses` | `azure_responses`
- `max_concurrent_jobs` — Episode-level parallelism (how many episodes run at once)
- `resume` — `skip_completed` skips episodes with existing successful metrics; `off` reruns everything; `force_rerun` forces rerunning all episodes regardless of existing successful metrics (overrides `skip_completed` behavior)
- `live_summary` — Refresh `summary.json` after each episode

**`backends.{name}`** — Config for each backend:
- `api_key`, `base_url` — API credentials (or set via env vars)
- `model` — Model identifier
- `max_concurrency` — Request-level concurrency gate (API rate limit)
- `max_retries`, `min_backoff`, `max_backoff` — Retry policy with exponential backoff

### Output structure

```text
dump_dir/
└── tag_{tag_id}/
    ├── summary.json                    # aggregated metrics
    └── {env_name}_seed_{seed}/
        ├── metrics.json                # per-episode results (success, reward, finish_reason)
        ├── messages.json               # full conversation history
        ├── assistant_texts.json        # model replies only
        ├── transcript.txt              # human-readable conversation
        └── images/
            └── turn_00_00.png          # observation images per turn
```

## 2. Scripts

Typical run script:

```bash
#!/bin/bash
# Run FrozenLake eval with sglang backend
cd /path/to/VAGEN
python -m vagen.evaluate.run_eval \
    --config examples/evaluate/frozenlake/config.yaml
```

Override model or backend on the fly:

```bash
# Switch to OpenAI
python -m vagen.evaluate.run_eval \
    --config examples/evaluate/frozenlake/config.yaml \
    run.backend=openai \
    backends.openai.model=gpt-4o-mini \
    experiment.dump_dir=./rollouts/gpt4o_mini
```

## 3. Custom Adapters

To add a new backend, implement `ModelAdapter` and register it:

```python
# my_adapter.py
from vagen.evaluate.adapters.base_adapter import ModelAdapter
from vagen.evaluate.registry import register_adapter, register_client

# Step 1: Register client factory
@register_client("my_backend")
def build_my_client(cfg):
    return MyAsyncClient(api_key=cfg.get("api_key"), base_url=cfg.get("base_url"))

# Step 2: Implement and register adapter
@register_adapter("my_backend")
class MyAdapter(ModelAdapter):

    def __init__(self, client, model: str):
        self.client = client
        self.model = model

    def format_system(self, text, images):
        # Convert system prompt + images to your API's message format
        return {"role": "system", "content": ...}

    def format_user_turn(self, text, images):
        # Convert user observation + images to your API's message format
        return {"role": "user", "content": ...}

    async def acompletion(self, messages, **chat_config):
        # Call your API and return the text response
        resp = await self.client.generate(model=self.model, messages=messages, **chat_config)
        return resp.text

    def is_retryable_error(self, exc):
        # Optional: customize retry behavior
        # Return True (retry), False (don't retry), or None (use default logic)
        return None
```

Then make sure it's imported in `register_builtins.py`:

```python
import my_adapter  # triggers @register_client and @register_adapter
```

Now use it in config:

```yaml
run:
  backend: "my_backend"

backends:
  my_backend:
    api_key: ""
    base_url: "http://..."
    model: "my-model"
    max_concurrency: 4
    max_retries: 6
    min_backoff: 0.5
    max_backoff: 8.0
```
