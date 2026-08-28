# Evaluation

Run LLM agents across vision-based gym environments and collect rollout metrics.

## 1. Config

```bash
python -m vagen.evaluation --config path/to/config.yaml
```

CLI overrides (OmegaConf dotlist):
```bash
python -m vagen.evaluation --config config.yaml run.backend=claude backends.claude.model=claude-opus-4-6
```

Below is a complete example, abridged from `examples/evaluate/frozenlake/config.yaml` —
read that file for the authoritative version:

```yaml
defaults:
  - ../../../vagen/configs/eval_default   # inherit shared backend definitions

fileroot: ${oc.env:VAGEN_EVAL_ROOT,./eval_runs}   # relative to THIS file's directory

envs:
  - name: FrozenLake                      # registered env class name
    n_envs: 128                           # how many episodes to run
    tag_id: frozenlake_test               # groups rollout outputs under tag_{tag_id}/
    seed: [20000,20127,1]                 # [min, max, occurrence-limit]; INCLUSIVE, so this
                                          # is 128 values for n_envs=128. Above the training
                                          # ranges on purpose -- same seed, same instance.
    max_turns: 5                          # max agent–env interaction turns per episode
    response_length_per_turn: 512         # ★ becomes the API call's max_tokens. Copy it
                                          # from the val config: without it the client
                                          # falls back to chat_config.max_tokens below,
                                          # and the policy gets room it never trained with
    config:                               # passed to the env constructor
      render_mode: vision
      size: 4
    chat_config:                          # ★ per-env sampling. Pin the temperature:
      temperature: 0                      # unset, the provider's default applies (1.0),
      max_tokens: 1024                    # so two checkpoints are not compared on equal
      top_p: 1.0                          # terms and a rerun does not reproduce
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
| `name` | str | Registered environment class (e.g. `FrozenLake`, `Sokoban`, `RemoteEnv`, `SpatialGym`) |
| `n_envs` | int | Number of episodes to run |
| `tag_id` | int/str | Output subdirectory name: `tag_{tag_id}/` |
| `seed` | list | `[base]`, `[min, max]`, or `[min, max, occurrence-limit]`. **Inclusive**, and the third element is a per-value cap, *not* a step. Explicit seeds go in `seed_list` |
| `seed_list` | list | Explicit seeds, at least `n_envs` of them; overrides `seed` |
| `max_turns` | int | Max agent–env turns per episode |
| `split` | str | Dataset split identifier (default: `"default"`) |
| `config` | dict | Kwargs passed to the environment constructor |
| `chat_config` | dict | Kwargs passed to the LLM completion call (temperature, etc.). `max_tokens` is clamped to `response_length_per_turn` when that is set |
| `harness` | str | Context policy: `concat` (default) \| `no_concat` \| `compact`, a registered name, or an import path `module:Class`. Any `BaseHarness` subclass works |
| `response_length_per_turn` | int | Hard cap on one generation; becomes the call's `max_tokens` |
| `max_response_length` | int | The response region a conversation must fit. Optional — unset, there is no accounting |
| `max_env_response_per_turn` | int | Ceiling on one observation; over it the text is cut. Default 2048 |
| `compact_budget`, `compact_summary_budget` | int | `compact` only. `compact` needs `compact_budget` or `max_response_length`, or no trigger can fire and it runs as concat -- so that is refused. `compact_summary_budget` alone does not satisfy it |
| `tokens_per_image` | int | What one image costs when sizes are estimated. It feeds the **compaction trigger**, so a value far from your environment's real frame cost makes `compact` misbehave |
| `tokenizer` | str | A HuggingFace id or path. Given one, text sizes are exact instead of 4 characters a token |

**`default_chat_config`** — Top-level fallback: applied to any env that doesn't define its own `chat_config`.

**`experiment`**:
- `dump_dir` — Root directory for rollout outputs
- `default_max_turns` — Fallback max_turns if env doesn't specify one

**`run`**:
- `backend` — Which backend to use. The names with a block in `vagen/configs/eval_default.yaml`: `openai` | `azure` | `sglang` | `vllm` | `together` | `claude` | `gemini`. Any other name needs a `backends.<name>:` block of its own; without one the run stops and lists what is configured.
- `max_concurrent_jobs` — Episode-level parallelism (how many episodes run at once)
- `resume` — `skip_completed` (default) skips episodes already completed **by the same model**; `force_rerun` deletes the previous rollouts and runs everything again; `off` runs everything and keeps what is there. Note YAML reads a bare `off` as the boolean `False`; both are accepted
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
    └── {YYYYmmdd-HHMMSS}-{uuid8}/
        ├── metrics.json                # per-episode results (success, reward, finish_reason)
        ├── messages.json               # full conversation history
        ├── assistant_texts.json        # model replies only
        ├── transcript.txt              # human-readable conversation
        └── images/
            └── turn_01_01.png          # 1-indexed; turn 01 is the reset observation
```

## 2. Scripts

Typical run script:

```bash
#!/bin/bash
# Run FrozenLake eval with sglang backend
cd /path/to/VAGEN
python -m vagen.evaluation \
    --config examples/evaluate/frozenlake/config.yaml
```

Override model or backend on the fly:

```bash
# Switch to OpenAI
python -m vagen.evaluation \
    --config examples/evaluate/frozenlake/config.yaml \
    run.backend=openai \
    backends.openai.model=gpt-4o-mini \
    experiment.dump_dir=./rollouts/gpt4o_mini
```

## 3. Custom Adapters

To add a new backend, implement `EvaluationBackend` and register it:

```python
# my_adapter.py
from vagen.evaluation.backends import EvaluationBackend, register_adapter, register_client

# Step 1: Register client factory
@register_client("my_backend")
def build_my_client(cfg):
    return MyAsyncClient(api_key=cfg.get("api_key"), base_url=cfg.get("base_url"))

# Step 2: Implement and register adapter
@register_adapter("my_backend")
class MyAdapter(EvaluationBackend):

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
