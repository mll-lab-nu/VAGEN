# VAGEN WebArena Env

One session = one Playwright browser context = one WebArena task.
Concurrency comes from a handler-owned pool of `K` Chromium browsers,
each hosting up to `M` contexts. Total capacity = `K × M` sessions.


## Setup

### 1. SSH tunnel

WebArena Docker services run on a remote host. Forward their ports to localhost:

```bash
ssh -i ~/.ssh/id_ed25519 -p <REMOTE_PORT> -o StrictHostKeyChecking=accept-new \
    root@<REMOTE_HOST> \
    -L 7770:localhost:7770 \
    -L 7780:localhost:7780 \
    -L 9999:localhost:9999 \
    -L 8023:localhost:8023 \
    -L 8888:localhost:8888 \
    -L 4399:localhost:4399 \
    -N &
```

| Port | Service |
|------|---------|
| 7770 | Shopping |
| 7780 | Shopping Admin |
| 9999 | Reddit |
| 8023 | GitLab |
| 8888 | Wikipedia |
| 4399 | Homepage |

Check:

```bash
for p in 7770 7780 9999 8023 8888 4399; do
  timeout 2 bash -c "exec 3<>/dev/tcp/localhost/$p && echo $p OPEN" \
    2>/dev/null || echo "$p closed"
done
```

### 2. Conda env

Create or activate an env with Playwright installed:

```bash
conda activate webarena   # or whatever your env is called
```

First time, install extra deps:

```bash
pip install fastapi uvicorn matplotlib lxml beautifulsoup4 \
            scikit-image dashscope anthropic
pip install -U "openai>=1.0"
playwright install chromium   # if not already installed
```

### 3. Source env vars

Every shell (from the repo root):

```bash
source vagen/envs/webarena/setup_vars.sh
```

Without this, `env_config.py` raises `RuntimeError: WebArena URL env vars not set`.

## Tests

All tests run from the repo root.

```bash
conda activate webarena
source vagen/envs/webarena/setup_vars.sh
```

### Single session, real task

```bash
PYTHONPATH=. python -m vagen.envs.webarena.tests.test_env_local \
    --seed=0 --max_steps=3 --auth_cache_dir=./vagen/envs/webarena/.wa_auth
```

Runs `reset` → `do(action="Wait")` → `exit(...)` on `task[seed % 165]`.
First run generates cookies via `auto_login.py`; later runs reuse the cache.

### Parallel sessions

```bash
PYTHONPATH=. python -m vagen.envs.webarena.tests.test_handler_parallel \
    --n_browsers=2 --max_contexts_per_browser=2 \
    --n_sessions=4 --auth_cache_dir=./vagen/envs/webarena/.wa_auth
```

Watch `per_browser=[a, b]` in the output — should stay balanced.

## Serving (for trainer / agent_loop)

Two modes:

### (a) Single server, in-process pool

```bash
PYTHONPATH=. python -m vagen.envs.webarena.serve \
    --task_config_file=vagen/envs/webarena/config_files/normalized_test.json \
    --n_browsers=4 --max_contexts_per_browser=16 \
    --port=8002 --auth_cache_dir=./vagen/envs/webarena/.wa_auth
```

Clients use `vagen.envs_remote.GymImageEnvClient` against `http://localhost:8002`.

Caveat: Playwright sync calls can hang inside a browser's thread, and
Python can't kill a thread — one hang takes down all `M` sessions on
that browser and eventually deadlocks the pool. Use (b) for long-running
training jobs.

### (b) Supervisor + worker fleet (hang-resilient, recommended)

Each worker is a separate process running `serve.py` with
`n_browsers=1, max_contexts_per_browser=1`. A parent supervisor health-
checks each worker and `SIGKILL+respawn`s any that hang or die.

```bash
PYTHONPATH=. python -m vagen.envs.webarena.supervisor \
    --n_workers=8 --start_port=8002 \
    --task_config_file=vagen/envs/webarena/config_files/normalized_test.json \
    --auth_cache_dir=./vagen/envs/webarena/.wa_auth
```

Workers listen on `start_port, start_port+1, ..., start_port+n_workers-1`.
Clients pass the full list to `GymImageEnvClient`:

```python
env = GymImageEnvClient({
    "base_urls": [
        "http://localhost:8002",
        "http://localhost:8003",
        # ...
    ],
})
```

Supervisor flags:
- `--health_interval=30` seconds between health sweeps
- `--health_timeout=10` per-request `/health` timeout
- `--max_consecutive_health_failures=3` strikes before `SIGKILL+restart`
- `--startup_grace=30` skip health checks for this long after each (re)start
- `--startup_stagger=1.5` seconds between worker launches (avoids burst
  fork of N Chromiums on dense nodes)
- `--log_dir=./log_files/webarena_workers` per-worker stdout/stderr land here

Workers are launched with `OPENBLAS_NUM_THREADS=OMP_NUM_THREADS=MKL_NUM_THREADS=1`
in their environment. Without these caps, numpy/scipy/OpenBLAS auto-spawn
~`min(64, cpu_count())` threads *per worker process* at import time — on a
128-core node with 8 workers, that's 500+ threads burned before serving any
request, and `pthread_create EAGAIN` under burst load. See
`vagen/envs/webarena/supervisor.py:Worker.start()`.

Trade-off vs (a): per-worker auth cache regen runs on first start, so the
fleet takes ~`n_workers × 30s` longer to be fully ready (or just pre-warm
the shared `auth_cache_dir`).

## Evaluation

Reproduces the WebAgent-R1 BC paper's eval pipeline against the
`weizhepei/Qwen2.5-3B-WebArena-Lite-SFT-epoch-5` SFT checkpoint (or any
other OpenAI-compatible chat endpoint). Three processes: the env
supervisor, a model server (sglang/vllm/...), and the eval driver.

```bash
# Terminal 1: env supervisor (login node, behind the SSH tunnel)
source vagen/envs/webarena/setup_vars.sh
conda activate webarena
PYTHONPATH=. python -m vagen.envs.webarena.supervisor \
    --n_workers=8 --start_port=8002 \
    --task_config_file=vagen/envs/webarena/config_files/normalized_test_nomap.json \
    --auth_cache_dir=./vagen/envs/webarena/.wa_auth

# Terminal 2: model server on a GPU node (sglang here; vllm works too)
conda activate vagen
srun --jobid="$SLURM_JOBID" --overlap python -m sglang.launch_server \
    --host 0.0.0.0 --port 30000 \
    --model-path weizhepei/Qwen2.5-3B-WebArena-Lite-SFT-epoch-5 \
    --tp 1 --trust-remote-code --mem-fraction-static 0.80

# Terminal 3: eval driver (login node)
export OPENAI_API_KEY=sk-...    # used by the fuzzy_match LLM-as-judge in
                                # evaluation_harness; needs to be a valid key
conda activate vagen
PYTHONPATH=. python -m vagen.evaluate.run_eval \
    --config examples/evaluate/webarena/config.yaml \
    backends.sglang.base_url="http://<gpu_node>:30000/v1" \
    backends.sglang.model="weizhepei/Qwen2.5-3B-WebArena-Lite-SFT-epoch-5"
```

Config: [`examples/evaluate/webarena/config.yaml`](../../../examples/evaluate/webarena/config.yaml).

The config sets `data_source: webarena`, which activates the per-env
message-history compressor in `vagen/evaluate/vision_workflow.py` —
historical user turns' HTML observations are replaced by a placeholder
to mirror WebAgent-R1's `WebRLChatPromptConstructor` and avoid 32K
context overflow on multi-turn rollouts.

To override the seed list (e.g. full 165-task test):

```bash
... 'envs.0.seed_list=[0,1,2,...,164]' envs.0.n_envs=165
```

Rollouts (messages, screenshots, metrics) dump to
`$HOME/projects/vagen/rollouts/eval_webarena_smoke/`.

## Benchmark

Stress-test the running server with N concurrent clients running M steps
over R rounds. Each client: connect → reset → step × (M-1) → `exit(...)` → close.

Prereqs: tunnel up, auth cache pre-populated, server running on `--port`.

```bash
# Single server
PYTHONPATH=. python -m vagen.envs.webarena.benchmark \
    --base_urls=http://localhost:8002 \
    --num_rounds=2 --num_clients=16 --num_steps=3

# Supervisor fleet (comma-separated URLs)
PYTHONPATH=. python -m vagen.envs.webarena.benchmark \
    --base_urls=http://localhost:8002,http://localhost:8003,http://localhost:8004 \
    --num_rounds=2 --num_clients=24 --num_steps=3
```

Flags:
- `--num_clients` concurrent sessions per round
- `--num_steps` steps per client (last one is forced `exit` → triggers evaluator)
- `--num_rounds` how many rounds to repeat
- `--max_steps` env's internal cap (default 10)
- `--viewport_width / --viewport_height` browser viewport

### Interpreting latency

Under a pool of `K` browsers with `M` sessions each, each browser's Python
thread serializes `M` contexts' Playwright calls. A single step's latency is
roughly:

```
step_latency ≈ queue_position × (sleep_after_execution + playwright_op + obs_parse)
             ≈ (M/2)          × (3s                    + ~0.5s          + ~0.5s)
```

So with default `sleep_after_execution=3s` and M=4, expect ~8s median step
latency. Lower the sleep (see `WebArenaEnvConfig.sleep_after_execution`) for
actions that don't need page-settling time.

## Capacity tuning

| n_browsers | max_contexts | total sessions | base RAM | notes |
|-----------:|-------------:|---------------:|---------:|-------|
| 2          | 8            | 16             | ~600MB   | debug |
| 4          | 16           | 64             | ~1.2GB   | recommended starter |
| 8          | 16           | 128            | ~2.4GB   | high throughput |
| 16         | 4            | 64             | ~4.8GB   | CPU-heavy workloads |

- `n_browsers` = Python-level parallelism (each browser pins one thread).
- `max_contexts_per_browser` = memory amortization per browser.
- Network-heavy tasks benefit from more contexts per browser; CPU-heavy
  tasks benefit from more browsers.

## Troubleshooting

| Error | Cause / fix |
|-------|-------------|
| `WebArena URL env vars not set` | Forgot to `source setup_vars.sh` |
| `port XXXX: closed` / `ERR_CONNECTION_REFUSED` | SSH tunnel down — restart it |
| `auto_login failed` | Tunnel / Docker issue — delete `auth_cache_dir` and retry |
| `evaluator failed: No module named 'X'` | Optional LLM provider missing; only matters for `fuzzy_match` tasks |
| `greenlet.error` / `Playwright Sync API inside asyncio loop` | A Playwright call bypassed `BrowserSlot.run()` |

## Kill tunnel

```bash
pkill -f '<REMOTE_HOST>'
```

## Known issues

- **BrowserPool slot leak under error paths.** If `env.reset()` raises after
  `acquire_slot()` but the cleanup path doesn't reach `release_slot()`, the
  slot stays held forever. Workers eventually fail `acquire_slot` with a 300s
  timeout. Long-running fleets gradually lose capacity. Workaround: restart
  the offending worker (supervisor will respawn it cleanly). Real fix: audit
  all exception paths in `handler.connect()` and `webarena_env.py:close()`.

- **`sleep_after_execution` is a single knob for two phases.** Observation
  diffing (5 seeds, sleep 3.0 vs 0.5 / 1.5) shows page LOAD (`reset`'s
  `goto`) needs ≥3s on shopping pages to capture lazy-loaded breadcrumb /
  product images, but post-load actions (scroll/wait/click) settle in
  ≤0.5s. Splitting into `sleep_after_navigation=3.0` and
  `sleep_after_action=0.5` would give ~2.8× step speedup without obs
  quality loss.

- **`auto_login.py` Magento (shopping_admin) selector is stale.** Produces
  a 1-cookie / 0-origin storage_state that doesn't authenticate. The
  Magento admin login button selectors no longer match the rendered DOM.
  Other 3 sites (gitlab/shopping/reddit) re-login fine.

- **Cluster cgroup `pids.max=1000` caps single-node fleet to ~8 workers.**
  Each worker (Python + Chromium + renderers) consumes ~70-90 tasks; at 16+
  workers, supervisor's `fork_exec` returns EAGAIN. For 32+ workers, deploy
  across multiple SLURM nodes — `GymImageEnvClient` already supports the
  multi-URL routing.
