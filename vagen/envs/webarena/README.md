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
    --seed=0 --max_steps=3 --auth_cache_dir=./.wa_auth
```

Runs `reset` → `do(action="Wait")` → `exit(...)` on `task[seed % 165]`.
First run generates cookies via `auto_login.py`; later runs reuse the cache.

### Parallel sessions

```bash
PYTHONPATH=. python -m vagen.envs.webarena.tests.test_handler_parallel \
    --n_browsers=2 --max_contexts_per_browser=2 \
    --n_sessions=4 --auth_cache_dir=./.wa_auth
```

Watch `per_browser=[a, b]` in the output — should stay balanced.

## Serving (for trainer / agent_loop)

```bash
PYTHONPATH=. python -m vagen.envs.webarena.serve \
    --task_config_file=vagen/envs/webarena/config_files/normalized_test.json \
    --n_browsers=4 --max_contexts_per_browser=16 \
    --port=8002 --auth_cache_dir=./.wa_auth
```

Clients use `vagen.envs_remote.GymImageEnvClient` against `http://localhost:8002`.

## Benchmark

Stress-test the running server with N concurrent clients running M steps
over R rounds. Each client: connect → reset → step × (M-1) → `exit(...)` → close.

Prereqs: tunnel up, auth cache pre-populated, server running on `--port`.

```bash
PYTHONPATH=. python -m vagen.envs.webarena.benchmark \
    --base_url=http://localhost:8002 \
    --num_rounds=2 --num_clients=16 --num_steps=3
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
