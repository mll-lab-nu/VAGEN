# Running VAGEN sweeps on a Slurm cluster

Three files. Nothing here contains a site's account names, paths or hostnames -- supply
those through the environment.

| file | what it does |
|---|---|
| `bootstrap.sh` | clone code, build the python env, download weights. Submit as a **CPU** job. |
| `train.sbatch` | one training run. |
| `pack.sbatch`  | several 4-GPU runs inside one allocation, split by `CUDA_VISIBLE_DEVICES`. |

## Site configuration

```bash
export CODE_ROOT=/path/on/your/home/fs      # holds VAGEN/ and verl/ as siblings; small
export DATA_ROOT=/path/on/your/shared/fs    # env, HF cache, checkpoints; tens of GB
export FLASH_ATTN_WHEEL=https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.1/flash_attn-2.8.1+cu12torch2.8cxx11abiFALSE-cp312-cp312-linux_x86_64.whl
```

`CODE_ROOT` and `DATA_ROOT` are separate on purpose: a home filesystem is usually a few
hundred GB and shared with everything else, while the environment alone is ~25 GB and
checkpoints are far larger.

## 1. Bootstrap (once)

```bash
sbatch --account=$SLURM_ACCOUNT --qos=$CPU_QOS --nodes=1 \
       --cpus-per-task=16 --mem=64G --time=05:00:00 \
       examples/slurm/bootstrap.sh
```

Idempotent -- rerun it after a partial failure and it resumes.

## 2. One run

```bash
sbatch --account=$SLURM_ACCOUNT --qos=$GPU_QOS --nodes=1 \
       --gpus-per-node=$GPU_TYPE:4 --cpus-per-task=48 --mem=400G \
       --time=48:00:00 --requeue --job-name=sokoban_token \
       --export=ALL,EXP_NAME=sokoban_token,ENV_NAME=sokoban,ADV_EST=token_level_gae,LAM=1.0 \
       examples/slurm/train.sbatch
```

With `STATE_REWARD=on`, ask for **one more GPU** than the trainer uses: the run starts its
own judge on the last one. Separate Slurm jobs land on separate nodes and cannot share a
judge over localhost, so each pays for its own.

## 3. A packed sweep

A QOS usually caps jobs-per-user well below GPUs-per-user, so packing is what lets a sweep
use the quota it has.

```bash
RUNS='mode_concat|sokoban|token_level_gae|1.0||||
      mode_no_concat|sokoban|token_level_gae|1.0|||no_concat|
      mode_compact|sokoban|token_level_gae|1.0|||compact|trainer.compact_budget=2000' \
sbatch --account=$SLURM_ACCOUNT --qos=$GPU_QOS --nodes=1 \
       --gpus-per-node=$GPU_TYPE:8 --cpus-per-task=96 --mem=800G \
       --mem=0 --time=48:00:00 --requeue examples/slurm/pack.sbatch
```

Fields: `name|env|adv_estimator|lam|loss_mode|lam_low|harness|extra_args`.

Two constraints that are not obvious and do not fail cleanly:

* **`--mem=0`**, not a number. One 3B run peaks near 790 GB of host RAM, so two do not fit
  in the 800 GB that looks generous; the OOM killer takes a Ray worker and the run dies
  with `ActorDiedError` minutes in, while Slurm still reports the job COMPLETED.
* **Pack runs of similar speed.** `no_concat` emits one training row per turn where
  `concat` emits one per episode, so it is about 2.4x slower per step (214 s against 89 s,
  measured on 5-turn Sokoban). Pairing a slow policy with a fast one leaves half the node
  idle for hours.

## The algorithm grid

| variant | `ADV_EST` | `LOSS_MODE` | `LAM` | `LAM_LOW` |
|---|---|---|---|---|
| **baseline** | `episode_gae` | *(vanilla)* | 1.0 | |
| token-level | `token_level_gae` | *(vanilla)* | 1.0 | |
| turn-level  | `turn_level_gae`  | `turn_gspo` | 0.95 | |
| bi-level    | `bi_level_gae`    | `turn_gspo` | 0.95 | 1.0 |

`gamma` is 1.0 throughout and the trainer refuses anything else for `bi_level_gae`: a
per-token clock and a per-turn clock disagree by `gamma ** turn_length`, which is a factor
set by how much the model wrote rather than by anything in the config.

`episode_gae` is the control the other three are read against: the same recursion, the
same critic, the same `lam`, with the episode's whole reward lumped onto its last token
instead of left where it was earned. That is what single-turn RLHF does, so the gap
between it and `token_level_gae` is the value of per-token placement, and the gap to
`bi_level_gae` is the value of the turn structure on top. Run it in the same layout as
whatever it is being compared with -- it stitches rows like the others, so `no_concat`
and `compact` are both fair.

`lam_low=1.0` is not a tuning choice. The turn-level signal reaches a token `d` positions
before the turn's end with weight `lam_low ** d`, so `0.95` delivers it to the last ~20
tokens of an action and to none of the others.

**State reward exists for Sokoban only** (`STATE_REWARD_SPECS`), so that axis of the grid
has three cells rather than six.

## Things that cost a day to learn

* **`max_model_len`.** Without it vLLM sizes the KV cache against the model's full context
  (128k here) and refuses to start. The message reports a memory figure, so it reads as an
  out-of-memory problem; raising or lowering `gpu_memory_utilization` cannot fix it, and
  each direction fails differently. Both the trainer's rollout and the judge need the cap.
* **`HF_HUB_OFFLINE=1`.** Many workers revalidating one shared-filesystem snapshot at once
  intermittently resolve an empty file list, which surfaces as `IndexError` deep inside a
  processor loader.
* **`RAY_TMPDIR` must be node-local and short.** Ray puts a unix socket under it and
  AF_UNIX caps the whole path at 107 bytes.
* **`TORCH_CUDA_ARCH_LIST`.** Unset, torch compiles the CUDA extensions for every
  architecture -- 30+ minutes before the first step, paid per job.
* **A job's liveness is `run.log`'s mtime, not its Slurm state.** `pack.sbatch` keeps
  running while the training processes under it are dead or wedged, so `squeue` reports
  `RUNNING` indefinitely and tailing the wrapper's stdout shows fresh lines from the
  wrapper itself. Check `stat -c %y $OUT/run.log` against the wall clock; a step is
  ~90-215 s depending on the context policy, so anything over ~15 minutes stale is hung.
* **Prefer one Slurm job per run over `pack.sbatch`.** Packing exists because the QOS
  caps *submitted* jobs (see below), not because it is better. Measured over one night:
  all three runs packed into a single job failed -- one never got past vLLM startup, one
  died at step 15 in verl's bucketed CUDA-IPC weight transfer with `TypeError: 'str'
  object is not callable` (a handle arriving off the wrong ZMQ message), and the third
  only became healthy once resubmitted standalone -- while nine separately submitted jobs
  ran seven hours without incident. The IPC socket path is
  `/tmp/rl-colocate-zmq-<ray_job_id>-replica-N-rank-M.sock`, and a Ray job id is a
  per-cluster counter rather than a global one, so two Ray clusters inside one Slurm
  allocation can agree on it; separate Slurm jobs get separate `/tmp` namespaces and
  cannot.
* **If you do pack, never pack `no_concat`.** It emits one row per turn against `concat`'s
  one per episode and runs ~2.4x slower per step, so a co-scheduled partner finishes hours
  early and its half of the node sits idle for the rest.
* **The QOS caps submitted jobs, pending included** (10 here), so an 11th run cannot be
  parked with `--dependency` -- `sbatch` refuses it outright. Queue the overflow in a file
  and drain it from a *CPU* job, which is charged against a different QOS and so does not
  consume one of the ten.
