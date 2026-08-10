#!/bin/bash
# One VAGEN training run on a Slurm cluster. Every experiment in a sweep shares this file
# and differs only in its environment, so the variants cannot drift apart -- which is how
# this repo's twenty standalone launch scripts once all lost the two flags that select
# VAGEN's own agent loop.
#
#   sbatch --account=$SLURM_ACCOUNT --qos=$GPU_QOS --nodes=1 \
#          --gpus-per-node=$GPU_TYPE:4 --cpus-per-task=48 --mem=400G \
#          --time=48:00:00 --requeue --job-name=my_run \
#          --export=ALL,EXP_NAME=my_run,ENV_NAME=sokoban,ADV_EST=token_level_gae,LAM=1.0 \
#          examples/slurm/train.sbatch
#
# Site configuration (no defaults -- set them for your cluster):
#   CODE_ROOT   parent of VAGEN/ and verl/; small, keep it on your home filesystem
#   DATA_ROOT   env, HF cache, checkpoints; large, keep it on the shared/scratch filesystem
# Per-run:
#   EXP_NAME ENV_NAME ADV_EST LAM      required
#   LOSS_MODE LAM_LOW STATE_REWARD     optional (STATE_REWARD=on adds a judge; ask for one
#                                      more GPU than the trainer uses)
#   HARNESS                            concat | no_concat | compact   (default concat)
#   EXTRA_ARGS                         free-form hydra overrides, e.g. trainer.compact_budget=2000
set -euo pipefail

: "${CODE_ROOT:?set CODE_ROOT to the directory holding VAGEN/ and verl/}"
: "${DATA_ROOT:?set DATA_ROOT to a large shared filesystem for env, cache and checkpoints}"
V=$CODE_ROOT/VAGEN
VERL=$CODE_ROOT/verl
PY=$DATA_ROOT/conda/verl/bin/python

export HF_HOME=$DATA_ROOT/hf_home
# The cache is complete after bootstrap.sh, so every network round-trip is pure risk.
# Nine concurrent jobs, each with several vLLM workers, revalidating the same shared-FS
# snapshot produced an empty file list often enough to kill most of a sweep -- surfacing
# as IndexError on resolved_image_processor_files[0], which names nothing useful.
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTHONPATH=$VERL:$V
export PYTHONUNBUFFERED=1
# Compile for the one architecture actually present. Unset, torch builds the CUDA
# extensions for every architecture it knows -- observed as 30+ minutes before the first
# training step, with every job paying it independently. 8.0 = A100; use 9.0 for H100.
export TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST:-8.0}
# Node-local and short. Ray puts a unix socket under RAY_TMPDIR and AF_UNIX caps the whole
# path at 107 bytes, which a shared-filesystem root can exhaust on its own; and a per-job
# cache on a network filesystem is neither shared nor fast.
export RAY_TMPDIR=/tmp/ray-$SLURM_JOB_ID
export TRITON_CACHE_DIR=/tmp/triton-$SLURM_JOB_ID
export XDG_CACHE_HOME=$DATA_ROOT/tmp/xdg-$SLURM_JOB_ID
mkdir -p "$RAY_TMPDIR" "$TRITON_CACHE_DIR" "$XDG_CACHE_HOME"

OUT=$DATA_ROOT/outputs/$EXP_NAME
mkdir -p "$OUT"

# max_turns is normalised to 5 for every environment. The repo ships sokoban at 20 and
# frozenlake at 5; 5 is the published setting and the only one under which the two are
# comparable. Written to a copy so the checkout stays clean and the value actually used
# is visible next to the run's own logs.
MAX_TURNS=${MAX_TURNS:-5}
SRC=$V/examples/train/$ENV_NAME
for split in train val; do
  f=$SRC/${split}_${ENV_NAME}_vision.yaml
  [ -f "$f" ] && sed "s/^\( *max_turns:\) *[0-9]\+/\1 $MAX_TURNS/" "$f" > "$OUT/$(basename "$f")"
done
echo "[cfg] $(grep -m1 max_turns "$OUT/train_${ENV_NAME}_vision.yaml")"

mapfile -t BASE < <(grep -vE '^\s*(#|$)' "$V/vagen/configs/baseline_vllm.flags" | sed "s|\$V|$V|g")

EXTRA=()
if [ -n "${LOSS_MODE:-}" ]; then
  EXTRA+=("actor_rollout_ref.actor.policy_loss.loss_mode=$LOSS_MODE")
  # turn_gspo and turn_ppo run in the actor worker, a separate process that builds its own
  # registry -- registering them on the driver registers them nowhere that matters.
  EXTRA+=("actor_rollout_ref.model.external_lib=vagen.custom_loss")
fi
[ -n "${LAM_LOW:-}" ] && EXTRA+=("+algorithm.lam_low=$LAM_LOW")
[ -n "${EXTRA_ARGS:-}" ] && EXTRA+=($EXTRA_ARGS)

if [ "${STATE_REWARD:-off}" = "on" ]; then
  # One judge per job, inside this job's own allocation: separate Slurm jobs land on
  # separate nodes and cannot share a localhost port. TP=1 on the last GPU, which is why
  # these jobs ask for one more GPU than the trainer uses.
  #
  # --max-model-len matters: without it vLLM sizes the KV cache against the model's full
  # context (262144 for this judge) and refuses to start, reporting a memory figure that
  # looks like an out-of-memory problem rather than a configuration one.
  JUDGE_GPU=$((SLURM_GPUS_ON_NODE - 1))
  CUDA_VISIBLE_DEVICES=$JUDGE_GPU "$PY" -m vllm.entrypoints.openai.api_server \
      --model Qwen/Qwen3-4B-Instruct-2507 --port 8123 \
      --tensor-parallel-size 1 --gpu-memory-utilization 0.30 --max-model-len 4096 \
      > "$OUT/judge.log" 2>&1 &
  trap 'kill %1 2>/dev/null || true' EXIT
  for i in $(seq 1 90); do
    curl -sf http://127.0.0.1:8123/v1/models >/dev/null 2>&1 && break; sleep 10
  done
  curl -sf http://127.0.0.1:8123/v1/models >/dev/null || { echo "[judge] never came up"; exit 1; }
  echo "[judge] ready"
  EXTRA+=("trainer.state_reward.state_estimation.enable=True"
          "trainer.state_reward.transition_prediction.enable=True")
  export CUDA_VISIBLE_DEVICES=$(seq -s, 0 $((SLURM_GPUS_ON_NODE - 2)))
fi

echo "[run] $EXP_NAME | $ENV_NAME | $ADV_EST | lam=$LAM | loss=${LOSS_MODE:-vanilla} | harness=${HARNESS:-concat}"
"$PY" -m vagen.main_ppo \
  --config-path="$V/vagen/configs" --config-name=vagen_multiturn \
  hydra.searchpath="[file://$VERL/verl/trainer/config]" \
  data.custom_cls.path="$V/vagen/gym_agent_dataset.py" \
  "${BASE[@]}" \
  data.train_files="$OUT/train_${ENV_NAME}_vision.yaml" \
  data.val_files="$OUT/val_${ENV_NAME}_vision.yaml" \
  actor_rollout_ref.model.path=Qwen/Qwen2.5-VL-3B-Instruct \
  critic.model.path=Qwen/Qwen2.5-VL-3B-Instruct \
  critic.enable=True \
  algorithm.adv_estimator="$ADV_EST" \
  algorithm.gamma=1.0 \
  algorithm.lam="$LAM" \
  trainer.harness=${HARNESS:-concat} \
  data.train_batch_size=128 \
  data.max_response_length=4000 \
  actor_rollout_ref.actor.ppo_mini_batch_size=32 \
  actor_rollout_ref.rollout.n=1 \
  actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.40 \
  actor_rollout_ref.rollout.max_num_batched_tokens=10000 \
  actor_rollout_ref.rollout.max_model_len=8192 \
  actor_rollout_ref.rollout.enforce_eager=True \
  actor_rollout_ref.rollout.free_cache_engine=True \
  actor_rollout_ref.rollout.enable_chunked_prefill=True \
  trainer.n_gpus_per_node=4 \
  trainer.nnodes=1 \
  trainer.critic_warmup=0 \
  trainer.save_freq=100 \
  trainer.test_freq=20 \
  trainer.total_training_steps=${TOTAL_STEPS:-401} \
  trainer.project_name=${PROJECT_NAME:-vagen_sweep} \
  trainer.experiment_name="$EXP_NAME" \
  trainer.default_local_dir="$OUT/checkpoints" \
  trainer.rollout_data_dir="$OUT/rollout_data" \
  trainer.validation_data_dir="$OUT/validation" \
  "${EXTRA[@]}" \
  2>&1 | tee "$OUT/run.log"
