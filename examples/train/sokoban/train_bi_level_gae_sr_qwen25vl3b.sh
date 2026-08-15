#!/bin/bash
# sokoban - bi_level_gae - concat - Qwen/Qwen2.5-VL-3B-Instruct
#
# Per-experiment settings only. Everything that makes a VAGEN run work at all lives in
# vagen/configs/baseline_vllm.flags and is read below -- in particular the two flags that
# select VAGEN's agent loop. Without them verl runs its own, and the job comes up looking
# healthy while none of this repo's rollout code executes -- so the shared flags file is
# the single place those live, and this script holds only what makes it this experiment.
#
# Anything after "${BASE[@]}" overrides it, and anything on the command line overrides
# that, so a one-off sweep needs no edit here.
set -eo pipefail

V=$(cd "$(dirname "$0")/../../.." && pwd)
SCRIPTDIR=$(cd "$(dirname "$0")" && pwd)
PROJECT_NAME=${PROJECT_NAME:-vagen_experiments}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-sokoban_bi_level_gae_sr_qwen25vl3b}
EXPERIMENT_DIR=${EXPERIMENT_DIR:-$V/exps/$PROJECT_NAME/$EXPERIMENT_NAME}
MODEL=${MODEL:-Qwen/Qwen2.5-VL-3B-Instruct}
JUDGE_MODEL=${JUDGE_MODEL:-Qwen/Qwen3-4B-Instruct-2507}
JUDGE_PORT=${JUDGE_PORT:-8123}
JUDGE_TP=${JUDGE_TP:-4}
JUDGE_MEM=${JUDGE_MEM:-0.10}
JUDGE_BASE_URL=${JUDGE_BASE_URL:-http://127.0.0.1:${JUDGE_PORT}/v1}
JUDGE_ENV=${JUDGE_ENV:-$(python3 -c 'import os,sys; print(os.path.dirname(os.path.dirname(sys.executable)))')}
START_JUDGE=${START_JUDGE:-1}
mkdir -p "$EXPERIMENT_DIR"

# verl is not imported as an installed package; it is a checkout, and it has to come
# first on PYTHONPATH so this fork wins over any other copy.
# Both layouts: the submodule at VAGEN/verl that the README creates, and a sibling
# checkout next to VAGEN. Probed for a file rather than the directory -- an uninitialised
# submodule leaves VAGEN/verl there but empty. Left unresolved this used to go on with
# VERL empty, which made hydra.searchpath "file:///verl/trainer/config" and failed later
# on something that does not mention verl.
# ★ A plain loop, not `VERL=${VERL:-$(...)}`. Under `set -e` a command substitution that
# exits non-zero kills the shell AT THE ASSIGNMENT, so the diagnostic below never ran: a
# clone without --recursive got exit 1 and no output at all, which is precisely the case
# the diagnostic exists for.
if [ -z "${VERL:-}" ]; then
    for d in "$V/verl" "$V/../verl"; do
        if [ -f "$d/verl/trainer/config/ppo_trainer.yaml" ]; then
            VERL=$(cd "$d" && pwd)
            break
        fi
    done
fi
if [ -z "$VERL" ]; then
    echo "verl not found at $V/verl or $V/../verl." >&2
    echo "Run: git submodule update --init --recursive   (or set VERL=/path/to/verl)" >&2
    exit 1
fi
export PYTHONPATH=${VERL:+$VERL:}$V${PYTHONPATH:+:$PYTHONPATH}
mapfile -t BASE < <(grep -vE '^\s*(#|$)' "$V/vagen/configs/baseline_vllm.flags" | sed "s|\$V|$V|g")

# State reward is part of the environment config, and its judge is an ordinary external
# service just like an evaluation model server. Start it explicitly, wait until it is
# healthy, and always reap it when training exits. Set START_JUDGE=0 to use an already
# running compatible endpoint named by JUDGE_BASE_URL/JUDGE_MODEL.
export JUDGE_BASE_URL JUDGE_MODEL
JUDGE_HEALTH_URL=${JUDGE_BASE_URL%/}
JUDGE_HEALTH_URL=${JUDGE_HEALTH_URL%/v1}/health
JUDGE_LOG="$EXPERIMENT_DIR/judge.log"
JUDGE_PID=
cleanup_judge() {
    if [ -n "${JUDGE_PID:-}" ]; then
        kill "$JUDGE_PID" 2>/dev/null || true
        wait "$JUDGE_PID" 2>/dev/null || true
    fi
}
trap cleanup_judge EXIT

if [ "$START_JUDGE" = 1 ]; then
    echo "[judge] starting $JUDGE_MODEL on :$JUDGE_PORT, log -> $JUDGE_LOG"
    ENV="$JUDGE_ENV" MODEL="$JUDGE_MODEL" PORT="$JUDGE_PORT" TP="$JUDGE_TP" MEM="$JUDGE_MEM" \
        bash "$V/scripts/launch_judge.sh" >"$JUDGE_LOG" 2>&1 &
    JUDGE_PID=$!
fi

for _ in $(seq 1 90); do
    curl -sf -m 3 "$JUDGE_HEALTH_URL" >/dev/null 2>&1 && break
    if [ -n "${JUDGE_PID:-}" ] && ! kill -0 "$JUDGE_PID" 2>/dev/null; then
        echo "judge died during startup; see $JUDGE_LOG" >&2
        tail -n 80 "$JUDGE_LOG" >&2 || true
        exit 1
    fi
    sleep 10
done
curl -sf -m 3 "$JUDGE_HEALTH_URL" >/dev/null || {
    echo "judge did not become healthy at $JUDGE_HEALTH_URL; see $JUDGE_LOG" >&2
    exit 1
}
echo "[judge] up"

PYTHONUNBUFFERED=1 python3 -m vagen.main_ppo \
    --config-path="$V/vagen/configs" --config-name=vagen_multiturn \
    hydra.searchpath="[file://$VERL/verl/trainer/config]" \
    data.custom_cls.path="$V/vagen/gym_agent_dataset.py" \
    "${BASE[@]}" \
    data.train_files="$SCRIPTDIR/train_sokoban_vision_sr.yaml" \
    data.val_files="$SCRIPTDIR/val_sokoban_vision_sr.yaml" \
    actor_rollout_ref.model.path="$MODEL" \
    critic.model.path="$MODEL" \
    critic.enable=True \
    algorithm.adv_estimator=bi_level_gae \
    +algorithm.high_level_gamma=0.9 \
    trainer.harness=concat \
    data.train_batch_size=128 \
    data.max_response_length=4000 \
    actor_rollout_ref.actor.ppo_mini_batch_size=32 \
    actor_rollout_ref.rollout.n=1 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.max_num_batched_tokens=10000 \
    actor_rollout_ref.rollout.enforce_eager=True \
    actor_rollout_ref.rollout.free_cache_engine=True \
    actor_rollout_ref.rollout.enable_chunked_prefill=True \
    trainer.n_gpus_per_node=4 \
    trainer.nnodes=1 \
    trainer.critic_warmup=0 \
    trainer.save_freq=100 \
    trainer.test_freq=20 \
    trainer.total_training_steps=401 \
    trainer.project_name="$PROJECT_NAME" \
    trainer.experiment_name="$EXPERIMENT_NAME" \
    trainer.default_local_dir="$EXPERIMENT_DIR/verl_checkpoints" \
    trainer.rollout_data_dir="$EXPERIMENT_DIR/rollout_data" \
    trainer.validation_data_dir="$EXPERIMENT_DIR/validation" \
    actor_rollout_ref.actor.checkpoint.save_contents="['model','hf_model','optimizer','extra']" \
    "$@" \
    2>&1 | tee "$EXPERIMENT_DIR/run.log"
