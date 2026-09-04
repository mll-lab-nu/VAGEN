#!/usr/bin/env bash
# sokoban - bi_level_gae - concat - Qwen2.5-VL-3B + state reward
set -eo pipefail

V=$(cd "$(dirname "$0")/../../.." && pwd)
SCRIPTDIR=$(cd "$(dirname "$0")" && pwd)
PROJECT_NAME=${PROJECT_NAME:-vagen-experiments}
BI_LEVEL_MIX=${BI_LEVEL_MIX:-0.75}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-sokoban_bi_level_gae_state_reward_qwen25vl3b}
EXPERIMENT_DIR=${EXPERIMENT_DIR:-$V/exps/$PROJECT_NAME/$EXPERIMENT_NAME}
MODEL=${MODEL:-Qwen/Qwen2.5-VL-3B-Instruct}
JUDGE_MODEL=${JUDGE_MODEL:-Qwen/Qwen3-4B-Instruct-2507}
JUDGE_PORT=${JUDGE_PORT:-8123}
JUDGE_TP=${JUDGE_TP:-4}
JUDGE_MEM=${JUDGE_MEM:-0.10}
JUDGE_BACKEND=${JUDGE_BACKEND:-sglang}
JUDGE_SEED=${JUDGE_SEED:-42}
JUDGE_ATTENTION_BACKEND=${JUDGE_ATTENTION_BACKEND:-flashinfer}
JUDGE_BASE_URL=${JUDGE_BASE_URL:-http://127.0.0.1:${JUDGE_PORT}/v1}
JUDGE_ENV=${JUDGE_ENV:-$(python3 -c 'import os,sys; print(os.path.dirname(os.path.dirname(sys.executable)))')}
START_JUDGE=${START_JUDGE:-1}
GAMMA_TURN=${GAMMA_TURN:-0.95}
LAMBDA_TURN=${LAMBDA_TURN:-0.95}
LAMBDA_TOKEN=${LAMBDA_TOKEN:-1.0}
ROLLOUT_BACKEND=${ROLLOUT_BACKEND:-sglang}
RUN_SEED=${RUN_SEED:-42}
N_GPUS_PER_NODE=${N_GPUS_PER_NODE:-4}
STATE_REWARD_CREDIT_SITE=${STATE_REWARD_CREDIT_SITE:-turn_end}
STATE_REWARD_SCORE_BASE=${STATE_REWARD_SCORE_BASE:-0.625}
STATE_REWARD_AGGREGATION=${STATE_REWARD_AGGREGATION:-episode_mean}
STATE_REWARD_SCORER=${STATE_REWARD_SCORER:-exact}
STATE_ESTIMATION_REWARD=${STATE_ESTIMATION_REWARD:-0.006}
TRANSITION_PREDICTION_REWARD=${TRANSITION_PREDICTION_REWARD:-0.006}
if [ "$STATE_REWARD_SCORER" = exact ]; then
    START_JUDGE=0
fi
mkdir -p "$EXPERIMENT_DIR"

if [ -z "${VERL:-}" ]; then
    for d in "$V/verl" "$V/../verl"; do
        if [ -f "$d/verl/trainer/config/ppo_trainer.yaml" ]; then
            VERL=$(cd "$d" && pwd)
            break
        fi
    done
fi
if [ -z "$VERL" ]; then
    echo "verl not found at $V/verl or $V/../verl" >&2
    exit 1
fi
export PYTHONPATH=${VERL:+$VERL:}$V${PYTHONPATH:+:$PYTHONPATH}
mapfile -t BASE < <(grep -vE '^\s*(#|$)' "$V/vagen/configs/training_defaults.flags" | sed "s|\$V|$V|g")

export JUDGE_BASE_URL JUDGE_MODEL STATE_REWARD_CREDIT_SITE STATE_REWARD_SCORE_BASE
export STATE_REWARD_AGGREGATION
export STATE_REWARD_SCORER
export STATE_ESTIMATION_REWARD TRANSITION_PREDICTION_REWARD
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

if [ "$STATE_REWARD_SCORER" != exact ]; then
    if [ "$START_JUDGE" = 1 ]; then
        ENV="$JUDGE_ENV" MODEL="$JUDGE_MODEL" PORT="$JUDGE_PORT" TP="$JUDGE_TP" MEM="$JUDGE_MEM" \
            BACKEND="$JUDGE_BACKEND" SEED="$JUDGE_SEED" ATTENTION_BACKEND="$JUDGE_ATTENTION_BACKEND" \
            bash "$V/scripts/launch_judge.sh" >"$JUDGE_LOG" 2>&1 &
        JUDGE_PID=$!
    fi
    for _ in $(seq 1 90); do
        curl -sf -m 3 "$JUDGE_HEALTH_URL" >/dev/null 2>&1 && break
        if [ -n "${JUDGE_PID:-}" ] && ! kill -0 "$JUDGE_PID" 2>/dev/null; then
            tail -n 80 "$JUDGE_LOG" >&2 || true
            exit 1
        fi
        sleep 10
    done
    curl -sf -m 3 "$JUDGE_HEALTH_URL" >/dev/null || { tail -n 80 "$JUDGE_LOG" >&2; exit 1; }
else
    echo "[judge] skipped for exact state-reward scorer"
fi

PYTHONUNBUFFERED=1 python3 -m vagen.training.main \
    --config-path="$V/vagen/configs" --config-name=vagen_multiturn \
    hydra.searchpath="[file://$VERL/verl/trainer/config]" \
    data.custom_cls.path="$V/vagen/training/dataset.py" \
    "${BASE[@]}" \
    data.train_files="$SCRIPTDIR/train_sokoban_vision_sr.yaml" \
    data.val_files="$SCRIPTDIR/val_sokoban_vision_sr.yaml" \
    actor_rollout_ref.model.path="$MODEL" \
    critic.model.path="$MODEL" \
    critic.enable=True \
    algorithm.adv_estimator=bi_level_gae \
    algorithm.gamma=1.0 \
    algorithm.lam=1.0 \
    +algorithm.gamma_turn="$GAMMA_TURN" \
    +algorithm.lambda_turn="$LAMBDA_TURN" \
    +algorithm.lambda_token="$LAMBDA_TOKEN" \
    +algorithm.bi_level_mix="$BI_LEVEL_MIX" \
    trainer.harness=concat \
    data.seed="$RUN_SEED" \
    data.train_batch_size=128 \
    data.max_response_length=4000 \
    actor_rollout_ref.actor.ppo_mini_batch_size=32 \
    actor_rollout_ref.rollout.n=1 \
    actor_rollout_ref.rollout.name="$ROLLOUT_BACKEND" \
    actor_rollout_ref.rollout.seed="$RUN_SEED" \
    actor_rollout_ref.rollout.full_determinism=True \
    actor_rollout_ref.rollout.skip_tokenizer_init=False \
    +actor_rollout_ref.rollout.engine_kwargs.sglang.attention_backend=flashinfer \
    +actor_rollout_ref.rollout.engine_kwargs.sglang.random_seed="$RUN_SEED" \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.max_num_batched_tokens=10000 \
    actor_rollout_ref.rollout.enforce_eager=True \
    actor_rollout_ref.rollout.free_cache_engine=False \
    actor_rollout_ref.rollout.enable_chunked_prefill=True \
    actor_rollout_ref.actor.data_loader_seed="$RUN_SEED" \
    actor_rollout_ref.actor.fsdp_config.seed="$RUN_SEED" \
    actor_rollout_ref.actor.fsdp_config.full_determinism=True \
    actor_rollout_ref.ref.fsdp_config.seed="$RUN_SEED" \
    actor_rollout_ref.ref.fsdp_config.full_determinism=True \
    critic.data_loader_seed="$RUN_SEED" \
    critic.fsdp.seed="$RUN_SEED" \
    critic.fsdp.full_determinism=True \
    trainer.n_gpus_per_node="$N_GPUS_PER_NODE" \
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
    "$@" 2>&1 | tee "$EXPERIMENT_DIR/run.log"
