#!/bin/bash
# sokoban - default_gae - concat - Qwen/Qwen3.5-4B
#
# Qwen3.5 is natively multimodal: there is no `-VL` variant to look for. The
# `Qwen3.5-VL-*` repos on the Hub are third-party derivatives.
#
# Needs a newer stack than the rest of these scripts, and says so only as
# `KeyError: 'qwen3_5'`:
#   transformers >= 5.2.0   4.57 has qwen3_vl but not this
#   vllm         >= ?       the text stack interleaves linear_attention with
#                           full_attention, so the engine has to know the architecture
# Untested here for that reason.
set -eo pipefail

V=$(cd "$(dirname "$0")/../../.." && pwd)
SCRIPTDIR=$(cd "$(dirname "$0")" && pwd)
PROJECT_NAME=${PROJECT_NAME:-vagen_experiments}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-sokoban_default_gae_qwen35_4b}
EXPERIMENT_DIR=${EXPERIMENT_DIR:-$V/exps/$PROJECT_NAME/$EXPERIMENT_NAME}
MODEL=${MODEL:-Qwen/Qwen3.5-4B}
mkdir -p "$EXPERIMENT_DIR"

# verl is not installed as a package here; it is the sibling checkout, and it has to
# come first so this fork is what gets imported rather than any other copy.
VERL=${VERL:-$(cd "$V/../verl" 2>/dev/null && pwd)}
export PYTHONPATH=${VERL:+$VERL:}$V${PYTHONPATH:+:$PYTHONPATH}
mapfile -t BASE < <(grep -vE '^\s*(#|$)' "$V/vagen/configs/baseline_vllm.flags" | sed "s|\$V|$V|g")

PYTHONUNBUFFERED=1 python3 -m vagen.main_ppo \
    --config-path="$V/vagen/configs" --config-name=vagen_multiturn \
    hydra.searchpath="[file://$VERL/verl/trainer/config]" \
    data.custom_cls.path="$V/vagen/gym_agent_dataset.py" \
    "${BASE[@]}" \
    data.train_files="$SCRIPTDIR/train_sokoban_vision.yaml" \
    data.val_files="$SCRIPTDIR/val_sokoban_vision.yaml" \
    actor_rollout_ref.model.path="$MODEL" \
    critic.model.path="$MODEL" \
    critic.enable=True \
    algorithm.adv_estimator=default_gae \
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
