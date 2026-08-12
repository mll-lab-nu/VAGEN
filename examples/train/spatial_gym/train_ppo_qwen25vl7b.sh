#!/bin/bash
# spatial_gym - default_gae - concat - Qwen/Qwen2.5-VL-7B-Instruct
#
# Per-experiment settings only. Everything that makes a VAGEN run work at all lives in
# vagen/configs/baseline_vllm.flags and is read below -- in particular the two flags that
# select VAGEN's agent loop. Without them verl runs its own, and the job comes up looking
# healthy while none of this repo's rollout code executes. All twenty of these scripts
# were in that state, and the duplication is why: each carried its own copy of the stack
# settings, and the copies stopped including the loop.
#
# Anything after "${BASE[@]}" overrides it, and anything on the command line overrides
# that, so a one-off sweep needs no edit here.
set -eo pipefail

V=$(cd "$(dirname "$0")/../../.." && pwd)
SCRIPTDIR=$(cd "$(dirname "$0")" && pwd)
PROJECT_NAME=${PROJECT_NAME:-vagen_spatial_gym}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-ppo_qwen25vl7b}
EXPERIMENT_DIR=${EXPERIMENT_DIR:-$V/exps/$PROJECT_NAME/$EXPERIMENT_NAME}
MODEL=${MODEL:-Qwen/Qwen2.5-VL-7B-Instruct}
mkdir -p "$EXPERIMENT_DIR"

# verl is not installed as a package here; it is the sibling checkout, and it has to
# come first so this fork is what gets imported rather than any other copy.
VERL=${VERL:-$(cd "$V/../verl" 2>/dev/null && pwd)}
export PYTHONPATH=${VERL:+$VERL:}$V${PYTHONPATH:+:$PYTHONPATH}
mapfile -t BASE < <(grep -vE '^\s*(#|$)' "$V/vagen/configs/baseline_vllm.flags" | sed "s|\$V|$V|g")

# train_batch_size is 32, not 128: the dataset is n_envs=50 and the loader drops
# the last partial batch, so 128 yields zero batches and the trainer asserts.
PYTHONUNBUFFERED=1 python3 -m vagen.main_ppo \
    --config-path="$V/vagen/configs" --config-name=vagen_multiturn \
    hydra.searchpath="[file://$VERL/verl/trainer/config]" \
    data.custom_cls.path="$V/vagen/gym_agent_dataset.py" \
    "${BASE[@]}" \
    data.train_files="$SCRIPTDIR/train_spatial_gym_vision.yaml" \
    data.val_files="$SCRIPTDIR/val_spatial_gym_vision.yaml" \
    actor_rollout_ref.model.path="$MODEL" \
    critic.model.path="$MODEL" \
    critic.enable=True \
    algorithm.adv_estimator=default_gae \
    trainer.harness=concat \
    data.train_batch_size=32 \
    data.max_prompt_length=4000 \
    data.max_response_length=2000 \
    actor_rollout_ref.actor.ppo_mini_batch_size=32 \
    actor_rollout_ref.rollout.n=1 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
    actor_rollout_ref.rollout.max_num_batched_tokens=10000 \
    actor_rollout_ref.rollout.enforce_eager=True \
    actor_rollout_ref.rollout.free_cache_engine=True \
    actor_rollout_ref.rollout.enable_chunked_prefill=True \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.critic_warmup=0 \
    trainer.save_freq=5 \
    trainer.test_freq=2 \
    trainer.total_training_steps=10 \
    trainer.project_name="$PROJECT_NAME" \
    trainer.experiment_name="$EXPERIMENT_NAME" \
    trainer.default_local_dir="$EXPERIMENT_DIR/verl_checkpoints" \
    trainer.rollout_data_dir="$EXPERIMENT_DIR/rollout_data" \
    trainer.validation_data_dir="$EXPERIMENT_DIR/validation" \
    actor_rollout_ref.actor.checkpoint.save_contents="['model','hf_model','optimizer','extra']" \
    "$@" \
    2>&1 | tee "$EXPERIMENT_DIR/run.log"
