#!/bin/bash
# sokoban - grpo - concat - Qwen/Qwen3-0.6B
#
# No KL term, though this script used to ask for one. It needs a reference policy and
# this trainer path has none to give: under LoRA the reference is the actor with the
# adapter off, so main_ppo registers no RefPolicy worker while the separated trainer asks
# for one regardless -- KeyError before step 0. Without LoRA it registers ActorRolloutRef
# instead, which that trainer does not look for either.
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
PROJECT_NAME=${PROJECT_NAME:-vagen_experiments}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-sokoban_grpo_qwen3_06b_topp_filter_lora_text}
EXPERIMENT_DIR=${EXPERIMENT_DIR:-$V/exps/$PROJECT_NAME/$EXPERIMENT_NAME}
MODEL=${MODEL:-Qwen/Qwen3-0.6B}
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
    data.train_files="$SCRIPTDIR/train_sokoban_free_wm_text.yaml" \
    data.val_files="$SCRIPTDIR/val_sokoban_free_wm_text.yaml" \
    actor_rollout_ref.model.path="$MODEL" \
    critic.model.path="$MODEL" \
    algorithm.adv_estimator=grpo \
    actor_rollout_ref.actor.optim.lr=3e-5 \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.rollout.max_model_len=5000 \
    trainer.log_val_generations=16 \
    trainer.harness=concat \
    data.train_batch_size=16 \
    data.max_response_length=4000 \
    actor_rollout_ref.actor.ppo_mini_batch_size=16 \
    actor_rollout_ref.rollout.n=4 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.3 \
    actor_rollout_ref.rollout.max_num_batched_tokens=5000 \
    actor_rollout_ref.rollout.enforce_eager=True \
    actor_rollout_ref.rollout.free_cache_engine=True \
    actor_rollout_ref.rollout.enable_chunked_prefill=False \
    actor_rollout_ref.model.lora_rank=32 \
    actor_rollout_ref.model.lora_alpha=32 \
    actor_rollout_ref.model.target_modules=all-linear \
    actor_rollout_ref.rollout.load_format=safetensors \
    filter.enable=True \
    trainer.n_gpus_per_node=1 \
    trainer.nnodes=1 \
    trainer.critic_warmup=0 \
    trainer.save_freq=40 \
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
