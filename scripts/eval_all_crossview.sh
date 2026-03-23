#!/bin/bash
set -e

# ============================================================
# Batch evaluation script for all crossview experiments
# Runs val_only mode on best checkpoints with full test set (1050)
# Usage: nohup bash scripts/eval_all_crossview.sh > logs/eval_all.log 2>&1 &
# ============================================================

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR/.."
eval "$(conda shell.bash hook 2>/dev/null)"
conda activate vagen
export CUDA_VISIBLE_DEVICES=0,1,2,3
export VLLM_ATTENTION_BACKEND=XFORMERS
export PYTHONHASHSEED=0
export RAY_OBJECT_STORE_MEMORY=20000000000

CKPT_BASE="checkpoints/vagen_crossview_new"

# SFT checkpoint paths (for model architecture init)
# Set SFT_CKPT_DIR to your SFT results directory before running, e.g.:
#   export SFT_CKPT_DIR=/path/to/your/sft/results
if [ -z "$SFT_CKPT_DIR" ]; then
    echo "ERROR: SFT_CKPT_DIR is not set. Please set it to your SFT results directory."
    echo "  export SFT_CKPT_DIR=/path/to/sft/results"
    exit 1
fi
SFT_BASERL="${SFT_CKPT_DIR}/ff_rsn/checkpoint-57"
SFT_AUG_COGMAP="${SFT_CKPT_DIR}/aug_cgmap_ffr_out/checkpoint-45"
SFT_PLAIN_COGMAP="${SFT_CKPT_DIR}/plain_cgmap_ffr_out/checkpoint-50"
BASE_MODEL="Qwen/Qwen2.5-VL-3B-Instruct"

run_eval() {
    local EXPERIMENT_NAME=$1
    local EVAL_YAML=$2
    local MODEL_PATH=$3
    local CKPT_STEP=$4

    local RESUME_PATH="${CKPT_BASE}/${EXPERIMENT_NAME}/global_step_${CKPT_STEP}"

    echo "============================================================"
    echo "[$(date)] Starting eval: $EXPERIMENT_NAME (best step: $CKPT_STEP)"
    echo "  Model: $MODEL_PATH"
    echo "  Checkpoint: $RESUME_PATH"
    echo "============================================================"

    # Verify checkpoint exists
    if [ ! -d "$RESUME_PATH" ]; then
        echo "ERROR: Checkpoint not found: $RESUME_PATH"
        echo "[$(date)] SKIPPED: $EXPERIMENT_NAME"
        return 1
    fi

    # Create eval dataset
    mkdir -p "eval_data/$EXPERIMENT_NAME"
    python -m vagen.env.create_dataset \
        --force_gen \
        --yaml_path "$EVAL_YAML" \
        --train_path "eval_data/$EXPERIMENT_NAME/train.parquet" \
        --test_path "eval_data/$EXPERIMENT_NAME/test.parquet"

    # Run evaluation (val_only)
    python3 -m vagen.trainer.main_ppo \
        algorithm.adv_estimator=grpo \
        algorithm.high_level_gamma=1.0 \
        data.train_files="eval_data/$EXPERIMENT_NAME/train.parquet" \
        data.val_files="eval_data/$EXPERIMENT_NAME/test.parquet" \
        data.train_batch_size=32 \
        data.max_prompt_length=1024 \
        data.max_response_length=1536 \
        data.max_trajectory_length=3600 \
        data.image_key=images \
        data.truncation=left \
        actor_rollout_ref.model.path="$MODEL_PATH" \
        actor_rollout_ref.actor.optim.lr=1e-6 \
        actor_rollout_ref.model.use_remove_padding=True \
        actor_rollout_ref.actor.ppo_mini_batch_size=32 \
        actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
        actor_rollout_ref.actor.use_kl_loss=False \
        actor_rollout_ref.actor.kl_loss_coef=0.001 \
        actor_rollout_ref.actor.kl_loss_type=mse \
        actor_rollout_ref.model.enable_gradient_checkpointing=True \
        actor_rollout_ref.actor.fsdp_config.param_offload=True \
        actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
        actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
        actor_rollout_ref.rollout.tensor_model_parallel_size=4 \
        actor_rollout_ref.rollout.name=vllm \
        actor_rollout_ref.rollout.gpu_memory_utilization=0.3 \
        actor_rollout_ref.rollout.enable_chunked_prefill=False \
        actor_rollout_ref.rollout.enforce_eager=False \
        actor_rollout_ref.rollout.free_cache_engine=False \
        actor_rollout_ref.rollout.n=1 \
        actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
        actor_rollout_ref.ref.fsdp_config.param_offload=True \
        actor_rollout_ref.rollout.top_p=0.95 \
        actor_rollout_ref.rollout.temperature=0.7 \
        critic.optim.lr=1e-5 \
        critic.model.use_remove_padding=True \
        critic.model.path="$MODEL_PATH" \
        critic.model.enable_gradient_checkpointing=True \
        critic.ppo_micro_batch_size_per_gpu=1 \
        critic.model.fsdp_config.param_offload=False \
        critic.model.fsdp_config.optimizer_offload=False \
        algorithm.kl_ctrl.kl_coef=0.001 \
        trainer.critic_warmup=0 \
        trainer.logger='[console,wandb]' \
        trainer.project_name='vagen_crossview_eval' \
        trainer.experiment_name="${EXPERIMENT_NAME}-eval" \
        trainer.n_gpus_per_node=4 \
        trainer.nnodes=1 \
        trainer.save_freq=20 \
        trainer.test_freq=20 \
        trainer.total_training_steps=200 \
        trainer.remove_previous_ckpt_in_save=True \
        trainer.resume_mode="$RESUME_PATH" \
        rollout_manager.max_turns=1 \
        rollout_manager.window_size=5 \
        rollout_manager.use_multi_turn_reward=False \
        rollout_manager.use_loss_mask=True \
        rollout_manager.use_gae_mask=True \
        trainer.val_before_train=True \
        trainer.val_generations_to_log_to_wandb=0 \
        trainer.val_only=True \
        rollout_manager.n_trajectory=8

    echo "[$(date)] Finished eval: $EXPERIMENT_NAME"
    echo ""

    # Clean up ray between experiments
    pkill -9 -f ray || true
    sleep 15
}

# Create log directory
mkdir -p logs

# Clean up any leftover processes
pkill -9 -f ray || true
sleep 15

# ============================================================
# 1. baseRL (from scratch) - best step 120
# ============================================================
run_eval "crossview-baseRL" \
    "scripts/examples/crossview/baseRL/eval_env_config.yaml" \
    "$BASE_MODEL" \
    120

# ============================================================
# 2. cogmap_reasoning_no_sft (from scratch) - best step 100
# ============================================================
run_eval "crossview-cogmap_reasoning_no_sft" \
    "scripts/examples/crossview/cogmap_reasoning_no_sft/eval_env_config.yaml" \
    "$BASE_MODEL" \
    100

# ============================================================
# 3. cogmap_reasoning_plain_no_sft (from scratch) - best step 60
# ============================================================
run_eval "crossview-cogmap_reasoning_plain_no_sft" \
    "scripts/examples/crossview/cogmap_reasoning_plain_no_sft/eval_env_config.yaml" \
    "$BASE_MODEL" \
    60

# ============================================================
# 4. baseRL_sft (from SFT) - best step 180
# ============================================================
run_eval "crossview-baseRL_sft" \
    "scripts/examples/crossview/baseRL_sft/eval_env_config.yaml" \
    "$SFT_BASERL" \
    180

# ============================================================
# 5. cogmap_reasoning (from SFT) - best step 140
# ============================================================
run_eval "crossview-cogmap_reasoning" \
    "scripts/examples/crossview/cogmap_reasoning/eval_env_config.yaml" \
    "$SFT_AUG_COGMAP" \
    140

# ============================================================
# 6. cogmap_reasoning_plain (from SFT) - best step 180
# ============================================================
run_eval "crossview-cogmap_reasoning_plain" \
    "scripts/examples/crossview/cogmap_reasoning_plain/eval_env_config.yaml" \
    "$SFT_PLAIN_COGMAP" \
    180

echo "============================================================"
echo "[$(date)] All evaluations completed!"
echo "============================================================"
