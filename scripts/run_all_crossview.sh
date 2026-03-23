#!/bin/bash
set -e

# ============================================================
# Relay training script for all crossview experiments
# Run this on qgpu2011 with: bash scripts/run_all_crossview.sh
# ============================================================

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR/.."
eval "$(conda shell.bash hook 2>/dev/null)"
conda activate vagen
export CUDA_VISIBLE_DEVICES=0,1,2,3
export VLLM_ATTENTION_BACKEND=XFORMERS
export PYTHONHASHSEED=0
export RAY_OBJECT_STORE_MEMORY=20000000000

# SFT checkpoint paths
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

run_experiment() {
    local EXPERIMENT_NAME=$1
    local YAML_PATH=$2
    local MODEL_PATH=$3

    echo "============================================================"
    echo "[$(date)] Starting experiment: $EXPERIMENT_NAME"
    echo "  Model: $MODEL_PATH"
    echo "============================================================"

    # Create dataset
    mkdir -p "data/$EXPERIMENT_NAME"
    python -m vagen.env.create_dataset \
        --force_gen \
        --yaml_path "$YAML_PATH" \
        --train_path "data/$EXPERIMENT_NAME/train.parquet" \
        --test_path "data/$EXPERIMENT_NAME/test.parquet"

    # Run training
    python3 -m vagen.trainer.main_ppo \
        algorithm.adv_estimator=grpo \
        algorithm.high_level_gamma=1.0 \
        data.train_files="data/$EXPERIMENT_NAME/train.parquet" \
        data.val_files="data/$EXPERIMENT_NAME/test.parquet" \
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
        trainer.project_name='vagen_crossview_new' \
        trainer.experiment_name="$EXPERIMENT_NAME" \
        trainer.n_gpus_per_node=4 \
        trainer.nnodes=1 \
        trainer.save_freq=20 \
        trainer.test_freq=20 \
        trainer.total_training_steps=200 \
        trainer.remove_previous_ckpt_in_save=True \
        rollout_manager.max_turns=1 \
        rollout_manager.window_size=5 \
        rollout_manager.use_multi_turn_reward=False \
        rollout_manager.use_loss_mask=True \
        rollout_manager.use_gae_mask=True \
        trainer.val_before_train=True \
        trainer.val_generations_to_log_to_wandb=1 \
        trainer.val_only=False \
        rollout_manager.n_trajectory=8 \
        2>&1 | tee "logs/${EXPERIMENT_NAME}.log"

    echo "[$(date)] Finished experiment: $EXPERIMENT_NAME"
    echo ""

    # Clean up ray between experiments
    pkill -9 -f ray || true
    sleep 10
}

# Create log directory
mkdir -p logs

# Clean up any leftover processes
pkill -9 -f ray || true
sleep 15

# ============================================================
# Experiment 1: baseRL from SFT
# ============================================================
run_experiment "crossview-baseRL_sft" \
    "scripts/examples/crossview/baseRL_sft/env_config.yaml" \
    "$SFT_BASERL"

# ============================================================
# Experiment 5: cogmap_and_reasoning from SFT
# ============================================================
run_experiment "crossview-cogmap_reasoning" \
    "scripts/examples/crossview/cogmap_reasoning/env_config.yaml" \
    "$SFT_AUG_COGMAP"

# ============================================================
# Experiment 6: cogmap_and_reasoning_plain from SFT
# ============================================================
run_experiment "crossview-cogmap_reasoning_plain" \
    "scripts/examples/crossview/cogmap_reasoning_plain/env_config.yaml" \
    "$SFT_PLAIN_COGMAP"

echo "============================================================"
echo "[$(date)] All experiments completed!"
echo "============================================================"
