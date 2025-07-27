#!/bin/bash
set -e  # Exit immediately if a command exits with a non-zero status

# Configuration - Set these values manually
PORT=5000
CUDA_DEVICES="0,1,2,3,4,5,6,7"

# Get the directory of the script
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Extract experiment name from the path
EXPERIMENT_NAME=$(echo $SCRIPT_DIR | rev | cut -d'/' -f1-3 | rev | tr '/' '-')
echo "Experiment name: $EXPERIMENT_NAME"
echo "Using port: $PORT"
echo "Using CUDA devices: $CUDA_DEVICES"

# Create directories if they don't exist
mkdir -p "data/$EXPERIMENT_NAME"

# Set environment variables
export CUDA_VISIBLE_DEVICES=$CUDA_DEVICES
export VLLM_ATTENTION_BACKEND=XFORMERS
export PYTHONHASHSEED=0

# Activate conda environment
source activate vagen

echo "Starting server in background..."
# Start server in background and save PID
python -m vagen.server.server server.port=$PORT &
SERVER_PID=$!
echo "Server started with PID: $SERVER_PID"

# Create a cleanup function to kill server on script exit
cleanup() {
    echo "Cleaning up..."
    if kill -0 $SERVER_PID 2>/dev/null; then
        echo "Stopping server (PID: $SERVER_PID)..."
        kill $SERVER_PID
        wait $SERVER_PID 2>/dev/null || true
    fi
}

# Set trap to cleanup on script exit
trap cleanup EXIT INT TERM

# Wait for server to start
echo "Waiting for server to start on port $PORT..."
sleep 15

# Test if server is responsive
echo "Testing server connection..."
for i in {1..10}; do
    if curl -s "http://localhost:$PORT/health" > /dev/null 2>&1; then
        echo "Server is ready!"
        break
    elif [ $i -eq 10 ]; then
        echo "Server failed to start properly"
        exit 1
    else
        echo "Waiting for server... (attempt $i/10)"
        sleep 2
    fi
done

echo "Creating dataset..."
# Create the dataset
python -m vagen.env.create_dataset \
    --yaml_path "$SCRIPT_DIR/env_config.yaml" \
    --train_path "$SCRIPT_DIR/data/$EXPERIMENT_NAME/train.parquet" \
    --test_path "$SCRIPT_DIR/data/$EXPERIMENT_NAME/test.parquet" \
    2>&1 | tee "$SCRIPT_DIR/server.log"

echo "Starting training..."
# Start the training (output directly to console)
cd "$SCRIPT_DIR"
set -x  # Enable command echoing

python3 -m vagen.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    algorithm.high_level_gamma=1.0 \
    data.train_files="$SCRIPT_DIR/data/$EXPERIMENT_NAME/train.parquet" \
    data.val_files="$SCRIPT_DIR/data/$EXPERIMENT_NAME/test.parquet" \
    data.train_batch_size=16 \
    data.max_prompt_length=1024 \
    data.max_response_length=200 \
    data.max_trajectory_length=2400 \
    data.image_key=images \
    data.truncation=left \
    actor_rollout_ref.model.path=Qwen/Qwen2.5-VL-3B-Instruct \
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
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.1 \
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
    critic.model.path=Qwen/Qwen2.5-VL-3B-Instruct \
    critic.model.enable_gradient_checkpointing=True \
    critic.ppo_micro_batch_size_per_gpu=1 \
    critic.model.fsdp_config.param_offload=False \
    critic.model.fsdp_config.optimizer_offload=False \
    algorithm.kl_ctrl.kl_coef=0.001 \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.project_name='vagen_new' \
    trainer.experiment_name=$EXPERIMENT_NAME \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.save_freq=250 \
    trainer.test_freq=20 \
    trainer.total_training_steps=300 \
    rollout_manager.max_turns=3 \
    rollout_manager.window_size=5 \
    rollout_manager.use_multi_turn_reward=False \
    rollout_manager.use_loss_mask=True \
    rollout_manager.use_gae_mask=True \
    trainer.val_before_train=True \
    trainer.val_generations_to_log_to_wandb=8 \
    rollout_manager.n_trajectory=8 \
    rollout_manager.use_service=True \
    rollout_manager.timeout=300 \
    rollout_manager.base_url="http://localhost:$PORT"

echo "Training completed!" 