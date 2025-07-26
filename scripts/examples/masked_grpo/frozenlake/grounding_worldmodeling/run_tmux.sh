#!/bin/bash
set -e  # Exit immediately if a command exits with a non-zero status

# Interactive input for port and CUDA devices
read -p "Enter port number (default: 5000): " PORT_INPUT
PORT=${PORT_INPUT:-5000}

read -p "Enter CUDA devices (default: 0,1,2,3): " CUDA_DEVICES
CUDA_DEVICES=${CUDA_DEVICES:-0,1,2,3}

# Get the directory of the script
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Extract experiment name from the path
# This will take the last 3 parts of the path: format/sokoban/free_think
EXPERIMENT_NAME=$(echo $SCRIPT_DIR | rev | cut -d'/' -f1-3 | rev | tr '/' '-')
echo "Experiment name: $EXPERIMENT_NAME"

# Find available session names
find_available_session() {
  local base_name=$1
  local count=0
  
  while tmux has-session -t "${base_name}${count}" 2>/dev/null; do
    count=$((count+1))
  done
  
  echo "${base_name}${count}"
}

# Create session names
SERVER_SESSION=$(find_available_session "server")
TRAIN_SESSION=$(find_available_session "train")

echo "Creating tmux sessions: $SERVER_SESSION and $TRAIN_SESSION"
echo "Using port: $PORT"
echo "Using CUDA devices: $CUDA_DEVICES"

# Create directories if they don't exist
mkdir -p "data/$EXPERIMENT_NAME"

# Create server session
tmux new-session -d -s "$SERVER_SESSION"
# Configure server session with conda and environment variables
tmux send-keys -t "$SERVER_SESSION" "source activate vagen" C-m
tmux send-keys -t "$SERVER_SESSION" "export CUDA_VISIBLE_DEVICES=$CUDA_DEVICES" C-m
tmux send-keys -t "$SERVER_SESSION" "export VLLM_ATTENTION_BACKEND=XFORMERS" C-m
tmux send-keys -t "$SERVER_SESSION" "export PYTHONHASHSEED=0" C-m
# Start the server
tmux send-keys -t "$SERVER_SESSION" "python -m vagen.server.server server.port=$PORT" C-m

# Wait for server to start
echo "Waiting for server to start on port $PORT..."
sleep 10  # Adjust as needed

# Create training session
tmux new-session -d -s "$TRAIN_SESSION"
# Configure training session with conda and environment variables
tmux send-keys -t "$TRAIN_SESSION" "cd $SCRIPT_DIR" C-m
tmux send-keys -t "$TRAIN_SESSION" "source activate vagen" C-m
tmux send-keys -t "$TRAIN_SESSION" "export CUDA_VISIBLE_DEVICES=$CUDA_DEVICES" C-m
tmux send-keys -t "$TRAIN_SESSION" "export VLLM_ATTENTION_BACKEND=XFORMERS" C-m
tmux send-keys -t "$TRAIN_SESSION" "export PYTHONHASHSEED=0" C-m
tmux send-keys -t "$TRAIN_SESSION" "set -x" C-m

# First create the dataset
tmux send-keys -t "$TRAIN_SESSION" "python -m vagen.env.create_dataset \\
    --yaml_path \"$SCRIPT_DIR/env_config.yaml\" \\
    --train_path \"data/$EXPERIMENT_NAME/train.parquet\" \\
    --test_path \"data/$EXPERIMENT_NAME/test.parquet\"" C-m

# Then start the training
tmux send-keys -t "$TRAIN_SESSION" "python3 -m vagen.trainer.main_ppo \\
    algorithm.adv_estimator=grpo \\
    algorithm.high_level_gamma=1.0 \\
    data.train_files=data/$EXPERIMENT_NAME/train.parquet \\
    data.val_files=data/$EXPERIMENT_NAME/test.parquet \\
    data.train_batch_size=16 \\
    data.max_prompt_length=1024 \\
    data.max_response_length=200 \\
    data.max_trajectory_length=2400 \\
    data.image_key=images \\
    data.truncation=left \\
    actor_rollout_ref.model.path=OpenGVLab/InternVL3-1B-hf \\
    actor_rollout_ref.actor.optim.lr=1e-6 \\
    actor_rollout_ref.model.use_remove_padding=True \\
    actor_rollout_ref.actor.ppo_mini_batch_size=32 \\
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \\
    actor_rollout_ref.actor.use_kl_loss=False \\
    actor_rollout_ref.actor.kl_loss_coef=0.001 \\
    actor_rollout_ref.actor.kl_loss_type=mse \\
    actor_rollout_ref.model.enable_gradient_checkpointing=True \\
    actor_rollout_ref.actor.fsdp_config.param_offload=True \\
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \\
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \\
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \\
    actor_rollout_ref.rollout.name=vllm \\
    actor_rollout_ref.rollout.gpu_memory_utilization=0.1 \\
    actor_rollout_ref.rollout.enable_chunked_prefill=False \\
    actor_rollout_ref.rollout.enforce_eager=False \\
    actor_rollout_ref.rollout.free_cache_engine=False \\
    actor_rollout_ref.rollout.n=1 \\
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \\
    actor_rollout_ref.ref.fsdp_config.param_offload=True \\
    actor_rollout_ref.rollout.top_p=0.95 \\
    actor_rollout_ref.rollout.temperature=0.7 \\
    critic.optim.lr=1e-5 \\
    critic.model.use_remove_padding=True \\
    critic.model.path=OpenGVLab/InternVL3-1B-hf \\
    critic.model.enable_gradient_checkpointing=True \\
    critic.ppo_micro_batch_size_per_gpu=1 \\
    critic.model.fsdp_config.param_offload=False \\
    critic.model.fsdp_config.optimizer_offload=False \\
    algorithm.kl_ctrl.kl_coef=0.001 \\
    trainer.critic_warmup=0 \\
    trainer.logger=['console','wandb'] \\
    trainer.project_name='vagen_new' \\
    trainer.experiment_name=$EXPERIMENT_NAME \\
    trainer.n_gpus_per_node=1 \\
    trainer.nnodes=1 \\
    trainer.save_freq=250 \\
    trainer.test_freq=20 \\
    trainer.total_training_steps=300 \\
    rollout_manager.max_turns=3 \\
    rollout_manager.window_size=5 \\
    rollout_manager.use_multi_turn_reward=False \\
    rollout_manager.use_loss_mask=True \\
    rollout_manager.use_gae_mask=True \\
    trainer.val_before_train=True \\
    trainer.val_generations_to_log_to_wandb=8 \\
    rollout_manager.n_trajectory=8 \\
    rollout_manager.use_service=True \\
    rollout_manager.timeout=300 \\
    rollout_manager.base_url=\"http://localhost:$PORT\" \\
    2>&1 | tee $EXPERIMENT_NAME.log" C-m

echo "-----------------------------------------"
echo "Configuration Summary:"
echo "Port: $PORT"
echo "CUDA Devices: $CUDA_DEVICES"
echo "Server Session: $SERVER_SESSION"
echo "Training Session: $TRAIN_SESSION"
echo "-----------------------------------------"
echo "To attach to server session: tmux attach-session -t $SERVER_SESSION"
echo "To attach to training session: tmux attach-session -t $TRAIN_SESSION"
echo "NOTE: The sessions will remain active. To detach from a session use Ctrl+B followed by D"