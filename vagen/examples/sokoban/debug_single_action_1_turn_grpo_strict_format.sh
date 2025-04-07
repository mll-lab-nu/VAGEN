set -x

export VLLM_ATTENTION_BACKEND=XFORMERS

export PYTHONHASHSEED=0

python vagen/env/create_dataset.py \
  --yaml_path vagen/configs/sokoban/debug_1_step.yaml \
  --force_gen

# max_trajectory_length = max_prompt_length + max_response_length
#Set use_remove_padding to false, if true, causing batch size must be postive error in vllm

python3 -m vagen.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    algorithm.high_level_gamma=0.95 \
    data.train_files=data/sokoban-text-1-step/train.parquet \
    data.val_files=data/sokoban-text-1-step/test.parquet \
    data.train_batch_size=10 \
    data.max_prompt_length=768 \
    data.max_response_length=256 \
    data.max_trajectory_length=1024 \
    data.image_key=images \
    actor_rollout_ref.model.path=Qwen/Qwen2.5-0.5B-Instruct \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=False \
    actor_rollout_ref.actor.ppo_mini_batch_size=16 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.ppo_mini_batch_size=1 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=mse \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.4 \
    actor_rollout_ref.rollout.enable_chunked_prefill=False \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.free_cache_engine=False \
    actor_rollout_ref.rollout.n=1 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    +actor_rollout_ref.ref.use_ref=True \
    algorithm.kl_ctrl.kl_coef=0.001 \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.project_name='vagen' \
    trainer.experiment_name='debug_single_action_single_turn_grpo_0_5B_kl_strict_format' \
    trainer.n_gpus_per_node=2 \
    trainer.nnodes=1 \
    trainer.save_freq=100 \
    trainer.test_freq=5 \
    trainer.total_epochs=15 \
    rollout_manager.max_turns=1 \
    rollout_manager.window_size=5 \
    trainer.val_before_train=True \
    trainer.val_generations_to_log_to_wandb=8 \
    rollout_manager.n_trajectory=1 \
    rollout_manager.use_loss_mask=True \
    2>&1 | tee debug_single_action_single_turn_grpo_0_5B_kl_strict_format.log
