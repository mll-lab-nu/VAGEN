# set -x

# export VLLM_ATTENTION_BACKEND=XFORMERS

# python -m vagen.env.sokoban.create_dataset --data_dir data/sokoban-text

# # max_trajectory_length = max_prompt_length + max_response_length

# python3 -m vagen.trainer.main_ppo \
#     algorithm.adv_estimator=grpo \
#     data.train_files=data/sokoban-text/train.parquet \
#     data.val_files=data/sokoban-text/test.parquet \
#     data.train_batch_size=32 \
#     data.max_prompt_length=512 \
#     data.max_response_length=1536 \
#     +data.max_trajectory_length=2048 \
#     +data.max_response_per_turn=256 \
#     +actor_rollout_ref.rollout.max_response_per_turn=256 \
#     +actor_rollout_ref.rollout.max_trajectory_length=2048 \
#     data.image_key=images \
#     actor_rollout_ref.model.path=Qwen/Qwen2.5-0.5B-Instruct \
#     actor_rollout_ref.actor.optim.lr=1e-6 \
#     actor_rollout_ref.model.use_remove_padding=True \
#     actor_rollout_ref.actor.ppo_mini_batch_size=8 \
#     actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4 \
#     actor_rollout_ref.actor.use_kl_loss=True \
#     actor_rollout_ref.actor.kl_loss_coef=0.001 \
#     actor_rollout_ref.actor.kl_loss_type=low_var_kl \
#     actor_rollout_ref.model.enable_gradient_checkpointing=True \
#     actor_rollout_ref.actor.fsdp_config.param_offload=False \
#     actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
#     actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=4 \
#     actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
#     actor_rollout_ref.rollout.name=vllm \
#     actor_rollout_ref.rollout.gpu_memory_utilization=0.7 \
#     actor_rollout_ref.rollout.enable_chunked_prefill=False \
#     actor_rollout_ref.rollout.enforce_eager=False \
#     actor_rollout_ref.rollout.free_cache_engine=False \
#     actor_rollout_ref.rollout.n=1 \
#     actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4 \
#     actor_rollout_ref.ref.fsdp_config.param_offload=True \
#     algorithm.kl_ctrl.kl_coef=0.001 \
#     trainer.critic_warmup=0 \
#     trainer.logger=['console','wandb'] \
#     trainer.project_name='vagen' \
#     trainer.experiment_name='qwen2_5_05b_function_rm' \
#     trainer.n_gpus_per_node=1 \
#     trainer.nnodes=1 \
#     trainer.save_freq=-1 \
#     trainer.test_freq=-1 \
#     trainer.total_epochs=15 \
#     +max_turns=5 \
#     2>&1 | tee debug_qwen0_5_4_gpu_ppo.log
