#!/bin/bash
# Minimum example: single-GPU LoRA GRPO training with Qwen3-0.6B on sokoban (text mode)
# Note: LoRA currently requires vllm backend (sglang not yet supported)

set -x

PROJECT_NAME="vagen_experiments"
EXPERIMENT_NAME="sokoban_grpo_qwen3_06b_topp_filter_lora_text"

BASEDIR=$(pwd)
SCRIPTDIR=$(dirname "$0")
EXPERIMENT_DIR=${BASEDIR}/exps/${PROJECT_NAME}/${EXPERIMENT_NAME}
SAVE_CHECKPOINT_DIR=${EXPERIMENT_DIR}/verl_checkpoints
DATASET_TRAIN=${SCRIPTDIR}/train_sokoban_free_wm_text.yaml
DATASET_VAL=${SCRIPTDIR}/val_sokoban_free_wm_text.yaml
agent_loop_config_path=${BASEDIR}/vagen/configs/agent.yaml
REF_MODEL_PATH=Qwen/Qwen3-0.6B
mkdir -p ${EXPERIMENT_DIR}

PYTHONUNBUFFERED=1 python3 -m vagen.main_ppo \
    --config-path=${BASEDIR}/vagen/configs \
    --config-name='vagen_multiturn' \
    data.train_files=${DATASET_TRAIN} \
    data.val_files=${DATASET_VAL} \
    data.train_batch_size=16 \
    data.max_prompt_length=1000 \
    data.max_response_length=4000 \
    algorithm.adv_estimator=grpo \
    algorithm.kl_ctrl.kl_coef=0.001 \
    actor_rollout_ref.model.path=${REF_MODEL_PATH} \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.model.lora_rank=32 \
    actor_rollout_ref.model.lora_alpha=32 \
    actor_rollout_ref.model.target_modules=all-linear \
    actor_rollout_ref.actor.optim.lr=3e-5 \
    actor_rollout_ref.actor.ppo_mini_batch_size=16 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0.0 \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=1 \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.actor.checkpoint.save_contents=['model','hf_model','optimizer','extra'] \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.mode=async \
    actor_rollout_ref.rollout.load_format=safetensors \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.3 \
    actor_rollout_ref.rollout.n=4 \
    actor_rollout_ref.rollout.max_num_batched_tokens=5000 \
    actor_rollout_ref.rollout.max_model_len=5000 \
    actor_rollout_ref.rollout.enable_chunked_prefill=False \
    actor_rollout_ref.rollout.enforce_eager=True \
    actor_rollout_ref.rollout.free_cache_engine=True \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.multi_turn.enable=True \
    actor_rollout_ref.rollout.agent.agent_loop_config_path=$agent_loop_config_path \
    actor_rollout_ref.rollout.disable_log_stats=False \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.val_before_train=True \
    trainer.n_gpus_per_node=1 \
    trainer.nnodes=1 \
    trainer.save_freq=40 \
    trainer.test_freq=20 \
    trainer.project_name=${PROJECT_NAME} \
    trainer.experiment_name=${EXPERIMENT_NAME} \
    trainer.default_local_dir=${SAVE_CHECKPOINT_DIR} \
    trainer.validation_data_dir=${EXPERIMENT_DIR}/validation \
    trainer.rollout_data_dir=${EXPERIMENT_DIR}/rollout_data \
    trainer.log_val_generations=16 \
    trainer.max_actor_ckpt_to_keep=1 \
    trainer.max_critic_ckpt_to_keep=1 \
    filter.enable=True \
    trainer.total_training_steps=401 2>&1 | \
    tee ${EXPERIMENT_DIR}/${PROJECT_NAME}_${EXPERIMENT_NAME}.log >(tee ${BASEDIR}/${PROJECT_NAME}_${EXPERIMENT_NAME}.log >/dev/null)
