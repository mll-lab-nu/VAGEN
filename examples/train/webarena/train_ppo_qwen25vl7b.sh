#!/bin/bash
# WebArena environment training with PPO.
#
# Prerequisites (do these in separate terminals before running this script):
#
#   1) SSH tunnel to the WebArena Docker host (forward 7770/7780/9999/8023/8888/4399).
#      See vagen/envs/webarena/README.md for the exact command.
#
#   2) Source env vars (REQUIRED — env_config raises without these):
#        conda activate webarena
#        source vagen/envs/webarena/setup_vars.sh
#
#   3) Start the TRAIN server (loads 647 train tasks):
#        PYTHONPATH=. python -m vagen.envs.webarena.serve \
#            --task_config_file=vagen/envs/webarena/config_files/normalized_train.json \
#            --n_browsers=4 --max_contexts_per_browser=16 \
#            --port=8002 --auth_cache_dir=./.wa_auth
#
#   4) Start the VAL server (loads 165 official test tasks):
#        PYTHONPATH=. python -m vagen.envs.webarena.serve \
#            --task_config_file=vagen/envs/webarena/config_files/normalized_test.json \
#            --n_browsers=2 --max_contexts_per_browser=16 \
#            --port=8003 --auth_cache_dir=./.wa_auth
#
# Total rollout concurrency is bounded by n_browsers × max_contexts_per_browser
# on each server. Make sure the trainer's effective batch concurrency does not
# exceed that, otherwise step latency will spike.

set -x

PROJECT_NAME="vagen_webarena"
EXPERIMENT_NAME="webarena_ppo_qwen25vl7b"

BASEDIR=$(pwd)
SCRIPTDIR=$(dirname "$0")
EXPERIMENT_DIR=${BASEDIR}/exps/${PROJECT_NAME}/${EXPERIMENT_NAME}
SAVE_CHECKPOINT_DIR=${EXPERIMENT_DIR}/verl_checkpoints
DATASET_TRAIN=${SCRIPTDIR}/train_webarena.yaml
DATASET_VAL=${SCRIPTDIR}/val_webarena.yaml
agent_loop_config_path=${BASEDIR}/vagen/configs/agent.yaml
REF_MODEL_PATH=Qwen/Qwen2.5-VL-7B-Instruct
mkdir -p ${EXPERIMENT_DIR}

PYTHONUNBUFFERED=1 python3 -m vagen.main_ppo \
    --config-path=${BASEDIR}/vagen/configs \
    --config-name='vagen_multiturn' \
    data.train_files=${DATASET_TRAIN} \
    data.val_files=${DATASET_VAL} \
    data.train_batch_size=64 \
    data.max_prompt_length=12000 \
    data.max_response_length=2000 \
    algorithm.adv_estimator=gae \
    algorithm.kl_ctrl.kl_coef=0.0 \
    actor_rollout_ref.model.path=${REF_MODEL_PATH} \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.use_fused_kernels=True \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size=16 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.kl_loss_coef=0.0 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0.0 \
    actor_rollout_ref.actor.checkpoint.save_contents=['model','hf_model','optimizer','extra'] \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=1 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=sglang \
    actor_rollout_ref.rollout.mode=async \
    actor_rollout_ref.rollout.n=1 \
    actor_rollout_ref.rollout.max_num_batched_tokens=16000 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
    actor_rollout_ref.rollout.enforce_eager=True \
    actor_rollout_ref.rollout.free_cache_engine=True \
    actor_rollout_ref.rollout.enable_chunked_prefill=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.rollout.multi_turn.enable=True \
    actor_rollout_ref.rollout.agent.agent_loop_config_path=$agent_loop_config_path \
    actor_rollout_ref.rollout.disable_log_stats=False \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.val_before_train=True \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.save_freq=20 \
    trainer.test_freq=20 \
    trainer.project_name=${PROJECT_NAME} \
    trainer.experiment_name=${EXPERIMENT_NAME} \
    trainer.default_local_dir=${SAVE_CHECKPOINT_DIR} \
    trainer.validation_data_dir=${EXPERIMENT_DIR}/validation \
    trainer.rollout_data_dir=${EXPERIMENT_DIR}/rollout_data \
    trainer.log_val_generations=32 \
    critic.optim.lr=1e-5 \
    critic.model.use_remove_padding=True \
    critic.model.path=${REF_MODEL_PATH} \
    critic.model.enable_gradient_checkpointing=True \
    critic.ppo_micro_batch_size_per_gpu=1 \
    critic.model.fsdp_config.param_offload=True \
    critic.model.fsdp_config.optimizer_offload=True \
    huggingface_hub.hf_save_freq=200 \
    trainer.total_training_steps=401 2>&1 | \
    tee ${EXPERIMENT_DIR}/${PROJECT_NAME}_${EXPERIMENT_NAME}.log
