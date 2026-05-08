#!/bin/bash
# WebAgent-R1 reproduction on Qwen2.5-3B SFT (weizhepei) with M-GRPO.
#
# Hardware: 2× RTX PRO 6000 Blackwell (96 GB each = 192 GB total VRAM)
# Paper baseline: 8× A100 80GB (640 GB) — we have ~30% memory budget,
# so we keep paper-faithful per-step batch but smaller GRPO group size.
#
# Hyperparameters per WebAgent-R1 paper (arXiv 2505.16421v2, Appendix B):
#   lr = constant 1e-6, batch = 16, KL β = 0.001, clip ε = 0.2,
#   max_prompt = 16384, max_response = 1024, T = 1.0, top_p = 1.0
#
# Prerequisites (see RUN_INSTRUCTIONS.md; runbook below):
#   1. Two webarena servers running on dt-login03:
#        port 8002: serves normalized_train.json (RL training tasks)
#        port 8003: serves normalized_test.json   (validation tasks)
#      Each launched fresh (kill leaked sessions + chromium first).
#      Pool size: --n_browsers=8 --max_contexts_per_browser=8 (= 64 total)
#   2. SLURM job with 2× RTX PRO 6000 Blackwell, 24h+ allocation.
#   3. WANDB_API_KEY exported (or remove wandb from logger).
#   4. The verl symlink + .pth fix from earlier smoke test still in place.

set -x

PROJECT_NAME="webarena_r1_qwen25_3b"
EXPERIMENT_NAME="grpo_weizhepei_full"

BASEDIR=$(pwd)
SCRIPTDIR=$(dirname "$0")
EXPERIMENT_DIR=${BASEDIR}/exps/${PROJECT_NAME}/${EXPERIMENT_NAME}
SAVE_CHECKPOINT_DIR=${EXPERIMENT_DIR}/verl_checkpoints
DATASET_TRAIN=${SCRIPTDIR}/train_webarena_full.yaml
DATASET_VAL=${SCRIPTDIR}/val_webarena_full.yaml
agent_loop_config_path=${BASEDIR}/vagen/configs/agent.yaml

# weizhepei's SFT baseline — this is the RL initialization policy in the paper
REF_MODEL_PATH=weizhepei/Qwen2.5-3B-WebArena-Lite-SFT-epoch-5

mkdir -p ${EXPERIMENT_DIR}

# CPU thread caps to avoid pthread limits on shared nodes
export OPENBLAS_NUM_THREADS=4
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export TOKENIZERS_PARALLELISM=false
export HF_HOME=${HF_HOME:-/workspace/hf_cache}


which python
PYTHONUNBUFFERED=1 python -m vagen.main_ppo \
    --config-path=${BASEDIR}/vagen/configs \
    --config-name='vagen_multiturn' \
    \
    `# Data — paper sets max context 16384 + max_response 1024.` \
    `# max_prompt_length bumped from 15360 to 24000: WebArena page HTML` \
    `# routinely exceeds 16k tokens; smaller cap caused -442 max_tokens.` \
    data.train_files=${DATASET_TRAIN} \
    data.val_files=${DATASET_VAL} \
    data.train_batch_size=16 \
    data.max_prompt_length=24000 \
    data.max_response_length=1024 \
    \
    `# Algorithm = GRPO (paper's M-GRPO maps to verl's grpo adv_estimator)` \
    algorithm.adv_estimator=grpo \
    algorithm.kl_ctrl.kl_coef=0.0 \
    \
    `# Actor — lr 1e-6 constant, KL β=0.001 in-loss, clip 0.2` \
    actor_rollout_ref.model.path=${REF_MODEL_PATH} \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.use_fused_kernels=False \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.optim.lr_warmup_steps=0 \
    actor_rollout_ref.actor.optim.warmup_style=constant \
    actor_rollout_ref.actor.ppo_mini_batch_size=16 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.clip_ratio=0.2 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0.0 \
    actor_rollout_ref.actor.checkpoint.save_contents=['model','hf_model','optimizer','extra'] \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=1 \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    \
    `# Reference policy — frozen, used for KL` \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.ref.fsdp_config.param_offload=False \
    \
    `# Rollout — paper uses vLLM, gpu_mem 0.7, T=1.0, top_p=1.0` \
    `# Group size n=4 (down from paper's likely 8) to fit our smaller VRAM` \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.mode=async \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.n=4 \
    actor_rollout_ref.rollout.temperature=1.0 \
    actor_rollout_ref.rollout.top_p=1.0 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.55 \
    actor_rollout_ref.rollout.max_num_batched_tokens=16384 \
    actor_rollout_ref.rollout.enforce_eager=True \
    actor_rollout_ref.rollout.free_cache_engine=True \
    actor_rollout_ref.rollout.enable_chunked_prefill=True \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.disable_log_stats=False \
    actor_rollout_ref.rollout.multi_turn.enable=True \
    actor_rollout_ref.rollout.agent.num_workers=16 \
    actor_rollout_ref.rollout.agent.agent_loop_config_path=$agent_loop_config_path \
    \
    `# Critic — GRPO does not use a critic; turn it off via mini-bs=0 isn't supported,` \
    `# so we set it minimal and rely on adv_estimator=grpo to skip critic updates.` \
    `# Keeping critic config for verl validation; it will not be invoked.` \
    critic.optim.lr=1e-5 \
    critic.model.use_remove_padding=True \
    critic.model.path=${REF_MODEL_PATH} \
    critic.model.enable_gradient_checkpointing=True \
    critic.ppo_micro_batch_size_per_gpu=1 \
    critic.model.fsdp_config.param_offload=True \
    critic.model.fsdp_config.optimizer_offload=True \
    \
    `# Trainer — 2 GPUs, val every 25 steps, save every 25 steps` \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.val_before_train=True \
    trainer.n_gpus_per_node=2 \
    trainer.nnodes=1 \
    trainer.save_freq=25 \
    trainer.test_freq=25 \
    trainer.project_name=${PROJECT_NAME} \
    trainer.experiment_name=${EXPERIMENT_NAME} \
    trainer.default_local_dir=${SAVE_CHECKPOINT_DIR} \
    trainer.validation_data_dir=${EXPERIMENT_DIR}/validation \
    trainer.rollout_data_dir=${EXPERIMENT_DIR}/rollout_data \
    trainer.log_val_generations=8 \
    trainer.total_training_steps=200 2>&1 | \
    tee ${EXPERIMENT_DIR}/${PROJECT_NAME}_${EXPERIMENT_NAME}.log
