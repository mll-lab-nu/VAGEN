#!/bin/bash
# sokoban - default_gae - InternVL3.5-2B
#
# Select the row layout with HARNESS=concat|no_concat|compact. InternVL uses strict
# <think>...</think><answer>...</answer> parsing.
set -eo pipefail

V=$(cd "$(dirname "$0")/../../.." && pwd)
SCRIPTDIR=$(cd "$(dirname "$0")" && pwd)
HARNESS=${HARNESS:-concat}
case "$HARNESS" in
    concat|no_concat|compact) ;;
    *) echo "HARNESS must be concat, no_concat, or compact (got: $HARNESS)" >&2; exit 2 ;;
esac

PROJECT_NAME=${PROJECT_NAME:-vagen_experiments}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-sokoban_default_gae_internvl35_2b_${HARNESS}}
EXPERIMENT_DIR=${EXPERIMENT_DIR:-$V/exps/$PROJECT_NAME/$EXPERIMENT_NAME}
MODEL=${MODEL:-OpenGVLab/InternVL3_5-2B-hf}
COMPACT_BUDGET=${COMPACT_BUDGET:-2000}
COMPACT_SUMMARY_BUDGET=${COMPACT_SUMMARY_BUDGET:-300}
mkdir -p "$EXPERIMENT_DIR"

if [ -z "${VERL:-}" ]; then
    for d in "$V/verl" "$V/../verl"; do
        if [ -f "$d/verl/trainer/config/ppo_trainer.yaml" ]; then
            VERL=$(cd "$d" && pwd)
            break
        fi
    done
fi
if [ -z "$VERL" ]; then
    echo "verl not found at $V/verl or $V/../verl." >&2
    echo "Run: git submodule update --init --recursive   (or set VERL=/path/to/verl)" >&2
    exit 1
fi
export PYTHONPATH=${VERL:+$VERL:}$V${PYTHONPATH:+:$PYTHONPATH}
mapfile -t BASE < <(grep -vE '^\s*(#|$)' "$V/vagen/configs/baseline_vllm.flags" | sed "s|\$V|$V|g")

# InternVL's real Sokoban opening is 938 tokens; reopening from a full 300-token
# compact summary is 1244. The shared 1000-token prompt region cannot hold that
# second conversation, so reserve measured headroom explicitly below.
# InternVL also has no family-specific fused forward in verl. The generic fused forward
# is text-only and drops pixel_values, so actor/ref would train blind while vLLM sees the
# image. Keep the native multimodal forward until an adapter exists.
#
# The outer InternVL config inherits PretrainedConfig.tie_word_embeddings=True while its
# Qwen3 text_config and checkpoint both require False (the lm_head and token embedding
# are distinct). vLLM 0.22's Transformers-5 compatibility shim copies that outer default
# into text_config, silently discards the trained lm_head, and generates repetitive
# garbage. Override the outer value at engine construction; this changes no tokens or
# model output protocol, it only makes rollout load the checkpoint's real output head.
PYTHONUNBUFFERED=1 python3 -m vagen.training.main \
    --config-path="$V/vagen/configs" --config-name=vagen_multiturn \
    hydra.searchpath="[file://$VERL/verl/trainer/config]" \
    data.custom_cls.path="$V/vagen/training/dataset.py" \
    "${BASE[@]}" \
    data.train_files="$SCRIPTDIR/train_sokoban_vision_internvl.yaml" \
    data.val_files="$SCRIPTDIR/val_sokoban_vision_internvl.yaml" \
    actor_rollout_ref.model.path="$MODEL" \
    actor_rollout_ref.model.use_fused_kernels=False \
    critic.model.path="$MODEL" \
    critic.enable=True \
    algorithm.adv_estimator=default_gae \
    trainer.harness="$HARNESS" \
    trainer.compact_budget="$COMPACT_BUDGET" \
    trainer.compact_summary_budget="$COMPACT_SUMMARY_BUDGET" \
    data.train_batch_size=128 \
    data.max_prompt_length=1600 \
    data.max_response_length=4000 \
    actor_rollout_ref.actor.ppo_mini_batch_size=32 \
    actor_rollout_ref.rollout.n=1 \
    actor_rollout_ref.rollout.temperature=0.6 \
    actor_rollout_ref.rollout.top_k=50 \
    actor_rollout_ref.rollout.top_p=0.95 \
    actor_rollout_ref.rollout.val_kwargs.temperature=0.6 \
    actor_rollout_ref.rollout.val_kwargs.top_k=50 \
    actor_rollout_ref.rollout.val_kwargs.top_p=0.95 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.max_model_len=6000 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.max_num_batched_tokens=10000 \
    actor_rollout_ref.rollout.enforce_eager=True \
    actor_rollout_ref.rollout.free_cache_engine=True \
    actor_rollout_ref.rollout.enable_chunked_prefill=False \
    +actor_rollout_ref.rollout.engine_kwargs.vllm.hf_overrides.tie_word_embeddings=false \
    trainer.n_gpus_per_node=4 \
    trainer.nnodes=1 \
    trainer.critic_warmup=0 \
    trainer.save_freq=100 \
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
