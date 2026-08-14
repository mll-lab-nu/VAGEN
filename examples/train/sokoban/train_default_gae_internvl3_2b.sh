#!/bin/bash
# sokoban - default_gae - concat - OpenGVLab/InternVL3-2B-hf
#
# The `-hf` repo, not `OpenGVLab/InternVL3-2B`: the plain one ships its own modelling code
# and only loads under trust_remote_code, which the rollout engine does not use.
#
# Not an mrope model, so its processor has no get_rope_index and the agent loop takes the
# plain-position-ids path. tests/test_vlm_families.py covers that branch on this family.
#
# free_wm, to keep every non-Qwen2.5 script on one format. Note this family does *not*
# need it: InternVL3 tokenizes `<think>` as three ordinary text tokens, exactly like
# Qwen2.5-VL, so `wm` is reachable here. Switch this pair back to train/val_sokoban_vision
# .yaml if you would rather have it directly comparable with the Qwen2.5-VL runs.
#
# ★ DOES NOT RUN ON THIS STACK. vLLM 0.22 dies with a CUDA device-side assert
# (`IndexKernel.cu:111 index out of bounds`). The architecture IS supported; the fault is a
# mismatch between the image-placeholder count VAGEN puts in `prompt_token_ids` and what
# the engine expects. Leading hypothesis, unverified: InternVL's dynamic tiling
# (`max_dynamic_patch`) makes the tile count differ between VAGEN's processor call and
# vLLM's. Kept because the diagnosis is worth more than the deletion. See docs/issues.md.
set -eo pipefail

V=$(cd "$(dirname "$0")/../../.." && pwd)
SCRIPTDIR=$(cd "$(dirname "$0")" && pwd)
PROJECT_NAME=${PROJECT_NAME:-vagen_experiments}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-sokoban_default_gae_internvl3_2b}
EXPERIMENT_DIR=${EXPERIMENT_DIR:-$V/exps/$PROJECT_NAME/$EXPERIMENT_NAME}
MODEL=${MODEL:-OpenGVLab/InternVL3-2B-hf}
mkdir -p "$EXPERIMENT_DIR"

# verl is not imported as an installed package; it is a checkout, and it has to come
# first on PYTHONPATH so this fork wins over any other copy.
# Both layouts: the submodule at VAGEN/verl that the README creates, and a sibling
# checkout next to VAGEN. Probed for a file rather than the directory -- an uninitialised
# submodule leaves VAGEN/verl there but empty. Left unresolved this used to go on with
# VERL empty, which made hydra.searchpath "file:///verl/trainer/config" and failed later
# on something that does not mention verl.
# ★ A plain loop, not `VERL=${VERL:-$(...)}`. Under `set -e` a command substitution that
# exits non-zero kills the shell AT THE ASSIGNMENT, so the diagnostic below never ran: a
# clone without --recursive got exit 1 and no output at all, which is precisely the case
# the diagnostic exists for.
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

PYTHONUNBUFFERED=1 python3 -m vagen.main_ppo \
    --config-path="$V/vagen/configs" --config-name=vagen_multiturn \
    hydra.searchpath="[file://$VERL/verl/trainer/config]" \
    data.custom_cls.path="$V/vagen/gym_agent_dataset.py" \
    "${BASE[@]}" \
    data.train_files="$SCRIPTDIR/train_sokoban_vision_free_wm.yaml" \
    data.val_files="$SCRIPTDIR/val_sokoban_vision_free_wm.yaml" \
    actor_rollout_ref.model.path="$MODEL" \
    critic.model.path="$MODEL" \
    critic.enable=True \
    algorithm.adv_estimator=default_gae \
    trainer.harness=concat \
    data.train_batch_size=128 \
    data.max_response_length=4000 \
    actor_rollout_ref.actor.ppo_mini_batch_size=32 \
    actor_rollout_ref.rollout.n=1 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.max_num_batched_tokens=10000 \
    actor_rollout_ref.rollout.enforce_eager=True \
    actor_rollout_ref.rollout.free_cache_engine=True \
    actor_rollout_ref.rollout.enable_chunked_prefill=True \
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
