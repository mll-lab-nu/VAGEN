#!/bin/bash
# spatial_gym - bi_level_gae - concat - Qwen/Qwen2.5-VL-3B-Instruct
#
# Per-experiment settings only. Everything that makes a VAGEN run work at all lives in
# vagen/configs/baseline_vllm.flags and is read below -- in particular the two flags that
# select VAGEN's agent loop. Without them verl runs its own, and the job comes up looking
# healthy while none of this repo's rollout code executes. All twenty of these scripts
# were in that state, and the duplication is why: each carried its own copy of the stack
# settings, and the copies stopped including the loop.
#
# Anything after "${BASE[@]}" overrides it, and anything on the command line overrides
# that, so a one-off sweep needs no edit here.
# ★ The response region has to hold the WHOLE episode, because this reward is
# terminal-only. concat keeps all 11 turns in one conversation, and SpatialGym pays out
# only on the final cogmap turn (spatial_gym_env.py: `awaiting_cogmap_output`), which is
# reachable only after max_exp_steps=10 exploration turns. Worst case:
#
#     11 x response_length_per_turn(1024) + 10 x max_env_response_per_turn(700) = 18264
#
# At the old 2000, `exhausted()` (floor = min(g, n_r/4) = 500) ended every episode around
# turn 3 of 11 -- so the only scored turn was never reached and these runs could not score
# at all, while every other metric looked ordinary. Sized for the worst case deliberately:
# exploration turns really use ~150 tokens, but a budget that merely usually fits would
# cut the one turn that matters on the episodes that reason longest.
set -eo pipefail

V=$(cd "$(dirname "$0")/../../.." && pwd)
SCRIPTDIR=$(cd "$(dirname "$0")" && pwd)
PROJECT_NAME=${PROJECT_NAME:-vagen_spatial_gym}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-spatial_gym_bi_level_gae_qwen25vl3b}
EXPERIMENT_DIR=${EXPERIMENT_DIR:-$V/exps/$PROJECT_NAME/$EXPERIMENT_NAME}
MODEL=${MODEL:-Qwen/Qwen2.5-VL-3B-Instruct}
mkdir -p "$EXPERIMENT_DIR"

# verl is not imported as an installed package; it is a checkout, and it has to come
# first on PYTHONPATH so this fork wins over any other copy.
# Both layouts: the submodule at VAGEN/verl that the README creates, and a sibling
# checkout next to VAGEN. Probed for a file rather than the directory -- an uninitialised
# submodule leaves VAGEN/verl there but empty. Left unresolved this used to go on with
# VERL empty, which made hydra.searchpath "file:///verl/trainer/config" and failed later
# on something that does not mention verl.
VERL=${VERL:-$(for d in "$V/verl" "$V/../verl"; do
    [ -f "$d/verl/trainer/config/ppo_trainer.yaml" ] && (cd "$d" && pwd) && break
done)}
if [ -z "$VERL" ]; then
    echo "verl not found at $V/verl or $V/../verl." >&2
    echo "Run: git submodule update --init --recursive   (or set VERL=/path/to/verl)" >&2
    exit 1
fi
export PYTHONPATH=${VERL:+$VERL:}$V${PYTHONPATH:+:$PYTHONPATH}
mapfile -t BASE < <(grep -vE '^\s*(#|$)' "$V/vagen/configs/baseline_vllm.flags" | sed "s|\$V|$V|g")

# train_batch_size is 32, not 128: the dataset is n_envs=50 and the loader drops
# the last partial batch, so 128 yields zero batches and the trainer asserts.
PYTHONUNBUFFERED=1 python3 -m vagen.main_ppo \
    --config-path="$V/vagen/configs" --config-name=vagen_multiturn \
    hydra.searchpath="[file://$VERL/verl/trainer/config]" \
    data.custom_cls.path="$V/vagen/gym_agent_dataset.py" \
    "${BASE[@]}" \
    data.train_files="$SCRIPTDIR/train_spatial_gym_vision.yaml" \
    data.val_files="$SCRIPTDIR/val_spatial_gym_vision.yaml" \
    actor_rollout_ref.model.path="$MODEL" \
    critic.model.path="$MODEL" \
    critic.enable=True \
    algorithm.adv_estimator=bi_level_gae \
    +algorithm.high_level_gamma=0.9 \
    trainer.harness=concat \
    data.train_batch_size=32 \
    data.max_prompt_length=4000 \
    data.max_response_length=18432 \
    actor_rollout_ref.rollout.max_model_len=22528 \
    actor_rollout_ref.actor.ppo_mini_batch_size=32 \
    actor_rollout_ref.rollout.n=1 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.max_num_batched_tokens=10000 \
    actor_rollout_ref.rollout.enforce_eager=True \
    actor_rollout_ref.rollout.free_cache_engine=True \
    actor_rollout_ref.rollout.enable_chunked_prefill=True \
    trainer.n_gpus_per_node=8 \
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
