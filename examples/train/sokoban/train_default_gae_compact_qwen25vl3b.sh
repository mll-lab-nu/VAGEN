#!/bin/bash
# sokoban - default_gae - compact - Qwen/Qwen2.5-VL-3B-Instruct
#
# harness and adv_estimator are orthogonal -- the harness lays an episode out in rows,
# the estimator stitches them back -- so any trajectory estimator can be swapped in from
# the command line.
set -eo pipefail

V=$(cd "$(dirname "$0")/../../.." && pwd)
SCRIPTDIR=$(cd "$(dirname "$0")" && pwd)
PROJECT_NAME=${PROJECT_NAME:-vagen_experiments}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-sokoban_compact_qwen25vl3b}
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

# Sized for a compaction every ~5 turns. Sokoban vision, measured (budget.py header):
# system prompt 589, observation ~58, a real turn ~164, summarise request 15.
#
# A conversation opened on a summary holds 589 + k + 222n after n turns and compacts when
# the next would not fit, so n=5 needs 589 + k + 1110 <= m < 589 + k + 1332.
#
# k is set rather than derived: derived it is min(g, m//4) = 512, which pushes m to ~2300
# and lets the first conversation -- no summary to carry -- run 7 turns against 5 for
# every one after it. At k=300, m in [1999, 2221] and the first gets 6.
#
# ★ 1200, not 2100, and it is tied to max_turns=5 in the dataset yaml. The arithmetic
# above is per conversation, so a budget only compacts if a whole episode does NOT fit
# inside one: at m=2100 a 5-turn episode fits, compaction never fires, and this arm
# silently measures concat -- the same numbers as the default_gae script, under a name
# that says otherwise. 1200 buys about three turns and then two, so a 5-turn episode
# compacts once.
#
# Measured on the cluster at this setting: 128 episodes produced 178 rows, i.e. ~0.4
# compactions per episode -- below one because sokoban episodes often end before turn 5.
# Raise max_turns and this has to come down with it, or the arm degenerates again.
COMPACT_BUDGET=${COMPACT_BUDGET:-1200}
COMPACT_SUMMARY_BUDGET=${COMPACT_SUMMARY_BUDGET:-300}

PYTHONUNBUFFERED=1 python3 -m vagen.training.main \
    --config-path="$V/vagen/configs" --config-name=vagen_multiturn \
    hydra.searchpath="[file://$VERL/verl/trainer/config]" \
    data.custom_cls.path="$V/vagen/training/dataset.py" \
    "${BASE[@]}" \
    data.train_files="$SCRIPTDIR/train_sokoban_vision.yaml" \
    data.val_files="$SCRIPTDIR/val_sokoban_vision.yaml" \
    actor_rollout_ref.model.path="$MODEL" \
    critic.model.path="$MODEL" \
    critic.enable=True \
    algorithm.adv_estimator=default_gae \
    trainer.harness=compact \
    trainer.compact_budget=$COMPACT_BUDGET \
    trainer.compact_summary_budget=$COMPACT_SUMMARY_BUDGET \
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
