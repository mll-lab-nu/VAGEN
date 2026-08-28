#!/bin/bash
# sokoban - default_gae - concat - Qwen/Qwen3.5-4B, with its NATIVE THINKING CHANNEL ON.
#
# The sibling script `train_default_gae_qwen35_4b.sh` runs the same model with thinking
# off, under `wm_think`. That arm is the baseline this one is measured against, and it is a
# strong one: it converges to 0.97 success at ~250 tokens a turn. Read any result here
# against that, not against zero.
#
# Three things have to line up, and each fails silently on its own:
#
# 1. `enable_thinking=True`. The default (vagen_multiturn.yaml) is False, which makes
#    Qwen3.5's template emit a pre-closed EMPTY `<think></think>` into the prompt -- the
#    model then writes plain prose and there is no reasoning under response_mask to train.
#    True leaves the block OPEN at the end of the generation prompt, so the reasoning is
#    generated as part of the response.
#
# 2. `prompt_format: free_think` (in the yaml). This experiment intentionally trains the
#    compact think/answer protocol instead of full WM. free_think asks for
#    `</think>` and then `<answer>`, and its opening tag is optional precisely because the
#    template already wrote one.
#
# 3. `reasoning_config` below. It tells the engine where a reasoning block starts and ends
#    so that the yaml's `thinking_token_budget` can be enforced by forcing the closing
#    token. vLLM REFUSES every request if the budget is set and this is not -- loudly, at
#    least. This is per-family knowledge, which is why it lives here next to MODEL rather
#    than in VAGEN: another family spells its delimiters differently.
#
#    ★ The nested quoting is load-bearing. Hydra's override lexer rejects a bare `<think>`
#    with LexerNoViableAltException, so the single quotes have to survive the shell and
#    arrive at hydra intact.
#
# Budget: 5 x 2048 + 4 x 96(observation) = 10624 against max_response_length=11264; the
# system prompt (370) and first observation (80) sit in the 1000-token prompt region.
set -eo pipefail

V=$(cd "$(dirname "$0")/../../.." && pwd)
SCRIPTDIR=$(cd "$(dirname "$0")" && pwd)
PROJECT_NAME=${PROJECT_NAME:-vagen_experiments}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-sokoban_default_gae_qwen35_4b_think}
EXPERIMENT_DIR=${EXPERIMENT_DIR:-$V/exps/$PROJECT_NAME/$EXPERIMENT_NAME}
MODEL=${MODEL:-Qwen/Qwen3.5-4B}
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

PYTHONUNBUFFERED=1 python3 -m vagen.training.main \
    --config-path="$V/vagen/configs" --config-name=vagen_multiturn \
    hydra.searchpath="[file://$VERL/verl/trainer/config]" \
    data.custom_cls.path="$V/vagen/training/dataset.py" \
    "${BASE[@]}" \
    data.train_files="$SCRIPTDIR/train_sokoban_vision_free_think.yaml" \
    data.val_files="$SCRIPTDIR/val_sokoban_vision_free_think.yaml" \
    data.apply_chat_template_kwargs.enable_thinking=True \
    +actor_rollout_ref.rollout.engine_kwargs.vllm.reasoning_config.reasoning_start_str="'<think>'" \
    +actor_rollout_ref.rollout.engine_kwargs.vllm.reasoning_config.reasoning_end_str="'</think>'" \
    actor_rollout_ref.model.path="$MODEL" \
    critic.model.path="$MODEL" \
    critic.enable=True \
    algorithm.adv_estimator=default_gae \
    trainer.harness=concat \
    data.train_batch_size=128 \
    data.max_response_length=11264 \
    actor_rollout_ref.actor.ppo_mini_batch_size=32 \
    actor_rollout_ref.rollout.n=1 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.max_num_batched_tokens=13312 \
    actor_rollout_ref.rollout.max_model_len=13312 \
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
