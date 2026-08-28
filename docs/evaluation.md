# Evaluation

Evaluation runs the **same harness-owned episode loop as training**, against any supported
backend. `vagen/rollout/runner.py` only wires lifecycle and reward recording. So an eval number and a
training `val-core` number are comparable — provided the configs agree, which is what
`tests/test_eval_matches_val.py` checks for the shipped pairs.

## Evaluating a model you did not train

```bash
MODEL_PATH=Qwen/Qwen2.5-VL-3B-Instruct \
  bash examples/evaluate/sokoban/vllm/eval_qwen25_vl_3b.sh
```

That starts a vLLM server, runs the config next to it, writes rollouts and a
`summary.json`, and shuts the server down. There is one per environment —
`examples/evaluate/<env>/vllm/` for sokoban, frozenlake, navigation, primitive_skill and
spatial_gym; the last three also need an environment server or a dataset, and each
launcher checks for its own and says what to run.

Any extra argument is a hydra override:

```bash
bash examples/evaluate/sokoban/vllm/eval_qwen25_vl_3b.sh 'envs.0.harness=no_concat'
```

For a native-thinking model such as Qwen3.5, use `wm_think`, configure the server's
reasoning delimiters, and bound thinking so the model still has room for the canonical WM
suffix and final answer:

```bash
VLLM_REASONING_CONFIG='{"reasoning_start_str":"<think>","reasoning_end_str":"</think>"}' \
  MODEL_PATH=Qwen/Qwen3.5-4B \
  bash examples/evaluate/sokoban/vllm/eval_qwen25_vl_3b.sh \
  envs.0.config.prompt_format=wm_think \
  envs.0.chat_config.extra_body.chat_template_kwargs.enable_thinking=true \
  envs.0.chat_config.extra_body.thinking_token_budget=128
```

Without `--reasoning-config`, vLLM rejects `thinking_token_budget`. Without a thinking
budget, a model may spend the complete per-turn response allowance before emitting
`<perception>` or `<answer>`.

## Evaluating a checkpoint you trained

Training writes to `trainer.default_local_dir`, which every shipped script sets to
`$V/exps/$PROJECT_NAME/$EXPERIMENT_NAME/verl_checkpoints`. What lands there depends on
`actor_rollout_ref.actor.checkpoint.save_contents`; the scripts ship
`['model','hf_model','optimizer','extra']`, and it is **`hf_model`** you want — a plain
HuggingFace directory that vLLM can serve directly.

```bash
CKPT=exps/vagen_experiments/sokoban_default_gae_qwen25vl3b/verl_checkpoints
ls $CKPT                                   # global_step_100/ global_step_200/ ...
MODEL_PATH=$CKPT/global_step_200/actor/huggingface \
  bash examples/evaluate/sokoban/vllm/eval_qwen25_vl_3b.sh
```

Without `hf_model` in `save_contents` you get sharded FSDP state instead, and it has to be
merged first — `python -m verl.model_merger` in the submodule.

With `trainer.save_best_actor: true` (the default) the actor from the best-validating step
is kept separately under `best_actor/`, selected on the mean of
`val-core/<env>/reward/mean@1` across environments.

!!! warning "Put the model in the dump directory"
    Rollouts are keyed on `(env, seed, tag_id, model)` and `resume: skip_completed` skips
    on a match. Two checkpoints evaluated into one dump directory are kept apart by the
    model field, but the *summary* is per-directory, so two checkpoints sharing one would
    have the second overwrite the first's `summary.json`. Every launcher puts
    `$MODEL_NAME` in the path for that reason, and derives it with `vagen_model_name`
    (`examples/evaluate/common.sh`) rather than `basename` — every verl checkpoint's
    basename is the literal string `huggingface`, `best_actor/` included.

## Comparing context policies

The policy is a config key, the same one training uses, so an ablation is one override:

```bash
for H in concat no_concat compact; do
  bash examples/evaluate/sokoban/vllm/eval_qwen25_vl_3b.sh "envs.0.harness=$H"
done
```

`compact` additionally needs `compact_budget` or `max_response_length`; with neither, no
trigger can fire and it would silently be `concat`, so it is refused. Any `BaseHarness`
subclass works here too, by registered name or as an import path — see
[Configuration](configuration.md).

## What comes out

```text
<fileroot>/rollouts/<experiment>/tag_<tag_id>/
├── summary.json                 # success_rate, avg_cumulative_reward, avg_turns
└── <timestamp>-<uuid8>/
    ├── metrics.json             # one episode: success, reward, finish_reason, model
    ├── messages.json            # every conversation the harness opened
    ├── assistant_texts.json
    ├── transcript.txt
    └── images/turn_01_01.png    # 1-indexed; turn 01 is the reset observation
```

`finish_reason` is one of `done`, `max_turns`, `no_room` (the conversation ran out of
response region), `env_error`, `setup_error`, `empty_generation` (the backend returned no
text), or `error`. Only the first three count as episodes; the rest are listed under
`error_rollouts` and are deleted by the next resumed run.

If *every* episode ends in one of the failure reasons, the run stops with a non-zero exit
rather than writing `success_rate: 0.0` — an evaluation where nothing ran is not a score of
zero, and it used to be reported as one.

See [`vagen/evaluation/README.md`](https://github.com/mll-lab-nu/VAGEN/blob/main/vagen/evaluation/README.md)
for the full config reference.
