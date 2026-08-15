# Configuration

A run is assembled from three places, and knowing which is which saves most of the
confusion:

| | what it sets | where |
|---|---|---|
| **The training script** | model, GPUs, batch sizes, budgets — as hydra overrides | `examples/train/<env>/*.sh` |
| **The dataset yaml** | which environments, how many, how many turns, per-turn budgets | `data.train_files` / `data.val_files` |
| **The base config** | everything else, with defaults | [`vagen/configs/vagen_multiturn.yaml`](../vagen/configs/vagen_multiturn.yaml) |

!!! warning "`baseline_vllm.flags` overrides the base config"
    Every shipped script sources `vagen/configs/baseline_vllm.flags`, and it wins over
    `vagen_multiturn.yaml`. Two that matter: `data.max_prompt_length` is **1000** in the
    flags file and 9000 in the yaml, and `actor_rollout_ref.rollout.name` is **vllm** in
    the flags file and `sglang` in the yaml. A model launcher can override the flags again
    later (InternVL uses 1600 prompt tokens and GLM uses 1400), so the final command-line
    occurrence wins.

---

## Context policy — `trainer.harness`

The central choice. An episode is many turns; a training row is one conversation. The
harness decides how the first maps onto the second.

| `harness` | one episode becomes | use when |
|---|---|---|
| `concat` | **one** row holding every turn | the default; the episode fits the response region |
| `no_concat` | **one row per turn**, each a fresh conversation | history is not needed, or will not fit |
| `compact` | a row per conversation, summarised and reopened when full | long episodes that must keep context |

`compact` is closely related to **CompactionRL** ([arXiv:2607.05378](https://arxiv.org/abs/2607.05378),
Li et al., 2026), which trains task execution and summary generation jointly under context
compaction. Here the summary is produced by the policy and trained with everything else, so
the same comparison applies.

```yaml
trainer:
  harness: compact
  compact_budget: 1200          # compact only: the conversation size that triggers a summary.
                                # ★ Size it against max_turns -- see the tip below. The
                                # shipped default is 4000, which at sokoban's max_turns: 5
                                # holds a whole episode and so never fires. 1200 is what
                                # train_default_gae_compact_qwen25vl3b.sh uses.
  compact_summary_budget: null  # null -> max(1, min(response_length_per_turn, compact_budget // 4))
```

!!! danger "`harness` and `algorithm.adv_estimator` are one choice"
    `no_concat` and `compact` split an episode across rows, so a per-row estimator scores a
    fraction of an episode as though it were the whole thing. The trainer **refuses** the
    combination at startup rather than training on it:

    ```
    ValueError: algorithm.adv_estimator=... scores one row at a time, but
    trainer.harness=... splits an episode across rows
    ```

    Use a trajectory estimator with those two — see the table below.

!!! tip "`compact_budget` has to be sized against `max_turns`"
    Compaction fires only if a whole episode does **not** fit in one conversation. At
    sokoban's `max_turns: 5` a budget of 2100 holds the entire episode, nothing is ever
    summarised, and the arm silently reports `concat`'s numbers under another name. 1200
    buys about three turns and then two.

### Adding your own

`harness` is not a closed set. Either register a `BaseHarness` subclass, or give an import
path and register nothing:

```python
from vagen.core.harness import BaseHarness
from vagen.harness import register_harness

@register_harness("mine")
class MyHarness(BaseHarness): ...
```

```yaml
# training
trainer:
  harness: mine    # a registered name; needs the module imported in every worker, below
```

```yaml
# evaluation -- examples/evaluate/<env>/config.yaml
envs:
  - name: Sokoban
    harness: mypkg.harnesses:MyHarness   # an import path
```

(Only one `harness` key per block. Writing it twice under the same `trainer:` is a duplicate
mapping key, and OmegaConf raises an error.)

In training, verl builds a separate registry in each worker process, so your module has to
be imported inside every one of them. That is what `actor_rollout_ref.model.external_lib`
does.

★ In evaluation, use the import path. `run_eval` never imports anything on your behalf —
there is no `external_lib` setting there — so a registered name on its own fails with:

```
unknown harness 'mine'; choose from ['compact', 'concat', 'no_concat']
```

---

## Advantage estimator — `algorithm.adv_estimator`

VAGEN overrides verl's default `gae` with `default_gae`. verl's own estimators open every
row with `nextvalues=0`, which is right only when the row ends in a true terminal state and
wrong when it is merely one turn of a longer trajectory.

| estimator | scores | extra parameter |
|---|---|---|
| `default_gae` | the trajectory, rows stitched back together — **the default** | — |
| `bi_level_gae` | turn-level outer chain, token-level inner — the published VAGEN algorithm | `+algorithm.high_level_gamma=0.9` in shipped experiments |
| `turn_level_gae` | per turn | — |
| `token_level_gae` | per token, across the trajectory | — |
| `trajectory_grpo` | group-relative, over trajectories | — |
| verl's `gae`, `grpo`, … | one row — **`concat` only** | — |

`high_level_gamma` does not appear in the base schema, so it goes on the command line. The
code can fall back to `algorithm.gamma`, but that is not the published/shipped experiment:
when the two clocks use the same default, bi-level GAE can degenerate toward token-level
GAE. Set it explicitly for reproduction.

```bash
algorithm.adv_estimator=bi_level_gae +algorithm.high_level_gamma=0.9
```

!!! danger "One more startup refusal"
    All but `trajectory_grpo` need a critic and refuse without `critic.enable=True` — a
    `ValueError` at startup, the same class of trap as the harness rule above.

    An estimator may also declare `undiscounted=True` at registration, which refuses the
    run unless `algorithm.gamma=1.0`. None of the shipped estimators does; it is there for
    a custom one that mixes a per-token and a per-turn clock, where the two only agree at
    1.0 and the disagreement otherwise looks like noise rather than a bug.

!!! warning "Timeout truncation is not yet value-bootstrapped"
    `TurnLimit` distinguishes an environment termination from a `max_turns` truncation,
    but the final observation value is not yet carried into the training batch. Both GAE
    implementations therefore currently start their backward recursion from zero at either
    kind of episode boundary. Do not describe a time-limit cutoff as theoretically terminal
    when interpreting returns; proper truncation bootstrapping remains follow-up work.

---

## The dataset yaml

One entry per environment family. `examples/train/sokoban/train_sokoban_vision.yaml` is the
worked example.

```yaml
envs:
  - name: Sokoban          # a key in vagen/configs/env_registry.yaml
    n_envs: 10000          # how many instances to materialise
    data_source: sokoban   # a label, for logging
    seed: [1, 10000, 1]
    max_turns: 5
    response_length_per_turn: 512
    max_env_response_per_turn: 256
    config: {}             # environment settings; shared wrappers are handled first
```

| key | meaning |
|---|---|
| `name` | registered environment name |
| `tag_id` | **required in an eval config.** Names the results directory (`tag_<id>`) and is part of the resume key, so two env entries in one run must not share one |
| `n_envs` | number of instances |
| `data_source` | label only |
| `seed` | see **Seeds** |
| `seed_list` | explicit seeds, at least `n_envs` of them; overrides `seed` |
| `max_turns` | environment steps per episode. A **cap**, not an expectation |
| `response_length_per_turn` | hard cap on one generation — it becomes `max_tokens` |
| `max_env_response_per_turn` | ceiling on one observation; over it the text is cut. Default 2048 |
| `thinking_token_budget` | tokens allowed *inside* a reasoning block — see **Budgets** |
| `env_response_length` | deprecated spelling of `max_env_response_per_turn`; setting both raises |
| `config` | environment-specific settings. `state_reward` is removed to build its shared wrapper; remaining keys initialize the environment config dataclass |

### Seeds

One of these three forms (not a block to paste — each line is a whole alternative):

```text
seed: [7]                # 1 element: a base seed; actual seeds are sampled from [0, 2**31-1]
seed: [0, 99]            # 2 elements: sampled from the INCLUSIVE range, with repeats
seed: [0, 99, 1]         # 3 elements: as above, each value used at most `limit` times
```

!!! warning "The third element is an occurrence limit, not a step, and the range is inclusive"
    With `limit: 1` the loader draws `n_envs` values from `range(min, max+1)` **without**
    replacement. So `n_envs` must equal `max - min + 1`, or one value is dropped — the
    *same* one on every run at a fixed `data.base_seed`, since the RNG is seeded from it — and where an environment indexes its dataset as
    `seed % len(dataset)`, the surplus wraps onto item 0 and scores it twice.
    `tests/test_eval_matches_val.py` checks this for the shipped eval configs.

Training and evaluation derive seeds identically, so one directive gives the same seeds on
both sides. The global offset is `data.base_seed`, which is **not** in verl's data schema
and so needs the `+` prefix on the command line:

```bash
+data.base_seed=1234
```

Without it hydra rejects the key; written as `data.base_seed=` it is silently ignored and
stays 0.

### Budgets

Two independent caps that do different things:

- **`response_length_per_turn`** becomes `max_tokens`. A guillotine: the model is never told
  about it, so it plans as if unbounded and is cut wherever it happens to be.
- **`thinking_token_budget`** makes the engine *force the closing tag* when a reasoning
  block runs long, so the turn is bounded rather than truncated and still produces an
  answer.

The second needs the engine to know where a reasoning block begins and ends. That is
per-model, so it lives in the training script rather than here:

```bash
+actor_rollout_ref.rollout.engine_kwargs.vllm.reasoning_config.reasoning_start_str="'<think>'" \
+actor_rollout_ref.rollout.engine_kwargs.vllm.reasoning_config.reasoning_end_str="'</think>'"
```

vLLM refuses the request if the budget is set and these are not. The nested quoting is
required: hydra's override lexer rejects a bare `<think>`.

!!! note "A thinking budget bounds the block, not the response"
    Measured on Qwen3.5-4B, `thinking_token_budget: 512` closes the block at exactly 511
    tokens — but the model often carries on in the visible content. It buys a well-formed,
    scoreable turn; it does not buy brevity.

---

## Per-environment `config:`

Most keys are passed to the environment's config dataclass, so they differ per environment
— **and so do the defaults for keys they share.** `state_reward` is the shared exception:
VAGEN removes it first and builds `StateRewardWrapper`; `max_turns` lives outside this block
and builds `TurnLimit`. Sokoban
(`vagen/envs/sokoban/sokoban_env.py`):

| key | default | notes |
|---|---|---|
| `render_mode` | `text` | `text` or `vision` |
| `prompt_format` | **`wm`** | see below |
| `format_reward` | **`0.1`** | ordinary shipped yamls set `0.02`; the state-reward yamls set `0.03` |
| `success_reward` | `1.0` | |
| `strict_format` | `true` | a malformed turn has its actions discarded |
| `use_example_in_sys_prompt` | `true` | |
| `max_actions_per_step`, `action_sep`, `dim_room`, `num_boxes`, `max_steps`, `min_solution_steps` | | |

FrozenLake's defaults are **not** the same — `prompt_format` defaults to `free_think` and
`format_reward` to `0.02`. Read the dataclass for the environment you are configuring.

### `prompt_format`

| format | shape | available on |
|---|---|---|
| `wm` | `<observation><think><answer><prediction>` | sokoban, frozenlake, primitive_skill |
| `free_think` | free reasoning closed by `</think>`, then an accepted action marker | sokoban, frozenlake, primitive_skill |
| `free_wm` | observation / answer / prediction, free prose between | **sokoban only** |
| `answer` | `<answer>` only | **sokoban only** |
| `wm`, `free_think`, `no_think`, `eval_mode` | navigation's own set, and it tags the action `<action>`, not `<answer>` | **navigation only** |

`SpatialGym` has no `prompt_format`: the field is `init=False` and it parses `THINK:` /
`FINAL ANSWER:` labels with a whole-text fallback.

!!! danger "Thinking delimiters are model-family specific"
    Qwen2.5-VL and InternVL3/3.5 can emit a literal `<think>` string. On **Qwen3-VL,
    Qwen3.5 and GLM**, thinking delimiters are reserved control tokens and the chat template
    may open the reasoning channel before generation. A strict `wm` prompt that demands a
    newly generated literal `<think>` can therefore score zero even when actions are sound.

    `free_think` permits either opening convention, but still requires reasoning to close
    with `</think>` before the action. For GLM, the parser also accepts its native
    `<|begin_of_box|>…<|end_of_box|>` action marker. This is parser equivalence only: VAGEN
    does not replace or rewrite the sampled text, token IDs, or rollout logprobs into
    `<answer>`. Note that `primitive_skill` *defaults* to `wm`.

### Model-family compatibility

The shared vLLM flags enable fused kernels, but the experimental model launchers need
family-specific exceptions:

- **InternVL3.5:** set `actor_rollout_ref.model.use_fused_kernels=False`; the generic fused
  path is text-only for an unknown VLM. Also keep
  `engine_kwargs.vllm.hf_overrides.tie_word_embeddings=false`, so vLLM loads the checkpoint's
  separate language-model head.
- **GLM-4.6V-Flash:** keep native thinking enabled, raw rollout logprobs enabled, and the
  GPT-J-style multimodal RoPE compatibility hook used by its launcher.

The shipped launchers still select vLLM. Installing the SGLang extra does not switch their
training backend. Before a long run, use the exact launcher and append
`trainer.val_only=true trainer.save_freq=-1 trainer.test_freq=-1` to validate prompt,
parser, vision inputs and harness together.

---

## State reward

An optional extra reward for what the agent *says* about the world, alongside what it does.
A judge model reads the `<observation>` and `<prediction>` sections, turns them into
structured relations, and compares those with the environment's real state. Off by default.

★ It is configured under **`envs[].config.state_reward`**, not under `trainer`. Training and
evaluation both construct the environment from that block, so the same reward definition
applies in both paths. The old `trainer.state_reward` location is deleted, with no fallback.

There are two halves to it:

1. **The per-environment settings**, under `envs[].config.state_reward`: which sections to
   score, the absolute reward for one perfect section on one turn, and the judge endpoint.
2. **The environment's own spec**, a `STATE_REWARD_SPEC` attribute on the environment class.
   This is what reads the true state and phrases the question for the judge, so it has to be
   written per environment. Sokoban's is
   [`vagen/envs/sokoban/state_reward_spec.py`](../vagen/envs/sokoban/state_reward_spec.py),
   set on the class in `sokoban_env.py`. Turning the reward on for an environment that has
   no spec raises an error rather than quietly scoring zero.

Sokoban already ships a spec. A complete working example is
[`examples/train/sokoban/train_bi_level_gae_sr_qwen25vl3b.sh`](../examples/train/sokoban/train_bi_level_gae_sr_qwen25vl3b.sh)
— the `sr` in the name. It starts the judge, waits for `/health`, and reaps it on exit; its
train/validation yaml files contain:

```yaml
envs:
  - name: Sokoban
    config:
      state_reward:
        state_estimation:      {enable: true, reward: 0.03}
        transition_prediction: {enable: true, reward: 0.03}
        score_base: 0.334
        judge_base_url: ${oc.env:JUDGE_BASE_URL,http://127.0.0.1:8123/v1}
        judge_model: ${oc.env:JUDGE_MODEL,Qwen/Qwen3-4B-Instruct-2507}
```

Before spending GPU time on it, check the judge can actually do the conversion on your
environment: `JUDGE_URL=http://127.0.0.1:8123/v1 python tools/judge_eval.py` scores it
against hand-labelled cases. A judge that reads the descriptions wrong pays a reward
signal that looks like learning.

**`state_estimation`** scores the `<observation>`, i.e. the state the agent acted *from*.
**`transition_prediction`** scores the `<prediction>`, the state it acted *into*. They switch
on independently, and whichever are on decide the response format the agent is asked for —
there is no separate prompt setting to keep in step.

**`reward` is absolute and per turn.** The number in the yaml is exactly what a perfect
description of that section pays on one turn; it is not divided by `max_turns` and there is
no `weight` or episode `budget`. In the example above, both judged sections over five
perfect turns pay `2 × 0.03 × 5 = 0.30`; format pays up to `0.03 × 5 = 0.15`. The shaping
cap is therefore `0.45`, and including Sokoban's `1.0` success reward the episode cap is
`1.45`. Raising `max_turns` changes that maximum explicitly rather than silently shrinking
every turn's learning signal.

**`score_base`** is subtracted from each description's F1 before it is paid, then the rest is
rescaled so a perfect description still earns the configured per-turn reward. It exists because scoring
about a third is free: naming any relation at all gets you there. Measured over 300 real
Sokoban starts, uniform random scores 0.334 and the best constant answer — "same, same",
which looks at nothing — scores 0.391. Set `0.0` to restore the old behaviour.

There is no state-reward `format_reward`. A turn that omits any enabled section earns no
state reward for that turn; writing the sections is the gate that makes them scoreable, not
a separate line item. The environment's own `format_reward` remains the single format knob.

There is also no `placement`. The environment pays each score on the final token of the span
that earned it. An estimator that requires one reward slot per turn performs that reduction
itself; `bi_level_gae` sums each turn onto its boundary before its outer recursion. This keeps
the environment independent of the estimator and preserves the richer per-span signal for
estimators that can use it.

---

## Logging and checkpoints

```yaml
trainer:
  val_before_train: true
  log_val_generations: 32
  val_log_select: balanced        # balanced | first | failures | successes | worst | best
  save_best_actor: true
  save_freq: 100                  # ★ NOT the default. verl's is -1, i.e. never save
  max_actor_ckpt_to_keep: 1
  max_critic_ckpt_to_keep: 1
  rollout_data_dir: ...           # per-step rollout dumps, as jsonl
  validation_data_dir: ...        # the same for validation. Every shipped script sets it
  replace_image_tokens_for_logging: true
  log_image: {enable: false, max_pending: 2, png_compress_level: 0}
```

!!! danger "`save_freq` has no useful default"
    verl's default is `-1` — never checkpoint. Every shipped script sets it by hand; a
    script written from this block without it trains for days and saves nothing.

!!! tip "Set `save_freq` against your wall clock"
    A requeued job resumes from the last checkpoint (`resume_mode: auto`), so `save_freq` is
    how much work a preemption costs. At 15 minutes a step against a 48-hour limit,
    `save_freq: 100` throws away up to 25 hours of it; `25` costs at most 6.

---

## Filters

```yaml
filter:
  name: reward_variance_top_p
  filter_kwargs: {top_p: 0.9}
  enable: false
```

`filter_kwargs` is splatted into the registered function, so the accepted keys are that
function's. The two built in take **different** ones:

One of these two (each line is a whole alternative):

```text
filter: {name: reward_variance,       filter_kwargs: {topk: 0.2,  ddof: 0}, enable: true}
filter: {name: reward_variance_top_p, filter_kwargs: {top_p: 0.9, ddof: 0}, enable: true}
```

An unrecognised key is ignored in silence, so `topk` on `reward_variance_top_p` leaves it
at its default `top_p` of 0.9.

See [Custom Filter](custom-filter.md), [Custom Metric](custom-metric.md) and
[Known Issues](issues.md).
