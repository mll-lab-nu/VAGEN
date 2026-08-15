# Configuration

A run is assembled from three places, and knowing which is which saves most of the
confusion:

| | what it sets | where |
|---|---|---|
| **The training script** | model, GPUs, batch sizes, budgets — as hydra overrides | `examples/train/<env>/*.sh` |
| **The dataset yaml** | which environments, how many, how many turns, per-turn budgets | `data.train_files` / `data.val_files` |
| **The base config** | everything else, with defaults | [`vagen/configs/vagen_multiturn.yaml`](https://github.com/mll-lab-nu/VAGEN/blob/main/vagen/configs/vagen_multiturn.yaml) |

!!! warning "`baseline_vllm.flags` overrides the base config"
    Every shipped script sources `vagen/configs/baseline_vllm.flags`, and it wins over
    `vagen_multiturn.yaml`. Two that matter: `data.max_prompt_length` is **1000** in the
    flags file and 9000 in the yaml, and `actor_rollout_ref.rollout.name` is **vllm** in
    the flags file and `sglang` in the yaml. Read the flags file, not only the yaml.

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

(One key per block: `harness` twice under one `trainer:` is a duplicate mapping key and
OmegaConf raises on it.)

For training, the module has to be imported inside the workers, which is what
`actor_rollout_ref.model.external_lib` is for: verl builds a registry per worker process.

★ Evaluation accepts both spellings but imports nothing, so in practice use the import path
there: `run_eval` has no `external_lib` hook, and a bare registered name fails with
`unknown harness 'mine'; choose from ['compact', 'concat', 'no_concat']`.

---

## Advantage estimator — `algorithm.adv_estimator`

VAGEN overrides verl's default `gae` with `default_gae`. verl's own estimators open every
row with `nextvalues=0`, which is right when a row is an episode and wrong when it is one
turn.

| estimator | scores | extra parameter |
|---|---|---|
| `default_gae` | the trajectory, rows stitched back together — **the default** | — |
| `bi_level_gae` | turn-level outer chain, token-level inner — the published VAGEN algorithm | `+algorithm.high_level_gamma` (optional; defaults to `algorithm.gamma`) |
| `turn_level_gae` | per turn | — |
| `token_level_gae` | per token, across the trajectory | — |
| `trajectory_grpo` | group-relative, over trajectories | — |
| verl's `gae`, `grpo`, … | one row — **`concat` only** | — |

`high_level_gamma` does not appear in a config file, so it goes on the command line, and it
falls back to `algorithm.gamma` with a warning if omitted.

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
    config: {}             # passed to the environment untouched -- see below
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
| `config` | environment-specific, passed straight through |

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

Passed to the environment's config dataclass untouched, so the keys differ per environment
— **and so do the defaults for keys they share.** Sokoban
(`vagen/envs/sokoban/sokoban_env.py`):

| key | default | notes |
|---|---|---|
| `render_mode` | `text` | `text` or `vision` |
| `prompt_format` | **`wm`** | see below |
| `format_reward` | **`0.1`** | the shipped yamls set `0.02`: at `max_turns: 5` the default pays 0.5 for formatting against 1.0 for solving |
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
| `free_think` | `<think>…</think>` then `<answer>` | sokoban, frozenlake, primitive_skill |
| `free_wm` | observation / answer / prediction, free prose between | **sokoban only** |
| `answer` | `<answer>` only | **sokoban only** |
| `wm`, `free_think`, `no_think`, `eval_mode` | navigation's own set, and it tags the action `<action>`, not `<answer>` | **navigation only** |

`SpatialGym` has no `prompt_format`: the field is `init=False` and it parses `THINK:` /
`FINAL ANSWER:` labels with a whole-text fallback.

!!! danger "`<think>` is a reserved token on some model families"
    On Qwen2.5-VL and InternVL3 it is ordinary text. On **Qwen3-VL, Qwen3.5 and GLM** it is
    a single reserved control token that the model will not emit as text — so `wm`, which
    requires it, scores **zero** there while every other metric looks healthy. Use `free_wm`
    or `free_think` on those families. Note that `primitive_skill` *defaults* to `wm`.

    `free_think` is for a model reasoning in its own thinking channel. A non-thinking model
    should stay on `wm`.

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
        state_estimation:      {enable: true, reward: 0.01}
        transition_prediction: {enable: true, reward: 0.01}
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
no `weight` or episode `budget`. In the example above, both sections over five perfect turns
pay `2 × 0.01 × 5 = 0.10`. Raising `max_turns` therefore changes the maximum possible episode
total explicitly rather than silently shrinking every turn's learning signal.

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
