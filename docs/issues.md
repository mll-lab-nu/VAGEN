# Known Issues & Fixes

## 1. B200 / RTX 6000 Pro: Attention backend error

When running on NVIDIA B200 or RTX 6000 Pro GPUs with sglang, the default attention backend may fail. Add the following flags to your training command:

```bash
+actor_rollout_ref.rollout.engine_kwargs.sglang.attention_backend=flashinfer \
+actor_rollout_ref.rollout.engine_kwargs.sglang.mm_attention_backend=triton_attn
```

## 2. Native-thinking models use `wm_think`, not a different WM schema

Every environment now shares the same structured world-model suffix:
`<perception>...</perception><reasoning>...</reasoning><prediction>...</prediction><answer>...</answer>`.
The action is last, so `</answer>` can stop generation without cutting off prediction.

Qwen3-VL, Qwen3.5 and GLM may begin generation inside a chat-template-owned thinking
channel. For Sokoban, select `wm_think`: it accepts either an explicit
`<think>...</think>` prefix or response text that begins inside the native block and emits
only `</think>`, then requires the canonical WM suffix. The historical `free_wm` value is
accepted as a compatibility alias but is no longer the documented name.

Malformed or legacy output can be mined for an action so a rollout can continue when
`strict_format: false`, but it always has `format_correct: false` and receives neither
format reward nor state-reward supervision.

What each environment offers:

| env | formats | default |
|---|---|---|
| sokoban | `wm`, `wm_think`, `free_think`, `answer` | `wm` |
| primitive_skill | `wm`, `free_think` | `wm` |
| frozenlake | `wm`, `free_think` | `free_think` |
| navigation | `wm`, `free_think`, `no_think`, `eval_mode` | `free_think` |

`wm_think` and `answer` exist only for sokoban. SpatialGym does not expose a
`prompt_format` field; `prompt_config.enable_think` selects the shared `free_think` or
answer-only protocol.

## 3. `thinking_token_budget` bounds the think block, not the response — and not the cost

For a model with a native reasoning channel (Qwen3-VL, Qwen3.5, GLM), `thinking_token_budget`
is passed to vLLM, which forces the closing `</think>` once the budget is spent. Measured
against a live `vllm serve Qwen/Qwen3.5-4B` with `--reasoning-config`:

| `thinking_token_budget` | think tokens | finish_reason | total tokens |
|---|---|---|---|
| unset | 889 | length (hit `max_tokens`) | 2048 |
| 512 | **511** | stop | 641 |
| 128 | **127** | stop | 1413 |

The parameter is exact. But vLLM forces the closing token mid-word, and the model does not
treat that as having finished reasoning — it carries on in the same register inside the
*visible* content, often never reaching `<answer>`. **A tighter think budget can produce a
longer response**: 1413 total tokens at budget 128 against 641 at budget 512, because what it
still wanted to say moved past the tag.

So the budget buys a well-formed `</think>` and a scoreable native-thinking prefix, not
brevity. Both `free_think` and `wm_think` tell the model to close native reasoning before
the machine-readable answer; the shipped Qwen3.5 thinking experiment uses `free_think`.

vLLM refuses the request if the budget is set and `reasoning_config` is not; see
`examples/train/sokoban/train_default_gae_qwen35_4b_think.sh` for the delimiters, which are
per-model-family knowledge and so live in the script rather than in VAGEN.

## 4. A `compact_budget` too small fails at runtime, not at startup

Under `trainer.harness=compact`, a budget that cannot hold the system prompt plus a summary
plus one generation closes every conversation after a single turn, raises
`CompactionMakesNoProgress` on every episode, and empties the batch. Measured at
`compact_budget=400` with a 589-token sokoban system prompt.

There is no static check, and there deliberately isn't one: the threshold depends on the
system prompt, which `vagen/harness/budget.py` cannot see, and a version of the check that
guessed refused three configurations that demonstrably run. The runtime error names this as
the likely cause, but the symptom arrives after the allocation is up.

If a compact run empties its batch immediately, suspect this number first. 1200 is what
`train_default_gae_compact_qwen25vl3b.sh` uses at `max_turns: 5`; the shipped default of 4000
holds a whole 5-turn sokoban episode, so nothing is ever summarised and the run is concat
under another name.
