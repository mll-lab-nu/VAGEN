# Known Issues & Fixes

## 1. B200 / RTX 6000 Pro: Attention backend error

When running on NVIDIA B200 or RTX 6000 Pro GPUs with sglang, the default attention backend may fail. Add the following flags to your training command:

```bash
+actor_rollout_ref.rollout.engine_kwargs.sglang.attention_backend=flashinfer \
+actor_rollout_ref.rollout.engine_kwargs.sglang.mm_attention_backend=triton_attn
```

## 2. SGLang worker dies unexpectedly

If SGLang workers die with no clear error message, this is likely caused by a `uvicorn` compatibility issue. Pin `uvicorn` to a version below 0.41:

```bash
pip install "uvicorn<0.41"
```

## 3. Model families that do not train on this stack

Two of the shipped sokoban scripts cannot run as of transformers 5.12.1 / vLLM 0.22. Both
fail *after* the allocation is up rather than at model load, so they cost a cluster job
before saying anything. They are kept, with the diagnosis in their headers, because the
diagnosis is the useful part.

| script | fails with | where it stands |
|---|---|---|
| `train_default_gae_glm41v_9b.sh` | `ImportError: apply_multimodal_rotary_pos_emb` at the first attention forward | transformers removed the symbol from `models.glm4v`; GLM-4V's mrope was refactored to `apply_rotary_pos_emb` + `rotate_half_llm`. `verl/verl/models/transformers/glm4v.py:313` still imports the old name. Porting it is real work and a subtly wrong rope corrupts training **silently** — there is no GLM baseline here to catch that against. **Note** the verl submodule *does* carry a `glm4v.py` patch (`27c51e9`, unpacking the vision tower output), so the file is not untouched — that patch fixes a different fault, and this one is what remains. |
| `train_default_gae_internvl3_2b.sh` | CUDA device-side assert, `IndexKernel.cu:111 index out of bounds` | The architecture is supported. The image-placeholder count VAGEN writes into `prompt_token_ids` disagrees with what the engine expects; the leading hypothesis is InternVL's dynamic tiling (`max_dynamic_patch`) giving different tile counts on either side. Unverified. |

`qwen2_5_vl` still carries the old rope symbol, which is why the Qwen scripts are
unaffected.

## 4. `<think>` is a reserved token on some families — do not use `prompt_format: wm` there

On Qwen2.5-VL and InternVL3, `<think>` is three ordinary text tokens. On **Qwen3-VL,
Qwen3.5 and GLM** it is a single reserved control token tied to the model's own thinking
channel, and the model will not emit it as text. Sokoban's `wm` format requires all four of
`<observation><think><answer><prediction>`, and a response missing any of them has its
whole action list discarded (`strict_format`, deliberate). So on those families `wm` scores
zero while every other metric looks healthy.

Measured on Qwen3-VL-4B: `wm` → format 0.000 / score 0.000; `free_wm` → format 0.969 /
score 0.602.

Use `free_wm` (observation/answer/prediction, free prose between) or `free_think`
(`</think>` then `<answer>`) on those families.

**Sokoban and primitive_skill both *default* to `wm`**, so a `<think>`-reserving model needs
`prompt_format` set explicitly in the dataset yaml on either. frozenlake and navigation
default to `free_think` and are unaffected. What each environment offers:

| env | formats | default |
|---|---|---|
| sokoban | `wm`, `free_wm`, `free_think`, `answer` | `wm` |
| primitive_skill | `wm`, `free_think` | `wm` |
| frozenlake | `wm`, `free_think` | `free_think` |
| navigation | `wm`, `free_think`, `no_think`, `eval_mode` | `free_think` |

`free_wm` and `answer` exist only for sokoban.

## 5. `thinking_token_budget` bounds the think block, not the response — and not the cost

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

So the budget buys a well-formed `</think>` and a scoreable turn, not brevity. For brevity,
change the prompt format — `free_think` is the only sokoban format whose instructions tell
the model to stop reasoning, and it is what the shipped Qwen3.5 thinking config uses.

vLLM refuses the request if the budget is set and `reasoning_config` is not; see
`examples/train/sokoban/train_default_gae_qwen35_4b_think.sh` for the delimiters, which are
per-model-family knowledge and so live in the script rather than in VAGEN.

## 6. A `compact_budget` too small fails at runtime, not at startup

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
