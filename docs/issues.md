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
| `train_default_gae_glm41v_9b.sh` | `ImportError: apply_multimodal_rotary_pos_emb` at the first attention forward | transformers removed the symbol from `models.glm4v`; GLM-4V's mrope was refactored to `apply_rotary_pos_emb` + `rotate_half_llm`. `verl/verl/models/transformers/glm4v.py:313` still imports the old name. Porting it is real work and a subtly wrong rope corrupts training **silently** — there is no GLM baseline here to catch that against. |
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
(`</think>` then `<answer>`) on those families. Note `free_wm` and `answer` exist **only**
for sokoban — frozenlake and primitive_skill offer `free_think` and `wm` only, and
primitive_skill *defaults* to `wm`, so a `<think>`-reserving model there needs
`prompt_format: free_think` set explicitly.
