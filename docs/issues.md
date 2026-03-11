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
