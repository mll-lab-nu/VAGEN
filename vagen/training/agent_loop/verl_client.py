"""VERL transport for the backend-neutral inference client.

Chat-template and multimodal rendering live under ``vagen.models``. This class only
accumulates the frames required by VERL's engine request and translates its output into
the shared ``BackendOutput`` contract.
"""

from __future__ import annotations

import hashlib
from typing import Any

from vagen.models import ModelAdapter, build_model_adapter
from vagen.rollout import BackendOutput, InferenceClient


def _sampling_seed(request_id: str, call_id: int, base_seed: int) -> int:
    payload = f"{request_id}\x1f{call_id}\x1f{base_seed}".encode("utf-8")
    return int.from_bytes(hashlib.blake2b(payload, digest_size=8).digest(), "big") % (2**31)


def _backend_sampling_params(
    params: dict[str, Any],
    *,
    backend: str,
    request_id: str,
    call_id: int,
    base_seed: int,
    full_determinism: bool,
) -> dict[str, Any]:
    """Translate backend-specific options and attach a stable per-call seed."""
    params = dict(params)
    if backend == "sglang":
        include_stop = params.pop("include_stop_str_in_output", None)
        if include_stop is not None:
            params.setdefault("no_stop_trim", bool(include_stop))
        if full_determinism:
            params.setdefault("sampling_seed", _sampling_seed(request_id, call_id, base_seed))
    elif full_determinism:
        params.setdefault("seed", _sampling_seed(request_id, call_id, base_seed))
    return params


def _collapse_placeholder_runs(token_ids: list[int], placeholders: set[int]) -> list[int]:
    """Collapse processor-expanded vision runs for SGLang's raw-image API.

    VAGEN stores the processor-expanded sequence because that is the sequence used by
    the actor and critic. SGLang's ``GenerateReqInput(input_ids=..., image_data=...)``
    decodes the ids and invokes the processor again, however, so handing it an expanded
    run makes every repeated image token look like a separate image placeholder. Send
    one token per run and let SGLang expand it back from the accompanying raw images.
    """
    collapsed: list[int] = []
    previous_was_placeholder = False
    for token_id in token_ids:
        token_id = int(token_id)
        is_placeholder = token_id in placeholders
        if not is_placeholder or not previous_was_placeholder:
            collapsed.append(token_id)
        previous_was_placeholder = is_placeholder
    return collapsed


class VerlClient(InferenceClient):
    """Talk to VERL's ``LLMServerClient`` using a model-family adapter."""

    def __init__(
        self,
        server_manager,
        tokenizer,
        processor,
        *,
        model_adapter: ModelAdapter | None = None,
        model_adapter_name: str = "auto",
        apply_chat_template_kwargs=None,
        mm_processor_kwargs=None,
        sampling_params=None,
        request_id=None,
        response_limit=None,
        backend="vllm",
        full_determinism=False,
        rollout_seed=0,
    ):
        super().__init__()
        self.server_manager = server_manager
        self.model = model_adapter or build_model_adapter(
            model_adapter_name,
            tokenizer,
            processor,
            apply_chat_template_kwargs=apply_chat_template_kwargs or {},
            mm_processor_kwargs=mm_processor_kwargs or {},
        )
        self.tokenizer = self.model.tokenizer
        self.processor = self.model.processor
        self.mm_processor_kwargs = mm_processor_kwargs or {}
        self.sampling_params = dict(sampling_params or {})
        self.request_id = request_id
        self.response_limit = response_limit
        self.backend = str(backend)
        self.full_determinism = bool(full_determinism)
        self.rollout_seed = int(rollout_seed)
        self._images: dict[str, list[Any]] = {}
        self._active: str | None = None

    def encode(self, messages: list[dict]) -> list[int]:
        ids, images = self.render(messages)
        if self._active is not None and images:
            self._images.setdefault(self._active, []).extend(images)
        return ids

    def render(self, messages: list[dict]) -> tuple[list[int], list[Any]]:
        opening = self._conversations[self._active].prompt_len is None if self._active else True
        return self.model.render(messages, opening=opening)

    def _placeholder_ids(self):
        """Compatibility for the training row assembler; owned by the model adapter."""
        return self.model.placeholder_ids()

    def _message_separator(self):
        """Compatibility for diagnostics while the implementation lives in the adapter."""
        method = getattr(self.model, "message_separator", None)
        return method() if method is not None else []

    async def generate(self, prompt_ids: list[int], **kwargs) -> BackendOutput:
        params = dict(self.sampling_params)
        params.update(kwargs.pop("sampling_params", {}) or {})
        params = _backend_sampling_params(
            params,
            backend=self.backend,
            request_id=str(self.request_id),
            call_id=max(self._call_counter - 1, 0),
            base_seed=self.rollout_seed,
            full_determinism=self.full_determinism,
        )
        if self.response_limit:
            asked = params.get("max_new_tokens")
            if asked is None:
                asked = self.response_limit
            if asked <= 0:
                raise ValueError(
                    f"max_new_tokens={asked}: there is no room left to generate"
                )
            params["max_new_tokens"] = min(asked, self.response_limit)

        images = self._images.get(self._active) or None
        server_prompt_ids = prompt_ids
        if self.backend == "sglang" and images:
            placeholders, _ = self.model.placeholder_ids()
            server_prompt_ids = _collapse_placeholder_runs(prompt_ids, placeholders)

        output = await self.server_manager.generate(
            request_id=self.request_id,
            prompt_ids=server_prompt_ids,
            sampling_params=params,
            image_data=images,
            mm_processor_kwargs=self.mm_processor_kwargs or None,
        )
        extra = getattr(output, "extra_fields", None) or {}
        lo, hi = extra.get("min_global_steps"), extra.get("max_global_steps")
        token_ids = list(output.token_ids)
        return BackendOutput(
            text=self.model.decode(token_ids),
            token_ids=token_ids,
            logprobs=list(output.log_probs) if output.log_probs else None,
            # SGLang re-expands the compressed multimodal request to this processor-
            # expanded sequence before sampling. Keep the local sequence as the training
            # prompt unless a backend explicitly reports the exact prompt it ran.
            prompt_token_ids=extra.get("prompt_token_ids") or (
                list(prompt_ids) if server_prompt_ids is not prompt_ids else None
            ),
            stop_reason=getattr(output, "stop_reason", None),
            weights_version=(int(lo), int(hi)) if lo is not None and hi is not None else None,
        )

    def _open(self, conversation_id):
        resolved = super()._open(conversation_id)
        self._active = resolved
        return resolved

    def images(self, conversation_id: str) -> list[Any]:
        return list(self._images.get(conversation_id) or [])


__all__ = ["VerlClient"]
