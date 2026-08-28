"""VERL transport for the backend-neutral inference client.

Chat-template and multimodal rendering live under ``vagen.models``. This class only
accumulates the frames required by VERL's engine request and translates its output into
the shared ``BackendOutput`` contract.
"""

from __future__ import annotations

from typing import Any

from vagen.models import ModelAdapter, build_model_adapter
from vagen.rollout import BackendOutput, InferenceClient


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
        if self.response_limit:
            asked = params.get("max_new_tokens")
            if asked is None:
                asked = self.response_limit
            if asked <= 0:
                raise ValueError(
                    f"max_new_tokens={asked}: there is no room left to generate"
                )
            params["max_new_tokens"] = min(asked, self.response_limit)

        output = await self.server_manager.generate(
            request_id=self.request_id,
            prompt_ids=prompt_ids,
            sampling_params=params,
            image_data=self._images.get(self._active) or None,
            mm_processor_kwargs=self.mm_processor_kwargs or None,
        )
        extra = getattr(output, "extra_fields", None) or {}
        lo, hi = extra.get("min_global_steps"), extra.get("max_global_steps")
        token_ids = list(output.token_ids)
        return BackendOutput(
            text=self.model.decode(token_ids),
            token_ids=token_ids,
            logprobs=list(output.log_probs) if output.log_probs else None,
            prompt_token_ids=extra.get("prompt_token_ids"),
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
