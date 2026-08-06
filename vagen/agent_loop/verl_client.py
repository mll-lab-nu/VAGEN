"""verl-backed inference client.

The backend half of §7: everything conversation- and mask-related is inherited from
``InferenceClient``; this only knows how to render messages with a processor and how to
call verl's server. Images accumulate per conversation because the engine re-processes
all of them on every call.
"""

from __future__ import annotations

from typing import Any

from vagen.core.client import BackendOutput, InferenceClient


def _parts(message: dict) -> list[dict]:
    content = message.get("content", "")
    return content if isinstance(content, list) else [{"type": "text", "text": str(content)}]


def images_of(message: dict) -> list[Any]:
    """Images carried alongside a message, in the order their placeholders appear."""
    return list(message.get("images") or [])


#: Prepended to a continuation span so the template emits no system block of its own,
#: then stripped. Both uses must be the same turn or the strip is the wrong length.
_PLACEHOLDER_TURN = {"role": "system", "content": "placeholder"}


class VerlClient(InferenceClient):
    """Talks to verl's ``LLMServerClient``."""

    def __init__(self, server_manager, tokenizer, processor, *, apply_chat_template_kwargs=None,
                 mm_processor_kwargs=None, sampling_params=None, request_id=None, response_limit=None):
        super().__init__()
        self.server_manager = server_manager
        self.tokenizer = tokenizer
        self.processor = processor
        self.apply_chat_template_kwargs = apply_chat_template_kwargs or {}
        self.mm_processor_kwargs = mm_processor_kwargs or {}
        self.sampling_params = dict(sampling_params or {})
        self.request_id = request_id
        self.response_limit = response_limit
        self._images: dict[str, list[Any]] = {}
        self._active: str | None = None

    # ------------------------------------------------------------------ encoding
    def encode(self, messages: list[dict]) -> list[int]:
        """Render messages to tokens, recording their images against the conversation.

        Rendered with a placeholder turn in front and then stripped, so a mid-conversation
        span is tokenized the way it will sit in the full sequence rather than as if it
        began the prompt -- chat templates prepend a system block otherwise.
        """
        if not messages:
            return []
        new_images = [image for message in messages for image in images_of(message)]
        if self._active is not None:
            self._images.setdefault(self._active, []).extend(new_images)

        opening = self._conversations[self._active].prompt_len is None if self._active else True
        flat = [{"role": m["role"], "content": _parts(m)} for m in messages]
        # Actually put the placeholder turn in front. Rendering the delta alone and then
        # dropping the placeholder's length is not the same thing: with no system message
        # of its own the template injects a default one ("You are a helpful assistant."),
        # which is longer, so the strip leaves its tail -- " helpful assistant.<|im_end|>"
        # -- spliced in ahead of every observation after the first. With the placeholder
        # present the template injects nothing and the strip is exact.
        if not opening:
            # In the same shape as the rest, and the same shape _template_prefix renders.
            # A template can tokenize a plain string differently from a one-element parts
            # list, and then the strip is measured against a render that never happened.
            flat = [{"role": _PLACEHOLDER_TURN["role"], "content": _parts(_PLACEHOLDER_TURN)}, *flat]

        if self.processor is not None:
            text = self.processor.apply_chat_template(
                flat, add_generation_prompt=True, tokenize=False, **self.apply_chat_template_kwargs
            )
            inputs = self.processor(
                text=[text], images=new_images or None, return_tensors="pt", **self.mm_processor_kwargs
            )
            ids = inputs["input_ids"].squeeze(0).tolist()
        else:
            if new_images:
                raise ValueError("environment produced images but the model has no processor")
            # Built from `flat`, which carries the placeholder turn when this is a
            # continuation. Rebuilding it from `messages` here dropped the placeholder in
            # the text-only path while the multimodal path kept it, so the two branches
            # stripped different things from different renders.
            ids = self.tokenizer.apply_chat_template(
                [{"role": m["role"], "content": _text_only(m)} for m in flat],
                add_generation_prompt=True, tokenize=True, return_dict=False,
                **self.apply_chat_template_kwargs,
            )
        if opening:
            return ids
        prefix = self._template_prefix()
        if ids[: len(prefix)] != prefix:
            # The strip is only safe if the rendered span really starts with it.
            raise ValueError(
                "chat template did not begin the continuation with the placeholder turn; "
                "stripping a fixed length here would corrupt the span"
            )
        return ids[len(prefix) :]

    def _template_prefix(self) -> list[int]:
        """Tokens a chat template emits before the first message's content.

        Cached: rendering it costs a template application and it never changes.
        """
        if getattr(self, "_prefix_cache", None) is None:
            placeholder = [{"role": _PLACEHOLDER_TURN["role"], "content": _parts(_PLACEHOLDER_TURN)}]
            if self.processor is not None:
                text = self.processor.apply_chat_template(
                    placeholder, add_generation_prompt=False, tokenize=False, **self.apply_chat_template_kwargs
                )
                self._prefix_cache = self.processor(text=[text], return_tensors="pt")["input_ids"].squeeze(0).tolist()
            else:
                self._prefix_cache = self.tokenizer.apply_chat_template(
                    placeholder, add_generation_prompt=False, tokenize=True, return_dict=False,
                    **self.apply_chat_template_kwargs,
                )
        return self._prefix_cache

    # ----------------------------------------------------------------- generating
    async def generate(self, prompt_ids: list[int], **kwargs) -> BackendOutput:
        params = dict(self.sampling_params)
        params.update(kwargs.pop("sampling_params", {}) or {})
        if self.response_limit:
            params["max_new_tokens"] = min(params.get("max_new_tokens") or self.response_limit, self.response_limit)

        output = await self.server_manager.generate(
            request_id=self.request_id,
            prompt_ids=prompt_ids,
            sampling_params=params,
            image_data=self._images.get(self._active) or None,
            mm_processor_kwargs=self.mm_processor_kwargs or None,
        )
        extra = getattr(output, "extra_fields", None) or {}
        return BackendOutput(
            text=self.tokenizer.decode(output.token_ids, skip_special_tokens=True),
            token_ids=list(output.token_ids),
            logprobs=list(output.log_probs) if output.log_probs else None,
            prompt_token_ids=extra.get("prompt_token_ids"),
        )

    # ------------------------------------------------------------------ plumbing
    def _open(self, conversation_id):
        """Track which conversation `encode` and `generate` are working on.

        They need it -- images accumulate per conversation, and a mid-conversation span
        is tokenized differently from an opening one -- and the base class resolves the
        id here.
        """
        resolved = super()._open(conversation_id)
        self._active = resolved
        return resolved

    def images(self, conversation_id: str) -> list[Any]:
        return list(self._images.get(conversation_id) or [])


def _text_only(message: dict) -> str:
    return "".join(part.get("text", "") for part in _parts(message))
