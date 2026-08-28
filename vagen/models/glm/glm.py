"""GLM-4.6V chat-template and multimodal rendering."""

from __future__ import annotations

from typing import Any

from vagen.models._common.base import ModelAdapter
from vagen.models._common.image_tokens import image_token_ids, vision_sentinel_ids

_SCAFFOLD = [
    {"role": "system", "content": "placeholder"},
    {"role": "user", "content": "placeholder"},
]


def _parts(message: dict) -> list[dict]:
    content = message.get("content", "")
    return content if isinstance(content, list) else [{"type": "text", "text": str(content)}]


def _text_only(message: dict) -> str:
    return "".join(part.get("text", "") for part in _parts(message))


class GLMModelAdapter(ModelAdapter):
    """Incrementally render GLM's ``[gMASK]<sop>`` conversation protocol."""

    def __init__(self, tokenizer, processor=None, *, apply_chat_template_kwargs=None,
                 mm_processor_kwargs=None):
        super().__init__(tokenizer, processor)
        self.apply_chat_template_kwargs = apply_chat_template_kwargs or {}
        self.mm_processor_kwargs = mm_processor_kwargs or {}
        self._prefix_cache = None
        self._ph_cache = None

    def render(self, messages: list[dict], *, opening: bool) -> tuple[list[int], list[Any]]:
        if not messages:
            return [], []
        images = [image for message in messages for image in list(message.get("images") or [])]
        flat = [{"role": message["role"], "content": _parts(message)} for message in messages]
        if not opening:
            flat = [*self._scaffold(), *flat]

        if self.processor is not None:
            text = self.processor.apply_chat_template(
                flat, add_generation_prompt=True, tokenize=False,
                **self.apply_chat_template_kwargs,
            )
            ids = self.processor(
                text=[text], images=images or None, return_tensors="pt",
                **self.mm_processor_kwargs,
            )["input_ids"].squeeze(0).tolist()
        else:
            if images:
                raise ValueError("environment produced images but the model has no processor")
            ids = self.tokenizer.apply_chat_template(
                [{"role": message["role"], "content": _text_only(message)} for message in flat],
                add_generation_prompt=True, tokenize=True, return_dict=False,
                **self.apply_chat_template_kwargs,
            )
        if opening:
            return list(ids), images

        prefix = self._template_prefix()
        if list(ids)[:len(prefix)] != prefix:
            raise ValueError(
                "GLM chat template did not begin the continuation with its scaffold; "
                "stripping it would corrupt the span"
            )
        return list(ids)[len(prefix):], images

    def _scaffold(self) -> list[dict]:
        return [{"role": turn["role"], "content": _parts(turn)} for turn in _SCAFFOLD]

    def _template_prefix(self) -> list[int]:
        if self._prefix_cache is None:
            scaffold = self._scaffold()
            if self.processor is not None:
                text = self.processor.apply_chat_template(
                    scaffold, add_generation_prompt=False, tokenize=False,
                    **self.apply_chat_template_kwargs,
                )
                self._prefix_cache = self.processor(
                    text=[text], return_tensors="pt"
                )["input_ids"].squeeze(0).tolist()
            else:
                self._prefix_cache = self.tokenizer.apply_chat_template(
                    [{"role": message["role"], "content": _text_only(message)}
                     for message in scaffold],
                    add_generation_prompt=False, tokenize=True, return_dict=False,
                    **self.apply_chat_template_kwargs,
                )
        return list(self._prefix_cache)

    def placeholder_ids(self) -> tuple[set[int], set[int]]:
        if self._ph_cache is None:
            source = self.processor or self.tokenizer
            self._ph_cache = (
                image_token_ids(source), vision_sentinel_ids(source)
            ) if source is not None else (set(), set())
        return self._ph_cache


__all__ = ["GLMModelAdapter"]
