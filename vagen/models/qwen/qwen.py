"""Qwen-family chat-template and multimodal rendering."""

from __future__ import annotations

import logging
from typing import Any

from vagen.models._common.base import ModelAdapter
from vagen.models._common.image_tokens import image_token_ids, vision_sentinel_ids

logger = logging.getLogger(__name__)

_PLACEHOLDER_TURNS = [
    {"role": "system", "content": "placeholder"},
    {"role": "user", "content": "placeholder"},
]


def _parts(message: dict) -> list[dict]:
    content = message.get("content", "")
    return content if isinstance(content, list) else [{"type": "text", "text": str(content)}]


def _text_only(message: dict) -> str:
    return "".join(part.get("text", "") for part in _parts(message))


def _placeholder_flat() -> list[dict]:
    return [{"role": turn["role"], "content": _parts(turn)} for turn in _PLACEHOLDER_TURNS]


class QwenModelAdapter(ModelAdapter):
    """Incrementally render Qwen2.5-VL, Qwen3-VL, and Qwen3.5 chat edges."""

    def __init__(self, tokenizer, processor=None, *, apply_chat_template_kwargs=None,
                 mm_processor_kwargs=None):
        super().__init__(tokenizer, processor)
        self.apply_chat_template_kwargs = apply_chat_template_kwargs or {}
        self.mm_processor_kwargs = mm_processor_kwargs or {}
        self._ph_cache = None
        self._prefix_cache = None
        self._sep_cache = None

    def render(self, messages: list[dict], *, opening: bool) -> tuple[list[int], list[Any]]:
        if not messages:
            return [], []
        images = [image for message in messages for image in list(message.get("images") or [])]
        flat = [{"role": message["role"], "content": _parts(message)} for message in messages]
        if not opening:
            flat = [*_placeholder_flat(), *flat]

        if self.processor is not None:
            text = self.processor.apply_chat_template(
                flat, add_generation_prompt=True, tokenize=False,
                **self.apply_chat_template_kwargs,
            )
            inputs = self.processor(
                text=[text], images=images or None, return_tensors="pt",
                **self.mm_processor_kwargs,
            )
            ids = inputs["input_ids"].squeeze(0).tolist()
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

        prefix = self._template_prefix()[: -len(self.message_separator()) or None]
        if list(ids)[: len(prefix)] != prefix:
            raise ValueError(
                "chat template did not begin the continuation with the placeholder turn; "
                "stripping a fixed length would corrupt the span"
            )
        return list(ids)[len(prefix):], images

    def placeholder_ids(self) -> tuple[set[int], set[int]]:
        if self._ph_cache is None:
            source = self.processor or self.tokenizer
            self._ph_cache = (
                image_token_ids(source), vision_sentinel_ids(source)
            ) if source is not None else (set(), set())
        return self._ph_cache

    def message_separator(self) -> list[int]:
        if self._sep_cache is not None:
            return self._sep_cache
        self._sep_cache = []
        try:
            eos = getattr(self.tokenizer, "eos_token_id", None)
            one = self._render_plain(_PLACEHOLDER_TURNS)
            if eos is not None and eos in one:
                sep = one[len(one) - 1 - one[::-1].index(eos):][1:]
                if sep and self._separator_reproduces_template(sep):
                    self._sep_cache = sep
        except Exception as exc:  # noqa: BLE001
            logger.warning("could not derive the message separator (%s)", exc)
        return self._sep_cache

    def _render_plain(self, messages) -> list[int]:
        return list(self.tokenizer.apply_chat_template(
            [{"role": message["role"], "content": _text_only(message)} for message in messages],
            add_generation_prompt=False, tokenize=True, return_dict=False,
            **self.apply_chat_template_kwargs,
        ))

    def _separator_reproduces_template(self, sep: list[int]) -> bool:
        user = {"role": "user", "content": "U"}
        assistant = {"role": "assistant", "content": "A"}
        canonical = list(self.tokenizer.apply_chat_template(
            [user, assistant, user], add_generation_prompt=True, tokenize=True,
            return_dict=False, **self.apply_chat_template_kwargs,
        ))
        opening = list(self.tokenizer.apply_chat_template(
            [user], add_generation_prompt=True, tokenize=True, return_dict=False,
            **self.apply_chat_template_kwargs,
        ))
        plain_user = self._render_plain([user])
        generated = self._render_plain([user, assistant])[len(plain_user):]
        if len(opening) > len(plain_user):
            generated = generated[len(opening) - len(plain_user):]
        if len(generated) <= len(sep):
            return False
        generated = generated[:-len(sep)]
        rest = canonical[len(opening) + len(generated):]
        return opening + generated + rest == canonical and rest[:len(sep)] == sep

    def _template_prefix(self) -> list[int]:
        if self._prefix_cache is None:
            placeholder = _placeholder_flat()
            if self.processor is not None:
                text = self.processor.apply_chat_template(
                    placeholder, add_generation_prompt=False, tokenize=False,
                    **self.apply_chat_template_kwargs,
                )
                self._prefix_cache = self.processor(
                    text=[text], return_tensors="pt"
                )["input_ids"].squeeze(0).tolist()
            else:
                self._prefix_cache = self.tokenizer.apply_chat_template(
                    [{"role": message["role"], "content": _text_only(message)}
                     for message in placeholder],
                    add_generation_prompt=False, tokenize=True, return_dict=False,
                    **self.apply_chat_template_kwargs,
                )
        return list(self._prefix_cache)


__all__ = ["QwenModelAdapter"]
