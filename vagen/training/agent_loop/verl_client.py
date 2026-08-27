"""VERL-backed inference client.

The backend half of §7: everything conversation- and mask-related is inherited from
``InferenceClient``; this only knows how to render messages with a processor and how to
call verl's server. Images accumulate per conversation because the engine re-processes
all of them on every call.
"""

from __future__ import annotations

import logging
from typing import Any

from vagen.rollout import BackendOutput, InferenceClient
from vagen.models import image_token_ids, vision_sentinel_ids

logger = logging.getLogger(__name__)


def _parts(message: dict) -> list[dict]:
    content = message.get("content", "")
    return content if isinstance(content, list) else [{"type": "text", "text": str(content)}]


def images_of(message: dict) -> list[Any]:
    """Images carried alongside a message, in the order their placeholders appear."""
    return list(message.get("images") or [])


#: Prepended to a continuation span so the template emits no system block of its own,
#: then stripped. Every use must render the same turns or the strip is the wrong length.
#:
#: The system turn is what suppresses the template's default system block ("You are a
#: helpful assistant."), which would otherwise appear in the continuation render but not
#: in the prefix and leave its tail spliced ahead of every observation.
#:
#: The user turn is for Qwen3.5, whose template scans for a non-``<tool_response>`` user
#: message and calls ``raise_exception('No user query found in messages.')`` when there is
#: none -- so a system turn alone cannot even be rendered to compute the prefix, let alone
#: prepended to an assistant-only span. Verified on Qwen2.5-VL, Qwen3-VL and Qwen3.5:
#: with both turns the prefix is 12 tokens on all three, no default system block is
#: injected anywhere, and ``render([*placeholder, *delta])`` begins with
#: ``render(placeholder)`` for assistant-only, user-only and mixed spans.
_PLACEHOLDER_TURNS = [
    {"role": "system", "content": "placeholder"},
    {"role": "user", "content": "placeholder"},
]


def _placeholder_flat() -> list[dict]:
    """The placeholder in the same parts-list shape as every other rendered message."""
    return [{"role": t["role"], "content": _parts(t)} for t in _PLACEHOLDER_TURNS]


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
        """Render and record. See ``render`` for why the two are separable."""
        ids, images = self.render(messages)
        if self._active is not None and images:
            self._images.setdefault(self._active, []).extend(images)
        return ids

    def _placeholder_ids(self):
        """The ids a picture sits behind, and the sentinels bracketing it. Cached: the
        answer is a property of the model, and this runs per call."""
        if getattr(self, "_ph_cache", None) is None:
            source = self.processor or self.tokenizer
            self._ph_cache = ((image_token_ids(source), vision_sentinel_ids(source))
                              if source is not None else (set(), set()))
        return self._ph_cache

    def render(self, messages: list[dict]) -> tuple[list[int], list]:
        """Tokens and frames for these messages, recording nothing.

        Split out from ``encode`` so a caller can ask "how big is this observation?"
        before deciding what to do with it. Asking used to mean encoding, and encoding
        recorded the frames against the conversation -- so measuring once shipped every
        picture twice, which is the alignment failure this codebase keeps finding.

        Rendered with a placeholder turn in front and then stripped, so a mid-conversation
        span is tokenized the way it will sit in the full sequence rather than as if it
        began the prompt -- chat templates prepend a system block otherwise.
        """
        if not messages:
            return [], []
        new_images = [image for message in messages for image in images_of(message)]

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
            flat = [*_placeholder_flat(), *flat]

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
            return ids, new_images
        prefix = self._template_prefix()[: -len(self._message_separator()) or None]
        if ids[: len(prefix)] != prefix:
            # The strip is only safe if the rendered span really starts with it.
            raise ValueError(
                "chat template did not begin the continuation with the placeholder turn; "
                "stripping a fixed length here would corrupt the span"
            )
        return ids[len(prefix) :], new_images

    def _message_separator(self) -> list[int]:
        """What the template puts after a message that the model never generates.

        Qwen closes every message with ``<|im_end|>\n``. The model stops at
        ``<|im_end|>``, so the newline is template output -- and stripping the placeholder
        turn whole took it away, leaving every continuation one token short of what the
        template would have produced. Rollout and training saw the same seam, so nothing
        downstream could tell; it is off-distribution input, not a mismatch.

        Derived, then **verified**. The derivation reads the tokenizer's declared
        terminator and takes what follows it, which is the same class of thing as reading
        ``image_token_id`` -- but a family whose chat template closes a message with
        something other than its ``eos_token`` would derive the wrong answer silently. So
        the result is checked against a real two-turn render, and on any mismatch this
        returns nothing and the old behaviour stands. Being one token short is survivable;
        splicing the wrong tokens into every turn boundary is not.
        """
        if getattr(self, "_sep_cache", None) is not None:
            return self._sep_cache
        self._sep_cache = []
        try:
            eos = getattr(self.tokenizer, "eos_token_id", None)
            one = self._render_plain(_PLACEHOLDER_TURNS)
            if eos is not None and eos in one:
                sep = one[len(one) - 1 - one[::-1].index(eos):][1:]
                if sep and self._separator_reproduces_the_template(sep):
                    self._sep_cache = sep
        except Exception as exc:  # noqa: BLE001 - a failed derivation must not stop a run
            logger.warning("could not derive the message separator (%s); continuations "
                           "will be one token short of the canonical render", exc)
        return self._sep_cache

    def _render_plain(self, messages) -> list[int]:
        """Tokenize messages through the chat template, no images, no generation prompt."""
        return list(self.tokenizer.apply_chat_template(
            [{"role": m["role"], "content": _text_only(m)} for m in messages],
            add_generation_prompt=False, tokenize=True, return_dict=False,
            **self.apply_chat_template_kwargs))

    def _separator_reproduces_the_template(self, sep: list[int]) -> bool:
        """Does building a two-turn exchange incrementally, with this separator, equal
        what the template produces in one go? If not, the derivation is wrong."""
        user = {"role": "user", "content": "U"}
        assistant = {"role": "assistant", "content": "A"}
        canonical = list(self.tokenizer.apply_chat_template(
            [user, assistant, user], add_generation_prompt=True, tokenize=True,
            return_dict=False, **self.apply_chat_template_kwargs))
        opening = list(self.tokenizer.apply_chat_template(
            [user], add_generation_prompt=True, tokenize=True, return_dict=False,
            **self.apply_chat_template_kwargs))
        # what the model would emit for that assistant turn: its content, then the close
        whole = self._render_plain([user, assistant])
        generated = whole[len(self._render_plain([user])):]
        generated = generated[len(opening) - len(self._render_plain([user])):] \
            if len(opening) > len(self._render_plain([user])) else generated
        if len(generated) <= len(sep):
            return False
        generated = generated[:-len(sep)]          # the model stops before the separator
        rest = canonical[len(opening) + len(generated):]
        return opening + generated + rest == canonical and rest[: len(sep)] == sep

    def _template_prefix(self) -> list[int]:
        """Tokens a chat template emits before the first message's content.

        Cached: rendering it costs a template application and it never changes.
        """
        if getattr(self, "_prefix_cache", None) is None:
            placeholder = _placeholder_flat()
            if self.processor is not None:
                text = self.processor.apply_chat_template(
                    placeholder, add_generation_prompt=False, tokenize=False, **self.apply_chat_template_kwargs
                )
                self._prefix_cache = self.processor(text=[text], return_tensors="pt")["input_ids"].squeeze(0).tolist()
            else:
                # `_text_only`, as on every other tokenizer-path render. A bare tokenizer's
                # template concatenates `message['content']` as a string, so handing it the
                # parts list raises TypeError -- which is what a text-only model did here,
                # on the first continuation, while both other call sites converted.
                self._prefix_cache = self.tokenizer.apply_chat_template(
                    [{"role": m["role"], "content": _text_only(m)} for m in placeholder],
                    add_generation_prompt=False, tokenize=True, return_dict=False,
                    **self.apply_chat_template_kwargs,
                )
        return self._prefix_cache

    # ----------------------------------------------------------------- generating
    async def generate(self, prompt_ids: list[int], **kwargs) -> BackendOutput:
        params = dict(self.sampling_params)
        params.update(kwargs.pop("sampling_params", {}) or {})
        if self.response_limit:
            # `or` would be wrong here: 0 is falsy, so a caller asking for zero tokens --
            # which is what a budget with no room left computes to -- would silently be
            # given the whole limit instead. The one case where the answer matters most
            # inverts into its opposite. A negative would pass through untouched.
            asked = params.get("max_new_tokens")
            if asked is None:
                asked = self.response_limit
            if asked <= 0:
                raise ValueError(
                    f"max_new_tokens={asked}: there is no room left to generate in. The "
                    f"caller has to decide what to do about that -- compact, or end the "
                    f"conversation -- rather than asking for a generation that cannot happen."
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
        # min/max only differ under partial rollout, where a resumed generation can span a
        # weight update. Read here because this is the only place they exist: the fields
        # are set by verl's resuming client and nothing downstream sees `output`.
        lo, hi = extra.get("min_global_steps"), extra.get("max_global_steps")
        return BackendOutput(
            text=self.tokenizer.decode(output.token_ids, skip_special_tokens=True),
            token_ids=list(output.token_ids),
            logprobs=list(output.log_probs) if output.log_probs else None,
            prompt_token_ids=extra.get("prompt_token_ids"),
            stop_reason=getattr(output, "stop_reason", None),
            weights_version=(int(lo), int(hi)) if lo is not None and hi is not None else None,
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
