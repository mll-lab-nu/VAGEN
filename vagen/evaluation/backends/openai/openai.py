from __future__ import annotations
import logging
import os
from typing import Any, Dict, Iterable, List, Tuple
from openai import AsyncAzureOpenAI, AsyncOpenAI
from PIL import Image
from vagen.evaluation.backends._common.base import EvaluationBackend
from vagen.evaluation.backends._common.rendering import pil_to_dataurl_png, compile_text_images_for_order
from vagen.evaluation.backends._common.registry import register_adapter, register_client


logger = logging.getLogger(__name__)


class OutputBudgetExceeded(RuntimeError):
    """The endpoint stopped at ``max_tokens`` before writing any content.

    Raised rather than returned empty so the episode is recorded as an error and
    can be re-run, instead of being written as a failure that resume then treats
    as finished work.
    """


@register_client("openai", "openai_responses")
def build_client_openai(cfg: Dict[str, Any]) -> AsyncOpenAI:
    api_key = cfg.get("api_key") or os.getenv("OPENAI_API_KEY", "")
    base_url = cfg.get("base_url")
    return AsyncOpenAI(api_key=api_key, base_url=base_url) if base_url else AsyncOpenAI(api_key=api_key)


@register_client("azure", "azure_responses")
def build_client_azure(cfg: Dict[str, Any]) -> AsyncAzureOpenAI:
    endpoint = cfg.get("azure_endpoint") or os.getenv("AZURE_OPENAI_ENDPOINT", "")
    api_key = cfg.get("azure_api_key") or os.getenv("AZURE_OPENAI_API_KEY", "") or os.getenv("AZURE_API_KEY", "")
    api_version = cfg.get("azure_api_version") or os.getenv("AZURE_OPENAI_API_VERSION", "2024-12-01-preview")
    if not endpoint or not api_key:
        raise ValueError("Azure endpoint/api_key missing.")
    return AsyncAzureOpenAI(api_version=api_version, azure_endpoint=endpoint, api_key=api_key)

@register_adapter("openai", "azure")
class OpenAIAdapter(EvaluationBackend):
    """
    OpenAI-compatible multimodal adapter:
    - messages use content parts with {"type": "text"} and {"type": "image_url"}.
    - capability flags allow omitting unsupported kwargs (e.g., o3).
    """

    def __init__(
        self,
        client,
        model: str,

    ):
        self.client = client
        self.model = model


    def _segments_to_content(self, segs: List[Tuple[str, Any]]) -> List[Dict[str, Any]]:
        content: List[Dict[str, Any]] = []
        for kind, val in segs:
            if kind == "text":
                if str(val).strip():
                    content.append({"type": "text", "text": str(val)})
            else:
                content.append({"type": "image_url", "image_url": {"url": pil_to_dataurl_png(val)}})
        return content

    def format_system(self, text: str, images: List[Image.Image]) -> Dict[str, Any]:
        segs = compile_text_images_for_order(text, images)
        return {"role": "system", "content": self._segments_to_content(segs)}

    def format_user_turn(self, text: str, images: List[Image.Image]) -> Dict[str, Any]:
        segs = compile_text_images_for_order(text, images)
        return {"role": "user", "content": self._segments_to_content(segs)}

    async def acompletion(self, messages: List[Dict[str, Any]], **chat_config: Any) -> str:

        resp = await self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            **chat_config,
        )
        choice = resp.choices[0]
        content = choice.message.content or ""
        # `finish_reason` carries the only signal that separates "the model declined"
        # from "the model was cut off". A reasoning model spends its output budget on
        # hidden thinking before it writes anything, so a budget that is merely too
        # small comes back as `length` with no content -- indistinguishable, once the
        # reason is dropped, from a refusal or a content filter. A harness then re-asks
        # against the same budget, which cannot help, and scores the episode as a
        # failure. Measured on one reasoning model at high effort, that silently took
        # out 29% of its rows.
        reason = getattr(choice, "finish_reason", None)
        if reason == "length" and not content.strip():
            raise OutputBudgetExceeded(
                f"{self.model} stopped at its output budget with no content "
                f"(finish_reason=length). Raise max_tokens, or "
                f"response_length_per_turn if a harness sets the per-call limit: "
                f"hidden reasoning is billed against it."
            )
        if reason == "length":
            logger.warning(
                "%s was truncated at its output budget after %d characters; "
                "keeping the partial reply.", self.model, len(content))
        return content
