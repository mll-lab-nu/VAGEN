from __future__ import annotations
from typing import Any, Dict, List, Tuple
from PIL import Image
from vagen.evaluation.backends._common.base import EvaluationBackend
from vagen.evaluation.backends._common.rendering import pil_to_dataurl_png, compile_text_images_for_order
from vagen.evaluation.backends._common.options import filter_chat_kwargs
from vagen.evaluation.backends._common.registry import register_adapter


@register_adapter("openai_responses", "azure_responses")
class OpenAIResponsesAdapter(EvaluationBackend):
    """
    Adapter for OpenAI Responses API (client.responses.create).
    Required for models like gpt-5.4-pro that only support the Responses API.
    """

    def __init__(self, client, model: str):
        self.client = client
        self.model = model

    def _segments_to_content(self, segs: List[Tuple[str, Any]]) -> List[Dict[str, Any]]:
        content: List[Dict[str, Any]] = []
        for kind, val in segs:
            if kind == "text":
                if str(val).strip():
                    content.append({"type": "input_text", "text": str(val)})
            else:
                content.append({"type": "input_image", "image_url": pil_to_dataurl_png(val)})
        return content

    def format_system(self, text: str, images: List[Image.Image]) -> Dict[str, Any]:
        segs = compile_text_images_for_order(text, images)
        return {"role": "system", "content": self._segments_to_content(segs)}

    def format_user_turn(self, text: str, images: List[Image.Image]) -> Dict[str, Any]:
        segs = compile_text_images_for_order(text, images)
        return {"role": "user", "content": self._segments_to_content(segs)}

    def format_assistant_turn(self, text: str) -> Dict[str, Any]:
        return {"role": "assistant", "content": [{"type": "output_text", "text": text}]}

    async def acompletion(self, messages: List[Dict[str, Any]], **chat_config: Any) -> str:
        """Translate the shared token limit to the Responses API's parameter.

        If a native limit is also configured, keep the smaller one so it cannot
        relax the evaluation client's per-turn ceiling.
        """
        params = filter_chat_kwargs(chat_config)
        max_tokens = params.pop("max_tokens", None)
        if max_tokens is not None:
            native_limit = params.get("max_output_tokens")
            params["max_output_tokens"] = (
                min(max_tokens, native_limit) if native_limit is not None else max_tokens
            )

        resp = await self.client.responses.create(
            model=self.model,
            input=messages,
            **params,
        )
        return resp.output_text or ""
