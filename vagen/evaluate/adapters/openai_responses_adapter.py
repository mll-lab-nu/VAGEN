# All comments are in English.
from __future__ import annotations
from typing import Any, Dict, List, Tuple
from PIL import Image
from vagen.evaluate.adapters.base_adapter import ModelAdapter
from vagen.evaluate.utils.mm_utils import pil_to_dataurl_png, compile_text_images_for_order
from vagen.evaluate.registry import register_adapter


@register_adapter("openai_responses", "azure_responses")
class OpenAIResponsesAdapter(ModelAdapter):
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

    async def acompletion(self, messages: List[Dict[str, Any]], **chat_config: Any) -> str:

        resp = await self.client.responses.create(
            model=self.model,
            input=messages,
            **chat_config,
        )
        return resp.output_text or ""
