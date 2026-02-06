# All comments are in English.
from __future__ import annotations
from typing import Any, Dict, Iterable, List, Tuple
from PIL import Image
from vagen.evaluate.adapters.base_adapter import ModelAdapter
from vagen.evaluate.utils.mm_utils import pil_to_dataurl_png, compile_text_images_for_order

class OpenAIAdapter(ModelAdapter):
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
        return resp.choices[0].message.content or ""
