from __future__ import annotations
from typing import Any, Dict, List, Tuple
from PIL import Image
from vagen.evaluation.backends._common.base import EvaluationBackend
from vagen.evaluation.backends._common.rendering import pil_to_dataurl_png, compile_text_images_for_order
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
        kwargs = dict(chat_config)
        # Ask for the reasoning summary as well: the raw thinking never leaves the API, the
        # summary is the only trace of it we can store (raw_responses.json).
        reasoning = kwargs.get("reasoning") if isinstance(kwargs.get("reasoning"), dict) else {}
        kwargs["reasoning"] = {**reasoning}
        kwargs["reasoning"].setdefault("summary", "auto")
        try:
            resp = await self.client.responses.create(model=self.model, input=messages, **kwargs)
        except Exception as e:  # noqa: BLE001 - a deployment that rejects `summary` gets one retry without it
            if "summary" in str(e).lower() and "summary" in kwargs["reasoning"]:
                kwargs["reasoning"] = {k: v for k, v in kwargs["reasoning"].items() if k != "summary"}
                if not kwargs["reasoning"]:
                    del kwargs["reasoning"]
                resp = await self.client.responses.create(model=self.model, input=messages, **kwargs)
            else:
                raise
        try:
            self.last_raw = resp.model_dump(mode="json")
        except Exception:  # noqa: BLE001
            self.last_raw = {"repr": repr(resp)}
        return resp.output_text or ""
