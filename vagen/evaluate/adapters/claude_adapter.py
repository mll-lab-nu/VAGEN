# All comments are in English.
from __future__ import annotations
from typing import Any, Dict, List, Tuple
from vagen.evaluate.adapters.base_adapter import ModelAdapter
from vagen.evaluate.utils.mm_utils import pil_to_dataurl_png, compile_text_images_for_order, parse_data_url
from vagen.evaluate.utils.adapter_utils import filter_chat_kwargs
from PIL import Image

class ClaudeAdapter(ModelAdapter):
    """
    Anthropic Claude Messages API adapter.
    - Convert our OpenAI-like message list into Anthropic's schema.
    - Use top-level 'system' string when present.
    - Images are passed as {"type":"image","source":{"type":"base64","media_type":...,"data":...}}
    """

    def __init__(
        self,
        client,                 # anthropic.AsyncAnthropic
        model: str,             # e.g., "claude-3-5-sonnet-latest"
    ):
        self.client = client
        self.model = model

    def _segments_to_oai_content(self, segs: List[Tuple[str, Any]]) -> List[Dict[str, Any]]:
        """Build OpenAI-like content parts with image_url data URL for re-use."""
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
        return {"role": "system", "content": self._segments_to_oai_content(segs)}

    def format_user_turn(self, text: str, images: List[Image.Image]) -> Dict[str, Any]:
        segs = compile_text_images_for_order(text, images)
        return {"role": "user", "content": self._segments_to_oai_content(segs)}

    def _to_anthropic_parts(self, content_parts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Convert our content parts into Anthropic 'content' parts."""
        out: List[Dict[str, Any]] = []
        for p in content_parts:
            if p.get("type") == "text":
                t = p.get("text", "")
                if str(t).strip():
                    out.append({"type": "text", "text": str(t)})
            elif p.get("type") == "image_url":
                url = p.get("image_url", {}).get("url", "")
                parsed = parse_data_url(url)
                if parsed is None:
                    # Skip non-data-url images for safety
                    continue
                mime, b64 = parsed
                out.append({
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": mime,
                        "data": b64,
                    }
                })
        return out

    def _to_anthropic_messages(self, messages: List[Dict[str, Any]]) -> tuple[str | None, List[Dict[str, Any]]]:
        """Split system text and convert remaining messages to Anthropic format."""
        system_texts: List[str] = []
        converted: List[Dict[str, Any]] = []
        for m in messages:
            role = m.get("role")
            content = m.get("content", [])
            if role == "system":
                # Concatenate all text parts to system
                if isinstance(content, list):
                    txt = " ".join(p.get("text", "") for p in content if p.get("type") == "text")
                else:
                    txt = str(content)
                if txt.strip():
                    system_texts.append(txt.strip())
                continue

            # user / assistant
            parts = content if isinstance(content, list) else [{"type": "text", "text": str(content)}]
            converted.append({
                "role": "user" if role == "user" else "assistant",
                "content": self._to_anthropic_parts(parts),
            })
        system = "\n".join(system_texts) if system_texts else None
        return system, converted

    async def acompletion(self, messages: List[Dict[str, Any]], **chat_config: Any) -> str:
        system, msgs = self._to_anthropic_messages(messages)

        max_tokens = chat_config.pop("max_tokens", None)
        if max_tokens is None:
            max_tokens = chat_config.pop("max_output_tokens", None)
        if max_tokens is None:
            max_tokens = 512

        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": msgs,
            "max_tokens": int(max_tokens),
        }
        if system:
            payload["system"] = system

        payload.update(filter_chat_kwargs(chat_config))

        resp = await self.client.messages.create(**payload)
        # Concatenate all text blocks
        texts: List[str] = []
        for block in getattr(resp, "content", []) or []:
            if getattr(block, "type", None) == "text":
                texts.append(getattr(block, "text", "") or "")
        return "\n".join([t for t in texts if t.strip()])
