from __future__ import annotations
from typing import Any, Dict, List, Tuple, Optional
import asyncio
from PIL import Image
from vagen.evaluate.adapters.base_adapter import ModelAdapter
from vagen.evaluate.utils.mm_utils import pil_to_dataurl_png, compile_text_images_for_order, parse_data_url


class GeminiAdapter(ModelAdapter):
    def __init__(self, client, model: str):
        self.client = client
        self.model = model

    def is_retryable_error(self, exc: Any):
        """Custom retry logic for Gemini API errors."""
        # Always retry ResourceExhausted (rate limit/quota exceeded)
        if exc.__class__.__name__ == "ResourceExhausted":
            return True

        # Always retry server errors (5xx)
        code = getattr(exc, "code", None)
        if isinstance(code, int) and 500 <= code < 600:
            return True

        # For other errors, use default logic
        return None

    def _segments_to_oai_content(self, segs):
        content = []
        for kind, val in segs:
            if kind == "text":
                if str(val).strip():
                    content.append({"type": "text", "text": str(val)})
            else:
                content.append({"type": "image_url", "image_url": {"url": pil_to_dataurl_png(val)}})
        return content

    def format_system(self, text: str, images: List[Image.Image]):
        segs = compile_text_images_for_order(text, images)
        return {"role": "system", "content": self._segments_to_oai_content(segs)}

    def format_user_turn(self, text: str, images: List[Image.Image]):
        segs = compile_text_images_for_order(text, images)
        return {"role": "user", "content": self._segments_to_oai_content(segs)}

    def _to_gemini_contents(self, messages: List[Dict[str, Any]]):
        system_texts, contents = [], []
        for m in messages:
            role, content = m.get("role"), m.get("content", [])
            if role == "system":
                if isinstance(content, list):
                    txt = " ".join(p.get("text", "") for p in content if p.get("type") == "text").strip()
                else:
                    txt = str(content).strip()
                if txt:
                    system_texts.append(txt)
                continue
            parts_list = []
            if not isinstance(content, list):
                content = [{"type": "text", "text": str(content)}]
            for p in content:
                if p.get("type") == "text":
                    t = p.get("text", "")
                    if str(t).strip():
                        parts_list.append({"text": str(t)})
                elif p.get("type") == "image_url":
                    url = p.get("image_url", {}).get("url", "")
                    parsed = parse_data_url(url)
                    if parsed is None:
                        continue
                    mime, b64 = parsed
                    parts_list.append({"inline_data": {"mime_type": mime, "data": b64}})
            gem_role = "user" if role == "user" else "model"
            contents.append({"role": gem_role, "parts": parts_list})
        system_instruction = "\n".join(system_texts) if system_texts else None
        return system_instruction, contents

    @staticmethod
    def _extract_text_from_response(resp: Any) -> str:
        out = []
        candidates = getattr(resp, "candidates", None) or []
        for cand in candidates:
            fr = getattr(cand, "finish_reason", None) or getattr(cand, "finishReason", None)
            content = getattr(cand, "content", None)
            parts = getattr(content, "parts", None) if content is not None else None
            if parts:
                for part in parts:
                    t = getattr(part, "text", None)
                    if isinstance(t, str) and t.strip():
                        out.append(t)
            if fr in (2, "MAX_TOKENS"):
                out.append("[Truncated: reached max_output_tokens]")
        return "\n".join([t for t in out if t.strip()])

    async def acompletion(self, messages: List[Dict[str, Any]], **chat_config: Any) -> str:
        genai = self.client.genai
        system_instruction, contents = self._to_gemini_contents(messages)

        # Pass system_instruction when constructing the model (not in generate_content)
        if system_instruction:
            model = genai.GenerativeModel(self.model, system_instruction=system_instruction)
        else:
            model = genai.GenerativeModel(self.model)

        def _call():
            return model.generate_content(
                contents=contents,
                **chat_config
            )

        resp = await asyncio.to_thread(_call)
        try:
            quick = getattr(resp, "text", None)
            if isinstance(quick, str) and quick.strip():
                return quick
        except Exception:
            pass
        text = self._extract_text_from_response(resp)
        return text or ""
