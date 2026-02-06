# All comments are in English.
from __future__ import annotations
import base64
import io
from datetime import datetime
from typing import Any, Dict, List, Tuple, Optional
from PIL import Image

IMAGE_PLACEHOLDER = "<image>"

def pil_to_dataurl_png(img: Image.Image) -> str:
    """Encode a PIL image as a data URL (PNG)."""
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/png;base64,{b64}"

def parse_data_url(data_url: str) -> Optional[Tuple[str, str]]:
    """
    Parse data URL. Return (mime_type, base64_string) or None if not data URL.
    Example: 'data:image/png;base64,AAAA...'
    """
    if not isinstance(data_url, str) or not data_url.startswith("data:"):
        return None
    try:
        header, b64 = data_url.split(",", 1)
        # header like: data:image/png;base64
        mime = header.split(";")[0].split(":")[1]
        return mime, b64
    except Exception:
        return None

def compile_text_images_for_order(text: str, images: List[Image.Image], placeholder: str = IMAGE_PLACEHOLDER) -> List[Tuple[str, Any]]:
    """
    Split text by placeholder and interleave with images preserving order.
    Returns a list of segments: [("text", str) | ("image", PIL.Image)]
    """
    parts = text.split(placeholder)
    segs: List[Tuple[str, Any]] = []
    for i, p in enumerate(parts):
        if p:
            segs.append(("text", p))
        if i < len(parts) - 1:
            if i < len(images):
                segs.append(("image", images[i]))
            else:
                segs.append(("text", ""))
    if len(images) > max(0, len(parts) - 1):
        for img in images[len(parts) - 1:]:
            segs.append(("image", img))
    return segs

def extract_images(obs: Dict[str, Any]) -> List[Image.Image]:
    """
    Heuristically extract images from an observation dict.
    - Prefer obs["multi_modal_input"]["<image>"] if present.
    - Else obs["images"] if it's a list of PIL.Image.
    - Else return [].
    """
    try:
        mm = obs.get("multi_modal_input", {})
        imgs = mm.get(IMAGE_PLACEHOLDER, [])
        if isinstance(imgs, list) and imgs and isinstance(imgs[0], Image.Image):
            return imgs
    except Exception:
        pass
    imgs = obs.get("images")
    if isinstance(imgs, list) and (not imgs or isinstance(imgs[0], Image.Image)):
        return imgs
    return []

def _now_tag() -> str:
    """Return a compact timestamp tag."""
    return datetime.now().strftime("%Y%m%d-%H%M%S")
