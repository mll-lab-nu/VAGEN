"""
Multipart encoding/decoding utilities for gym environment communication.

Protocol:
- Request/Response use multipart/form-data or multipart/mixed
- JSON metadata + optional PIL images
- Reuses ViewSuite's proven multipart implementation
"""

from __future__ import annotations

import io
import json
import uuid
from typing import Any, Dict, List, Optional, Tuple

from PIL import Image

try:
    from fastapi import UploadFile
except Exception:
    UploadFile = Any  # type: ignore


# ---------------------------------------------------------------------
# Encoding (client -> server, server -> client)
# ---------------------------------------------------------------------
def encode_multipart(
    data: Dict[str, Any],
    images: Optional[List[Image.Image]] = None,
    *,
    image_format: str = "PNG",
    image_mime: str = "image/png",
    boundary_prefix: str = "gym_env_",
) -> Tuple[str, bytes]:
    """
    Encode (data, images) into multipart body.

    Args:
        data: JSON-serializable dict (required, can be empty)
        images: Optional list of PIL images
        image_format: PIL save format (PNG, JPEG, etc.)
        image_mime: MIME type for images
        boundary_prefix: Boundary prefix for multipart

    Returns:
        (boundary, body_bytes)
    """
    boundary = f"{boundary_prefix}{uuid.uuid4().hex}"
    crlf = b"\r\n"
    bnd = boundary.encode("utf-8")
    body = bytearray()

    # Data part (always present, even if empty)
    data_bytes = json.dumps(data or {}, ensure_ascii=False).encode("utf-8")
    body += b"--" + bnd + crlf
    body += b'Content-Disposition: form-data; name="data"; filename="data.json"' + crlf
    body += b"Content-Type: application/json; charset=utf-8" + crlf + crlf
    body += data_bytes + crlf

    # Image parts (optional)
    for i, img in enumerate(images or []):
        buf = io.BytesIO()
        img.save(buf, format=image_format)
        img_bytes = buf.getvalue()

        body += b"--" + bnd + crlf
        body += f'Content-Disposition: form-data; name="images"; filename="{i}.png"'.encode("utf-8") + crlf
        body += f"Content-Type: {image_mime}".encode("utf-8") + crlf + crlf
        body += img_bytes + crlf

    # End boundary
    body += b"--" + bnd + b"--" + crlf
    return boundary, bytes(body)


# ---------------------------------------------------------------------
# Decoding (server <- client, client <- server)
# ---------------------------------------------------------------------
def _extract_boundary(content_type: str) -> str:
    """Extract boundary from Content-Type header."""
    ct = content_type or ""
    parts = [p.strip() for p in ct.split(";")]
    for p in parts:
        if p.lower().startswith("boundary="):
            b = p.split("=", 1)[1].strip().strip('"')
            if b:
                return b
    raise ValueError(f"Missing boundary in Content-Type: {content_type}")


def decode_multipart(content_type: str, body: bytes) -> Tuple[Dict[str, Any], List[Image.Image]]:
    """
    Decode multipart body into (data, images).

    Args:
        content_type: Content-Type header with boundary
        body: Raw multipart body bytes

    Returns:
        (data_dict, image_list)
    """
    boundary = _extract_boundary(content_type)
    marker = ("--" + boundary).encode("utf-8")

    data: Dict[str, Any] = {}
    images: List[Image.Image] = []

    chunks = body.split(marker)
    for chunk in chunks:
        chunk = chunk.strip()
        if not chunk or chunk == b"--":
            continue

        if chunk.endswith(b"--"):
            chunk = chunk[:-2].strip()

        header_blob, _, payload = chunk.partition(b"\r\n\r\n")
        if not payload:
            continue

        payload = payload.rstrip(b"\r\n")

        # Parse headers
        headers = header_blob.decode("utf-8", errors="ignore").split("\r\n")
        part_type = ""
        for line in headers:
            if ":" in line:
                k, v = line.split(":", 1)
                if k.strip().lower() == "content-type":
                    part_type = v.strip().lower()

        # JSON data part
        if "application/json" in part_type:
            try:
                obj = json.loads(payload.decode("utf-8"))
                data = obj if isinstance(obj, dict) else {"_data": obj}
            except Exception:
                data = {"_data_raw": payload.decode("utf-8", errors="ignore")}

        # Image part
        elif part_type.startswith("image/"):
            img = Image.open(io.BytesIO(payload)).convert("RGBA")
            images.append(img)

    return data, images


# ---------------------------------------------------------------------
# Server-side form parsing helpers
# ---------------------------------------------------------------------
def parse_data_field(data_str: Optional[str]) -> Dict[str, Any]:
    """
    Parse 'data' form field (JSON string).

    Args:
        data_str: JSON string from form field

    Returns:
        Parsed dict (empty dict if None/invalid)
    """
    if not data_str:
        return {}
    try:
        obj = json.loads(data_str)
        if isinstance(obj, dict):
            return obj
        return {"_data": obj}
    except Exception:
        return {"_data_raw": data_str}


async def read_images(files: Optional[List[UploadFile]]) -> List[Image.Image]:
    """
    Read uploaded images from FastAPI UploadFile list.

    Args:
        files: List of uploaded files

    Returns:
        List of RGBA PIL images
    """
    if not files:
        return []
    imgs: List[Image.Image] = []
    for f in files:
        raw = await f.read()
        img = Image.open(io.BytesIO(raw)).convert("RGBA")
        imgs.append(img)
    return imgs
