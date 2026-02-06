# All comments are in English.
from __future__ import annotations
from typing import Any, Dict, Iterable, Optional

def filter_chat_kwargs(
    chat_config: Optional[Dict[str, Any]] = None,
    *,
    unsupported: Optional[Iterable[str]] = None,
) -> Dict[str, Any]:
    """Remove None-valued or explicitly unsupported chat kwargs."""
    if not chat_config:
        return {}
    blocked = set(unsupported or [])
    return {k: v for k, v in chat_config.items() if v is not None and k not in blocked}
