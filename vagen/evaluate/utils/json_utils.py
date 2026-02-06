import json
from typing import Any, Union

JSONScalar = Union[str, int, float, bool, None]

def sanitize_for_json(
    obj: Any,
    *,
    replacement: JSONScalar = "not jsonable",
    coerce_keys_to_str: bool = True,
    drop_non_str_keys: bool = False,
    max_depth: int = 1000,
    _seen: set[int] | None = None,
    _depth: int = 0,
) -> Any:
    """
    Recursively clean an object so that `json.dumps` will not fail:
      - dict/list/tuple/set are traversed recursively.
      - Non-JSON-serializable leaves are replaced with `replacement`
        (default is "not jsonable", but you can set it to None).
      - dict keys:
          * If coerce_keys_to_str=True (default), non-string keys are converted to str.
          * If drop_non_str_keys=True, non-string keys are dropped.
          * If both False, the key becomes str(replacement).
      - Cyclic references are replaced with `replacement`.
      - `max_depth` prevents infinite recursion.

    Args:
        obj: Any Python object.
        replacement: Value to replace non-serializable leaves with.
        coerce_keys_to_str: Whether to convert non-string dict keys to str.
        drop_non_str_keys: Whether to drop non-string dict keys.
        max_depth: Maximum recursion depth.

    Returns:
        Cleaned object that can be passed to `json.dumps`.
    """
    if _seen is None:
        _seen = set()

    # Depth and cycle guard
    if _depth > max_depth:
        return replacement
    obj_id = id(obj)
    if isinstance(obj, (dict, list, tuple, set)) or hasattr(obj, "__dict__"):
        if obj_id in _seen:
            return replacement  # cycle detected
        _seen.add(obj_id)

    # JSON scalars
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        # Replace NaN and Inf (JSON standard does not allow them)
        if isinstance(obj, float) and (obj != obj or obj in (float("inf"), float("-inf"))):
            return replacement
        return obj

    # Dicts
    if isinstance(obj, dict):
        out: dict[str, Any] = {}
        for k, v in obj.items():
            # Handle keys
            if isinstance(k, str):
                key = k
            else:
                if coerce_keys_to_str:
                    try:
                        key = str(k)
                    except Exception:
                        key = str(replacement)
                elif drop_non_str_keys:
                    continue
                else:
                    key = str(replacement)
            out[key] = sanitize_for_json(
                v,
                replacement=replacement,
                coerce_keys_to_str=coerce_keys_to_str,
                drop_non_str_keys=drop_non_str_keys,
                max_depth=max_depth,
                _seen=_seen,
                _depth=_depth + 1,
            )
        return out

    # Lists, tuples, sets
    if isinstance(obj, (list, tuple, set)):
        return [
            sanitize_for_json(
                x,
                replacement=replacement,
                coerce_keys_to_str=coerce_keys_to_str,
                drop_non_str_keys=drop_non_str_keys,
                max_depth=max_depth,
                _seen=_seen,
                _depth=_depth + 1,
            )
            for x in obj
        ]

    # Objects with __dict__ → fallback to vars()
    if hasattr(obj, "__dict__"):
        try:
            return sanitize_for_json(
                vars(obj),
                replacement=replacement,
                coerce_keys_to_str=coerce_keys_to_str,
                drop_non_str_keys=drop_non_str_keys,
                max_depth=max_depth,
                _seen=_seen,
                _depth=_depth + 1,
            )
        except Exception:
            return replacement

    # Binary-like objects
    if isinstance(obj, (bytes, bytearray, memoryview)):
        return replacement

    # Special handling for numpy types (optional)
    try:
        import numpy as np
        if isinstance(obj, np.generic):
            return sanitize_for_json(
                obj.item(),
                replacement=replacement,
                coerce_keys_to_str=coerce_keys_to_str,
                drop_non_str_keys=drop_non_str_keys,
                max_depth=max_depth,
                _seen=_seen,
                _depth=_depth + 1,
            )
        if isinstance(obj, np.ndarray):
            return [
                sanitize_for_json(
                    x,
                    replacement=replacement,
                    coerce_keys_to_str=coerce_keys_to_str,
                    drop_non_str_keys=drop_non_str_keys,
                    max_depth=max_depth,
                    _seen=_seen,
                    _depth=_depth + 1,
                )
                for x in obj.tolist()
            ]
    except Exception:
        pass

    # Final fallback: test if it can be json.dumps'ed
    try:
        json.dumps(obj)
        return obj
    except Exception:
        return replacement


def safe_json_dumps(obj: Any, **sanitize_kwargs) -> str:
    """Convenience wrapper: clean object and dump to JSON string."""
    cleaned = sanitize_for_json(obj, **sanitize_kwargs)
    return json.dumps(cleaned, ensure_ascii=False)
