"""Utility functions for handling image tokens in text."""

import re
import warnings
from typing import Union, Optional


# Supported model patterns for image token replacement
SUPPORTED_PATTERNS = {
    "qwen2_vl": {
        # Qwen2-VL / Qwen2.5-VL: <|vision_start|><|image_pad|>...<|image_pad|><|vision_end|>
        "pattern": r"<\|vision_start\|>(<\|image_pad\|>)+<\|vision_end\|>",
        "processor_check": lambda p: p is not None and hasattr(p, 'image_processor') and "Qwen2VL" in p.image_processor.__class__.__name__,
    },
    "qwen_vl_legacy": {
        # Fallback pattern: consecutive <|image_pad|> without vision tags
        "pattern": r"(<\|image_pad\|>)+",
        "processor_check": lambda p: False,  # Only used as fallback
    },
    "imgpad": {
        # Alternative: <|imgpad|> tokens
        "pattern": r"(<\|imgpad\|>)+",
        "processor_check": lambda p: False,  # Only used as fallback
    },
}


def get_model_type(processor) -> Optional[str]:
    """Detect the model type from processor.

    Args:
        processor: The processor object

    Returns:
        Model type string or None if not detected
    """
    if processor is None:
        return None

    if hasattr(processor, 'image_processor'):
        class_name = processor.image_processor.__class__.__name__
        if "Qwen2VL" in class_name:
            return "qwen2_vl"

    return None


def replace_image_tokens_for_logging(
    texts: Union[str, list[str]],
    processor=None,
    tokenizer=None,
    replacement: str = "<image>"
) -> Union[str, list[str]]:
    """Replace image token sequences with a single placeholder for cleaner logging.

    This function takes decoded text that may contain image token sequences
    (e.g., <|vision_start|><|image_pad|>...<|vision_end|> for Qwen2-VL) and
    replaces each sequence with a single <image> placeholder.

    Currently supported models:
    - Qwen2-VL / Qwen2.5-VL: <|vision_start|><|image_pad|>...<|vision_end|> -> <image>

    Args:
        texts: A single string or list of strings to process
        processor: Optional processor to detect model type
        tokenizer: Optional tokenizer (for future use)
        replacement: The string to replace image sequences with (default: "<image>")

    Returns:
        Processed string(s) with image token sequences replaced

    Example:
        >>> text = "Hello <|vision_start|><|image_pad|><|image_pad|><|vision_end|> world"
        >>> replace_image_tokens_for_logging(text)
        "Hello <image> world"
    """
    model_type = get_model_type(processor)

    # Determine which patterns to try
    patterns_to_try = []

    if model_type == "qwen2_vl":
        patterns_to_try.append(SUPPORTED_PATTERNS["qwen2_vl"]["pattern"])
    else:
        # For unknown models, try all patterns and warn
        warned = False
        for name, config in SUPPORTED_PATTERNS.items():
            patterns_to_try.append(config["pattern"])

        # Check if text contains any known image tokens to decide if warning is needed
        sample_text = texts if isinstance(texts, str) else (texts[0] if texts else "")
        has_image_tokens = any(
            re.search(config["pattern"], sample_text)
            for config in SUPPORTED_PATTERNS.values()
        )

        if has_image_tokens and processor is not None:
            warnings.warn(
                f"Could not detect model type from processor. "
                f"Image token replacement currently has full support for: Qwen2-VL, Qwen2.5-VL. "
                f"Attempting to use fallback patterns.",
                UserWarning
            )

    def process_single_text(text: str) -> str:
        result = text
        for pattern in patterns_to_try:
            result = re.sub(pattern, replacement, result)
        return result

    if isinstance(texts, str):
        return process_single_text(texts)
    else:
        return [process_single_text(t) for t in texts]



