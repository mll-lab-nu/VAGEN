"""InternVL model-family adapter.

InternVL3/3.5 Hugging Face checkpoints use the same ChatML edge framing as Qwen, including
the generated ``<|im_end|>`` token followed by a template-owned newline. Reusing that
tested implementation is intentional protocol sharing; registration remains family-
specific so InternVL can diverge without changing VERL transport code.
"""

from vagen.models.qwen.qwen import QwenModelAdapter


class InternVLModelAdapter(QwenModelAdapter):
    """Render InternVL's ChatML-compatible conversation edges."""


__all__ = ["InternVLModelAdapter"]
