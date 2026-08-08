"""Where the local Qwen2.5-VL snapshot is, without hardcoding whose machine it is on.

The multimodal tests need a real processor -- the failures they exist for are invisible
against a fake tokenizer -- but a hardcoded absolute path both publishes the author's
home directory and makes the tests silently skip everywhere else. `hf_processor` returns
None rather than raising on a bad path, so a wrong path reads as "no processor available"
and the test passes by skipping.
"""

from __future__ import annotations

import glob
import os

MODEL_ID = "Qwen/Qwen2.5-VL-3B-Instruct"


def local_snapshot(model_id: str = MODEL_ID) -> str | None:
    """A downloaded snapshot of ``model_id``, or None if there is not one.

    ``VAGEN_TEST_MODEL`` wins, then the HF cache under ``HF_HOME``/``HF_HUB_CACHE``/the
    default. Returns None rather than a guess, so a caller can skip deliberately.
    """
    override = os.environ.get("VAGEN_TEST_MODEL")
    if override:
        return override
    cache = (os.environ.get("HF_HUB_CACHE")
             or os.path.join(os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface")), "hub"))
    hits = sorted(glob.glob(os.path.join(
        cache, "models--" + model_id.replace("/", "--"), "snapshots", "*")))
    return hits[-1] if hits else None
