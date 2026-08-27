"""Real processors from three model families, not mocks.

The mrope fallback and the processor-driven token lookup are both claims about what
real ``AutoProcessor`` objects look like, and a hand-written stub can only confirm the
shape I already assumed. These load the actual processor configs -- a few MB, no model
weights -- and check the two behaviours that differ per family.

Skipped when the configs are not cached, so an offline run does not fail.
"""

import pytest
import torch

pytestmark = pytest.mark.parametrize(
    "repo,expects_mrope,expected_token",
    [
        ("Qwen/Qwen2.5-VL-3B-Instruct", True, "<|image_pad|>"),
        ("Qwen/Qwen3-VL-4B-Instruct", True, "<|image_pad|>"),
        ("Qwen/Qwen3.5-4B", True, "<|image_pad|>"),
        ("OpenGVLab/InternVL3-1B-hf", False, "<IMG_CONTEXT>"),
        ("OpenGVLab/InternVL3_5-2B-hf", False, "<IMG_CONTEXT>"),
        ("zai-org/GLM-4.6V-Flash", True, "<|image|>"),
        ("llava-hf/llava-1.5-7b-hf", False, "<image>"),
    ],
)


def _processor(repo):
    from verl.utils import hf_processor

    try:
        processor = hf_processor(repo, trust_remote_code=True)
    except Exception as exc:  # noqa: BLE001 - offline or missing cache
        pytest.skip(f"{repo} unavailable: {exc}")
    if processor is None:
        pytest.fail(f"{repo} produced no processor; hf_processor swallowed something")
    return processor


def test_processor_loads_for_every_family(repo, expects_mrope, expected_token):
    """★ Non-mrope processors used to raise, which hf_processor caught and turned into
    None -- indistinguishable from a text-only model."""
    processor = _processor(repo)

    assert hasattr(processor, "get_rope_index") is expects_mrope


def test_position_ids_match_what_the_family_needs(repo, expects_mrope, expected_token):
    """★ mrope models must keep their 4-row position ids; the fallback for everyone
    else must not quietly swallow that."""
    from verl.experimental.agent_loop.agent_loop import AgentLoopWorker

    worker = AgentLoopWorker.__new__(AgentLoopWorker)
    worker.processor = _processor(repo)

    position_ids = worker._compute_position_ids(
        input_ids=torch.tensor([[5, 6, 7, 0]]),
        attention_mask=torch.tensor([[1, 1, 1, 0]]),
        # `mm_token_type_ids` because transformers>=5.3 made it a required argument of
        # get_rope_index, and the agent loop only builds one when the processor emitted
        # one. The real Qwen processor does emit it, so training is fine; a hand-built
        # dict that omits it takes a branch no real run takes.
        multi_modal_inputs={"image_grid_thw": None, "video_grid_thw": None,
                            "mm_token_type_ids": torch.zeros(1, 4, dtype=torch.long)},
    )

    expected = (1, 4, 4) if expects_mrope else (1, 4)
    assert tuple(position_ids.shape) == expected


def test_image_token_is_discovered_per_family(repo, expects_mrope, expected_token):
    """Three families, three different placeholder strings -- which is why this is read
    off the processor instead of matched against a table."""
    from vagen.models import get_image_token, replace_image_tokens_for_logging

    processor = _processor(repo)
    token = get_image_token(processor)

    assert token == expected_token
    assert replace_image_tokens_for_logging(f"a {token}{token}{token} b", processor) == "a <image> b"


def test_the_client_forwards_mm_processor_kwargs_both_ways(repo, expects_mrope, expected_token):
    """★ The prompt is tokenized on one side and the images are re-processed by the
    engine on the other, and the two must agree on how an image is tiled. Setting the
    knob on one side only guarantees the mismatch, which surfaces as a CUDA assert deep
    in the engine rather than as a configuration error.

    Checked on the parsed call, so a comment mentioning the name cannot stand in for the
    argument.
    """
    import ast
    import inspect

    from vagen.training.agent_loop.verl_client import VerlClient

    source = inspect.getsource(VerlClient)
    tree = ast.parse(source)

    # tokenizing side: every processor call carries them
    processor_calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr in {"apply_chat_template"}
    ]
    assert processor_calls, "no chat-template call found"
    for call in processor_calls:
        assert any(kw.arg is None for kw in call.keywords), "a template call drops the kwargs"

    # engine side: the generate call names them
    generates = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "generate"
    ]
    assert generates, "no generate() call found"
    for call in generates:
        assert "mm_processor_kwargs" in {kw.arg for kw in call.keywords}, (
            "generate() does not forward the tiling settings"
        )
