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
        ("OpenGVLab/InternVL3-1B-hf", False, "<IMG_CONTEXT>"),
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
        multi_modal_inputs={"image_grid_thw": None, "video_grid_thw": None},
    )

    expected = (1, 4, 4) if expects_mrope else (1, 4)
    assert tuple(position_ids.shape) == expected


def test_image_token_is_discovered_per_family(repo, expects_mrope, expected_token):
    """Three families, three different placeholder strings -- which is why this is read
    off the processor instead of matched against a table."""
    from vagen.utils.image_token_utils import get_image_token, replace_image_tokens_for_logging

    processor = _processor(repo)
    token = get_image_token(processor)

    assert token == expected_token
    assert replace_image_tokens_for_logging(f"a {token}{token}{token} b", processor) == "a <image> b"


def test_gym_loops_forward_mm_processor_kwargs(repo, expects_mrope, expected_token):
    """★ The prompt is tokenized by the agent loop but the images are re-processed by
    the inference engine, and both must agree on how an image is tiled. verl threads
    data.mm_processor_kwargs to the engine; a loop that drops it on the tokenizing side
    leaves the two disagreeing, which surfaces as a CUDA assert deep in the engine
    rather than a config error."""
    import inspect

    from vagen.agent_loop import gym_agent_loop, gym_agent_loop_no_concat

    for module in (gym_agent_loop, gym_agent_loop_no_concat):
        src = inspect.getsource(module)
        calls = src.count("self.processor(")
        forwards = src.count("**self._get_mm_processor_kwargs()")
        assert calls == forwards, f"{module.__name__}: {calls} processor calls, {forwards} forward the kwargs"
