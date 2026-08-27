"""The frames the rollout saw must also reach the model being optimised.

`AgentLoopOutput.multi_modal_data` is a plain dict, so a key the consumer does not read
is not an error -- it is silence. With the wrong key the processor is handed no images,
`multi_modal_inputs` comes back empty, and the actor/ref/critic forward passes run on
image-pad tokens with no vision features and text position ids. Qwen skips its
masked_scatter when pixel_values is absent, so nothing raises and the policy gradient is
computed on a blind model while the rollout sees the pictures.
"""

from __future__ import annotations

import inspect

import pytest


def _emitted_key():
    from vagen.training.agent_loop import gym_loop

    src = inspect.getsource(gym_loop.GymLoop._outputs)
    keys = {k for k in ("images", "image") if f'multi_modal_data={{"{k}"' in src}
    return keys


def _consumed_keys():
    from verl.experimental.agent_loop.agent_loop import AgentLoopWorker

    src = inspect.getsource(AgentLoopWorker._compute_multi_modal_inputs)
    return {m for m in ("images", "image", "videos", "audios")
            if f'multi_modal_data.get("{m}")' in src}


def test_the_loop_emits_the_key_the_consumer_reads():
    emitted, consumed = _emitted_key(), _consumed_keys()
    assert emitted, "the loop stopped publishing multi_modal_data at all"
    assert emitted <= consumed, (
        f"the loop emits {emitted} but the consumer reads {consumed}; "
        f"an unread key means the model trains without the images"
    )


def test_the_consumer_still_reads_images_plural():
    """If upstream renames it again this fails here rather than in a silent blind run."""
    assert "images" in _consumed_keys()


def test_the_guard_fails_when_the_emitted_key_is_wrong():
    """The guard is worth nothing unless a wrong key actually fails it. Parametrising
    over names and asserting they are absent restates the premise instead."""
    emitted, consumed = {"image"}, {"images", "videos", "audios"}
    assert not emitted <= consumed, "a wrong key would pass the comparison the guard uses"
    assert {"images"} <= consumed, "the right key would fail it"
