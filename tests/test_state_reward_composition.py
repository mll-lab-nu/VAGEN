"""The adapter and the reward wrapper have to agree, stacked as they actually stack.

Each side had tests and each side passed. The wrapper returned five values because that
is what ``BaseEnv`` specifies; the adapter unpacked four because that is what a gym env
returns -- and the adapter is what implements ``BaseEnv``, with the wrapper underneath
it. Composed, every step raised, 1006 turns in a row, and it reached the cluster looking
like a 3B model that could not earn reward under a new output format.

Nothing here needs a GPU, a judge, or a real environment. The bug was in the seam, and
the seam is free to test.
"""

from __future__ import annotations

import pytest

from vagen.agent_loop.gym_loop import GymEnvAdapter, _accepts_response
from vagen.rewards.state_reward import StateRewardSpec, StateRewardWrapper


class _Env:
    """A plain gym environment: four values out, no interest in the response."""

    def __init__(self):
        self.steps = 0

    async def reset(self, seed=None):
        return {"obs_str": "start"}, {}

    async def system_prompt(self):
        return {"obs_str": "rules"}

    async def step(self, action):
        self.steps += 1
        return {"obs_str": "next"}, 1.0, False, {"success": False}

    async def close(self):
        pass


class _Judge:
    """Returns one item per prompt so the F1 is well defined without a server."""

    def __init__(self):
        self.calls = 0

    async def parse_batch(self, prompts):
        self.calls += 1
        return [
            [{"object_id": "box", "vertical_relation": "below", "horizontal_relation": "same"}]
            for _ in prompts
        ]


def _spec():
    return StateRewardSpec(
        relations=lambda env: [
            {"object_id": "box", "vertical_relation": "below", "horizontal_relation": "same"}
        ],
        judge_prompt="structure this: {content}",
        object_weights={"box": 1.0},
        axes="axes",
        examples={"state_estimation": "<observation>...</observation>"},
    )


class _Tok:
    def encode(self, text, add_special_tokens=False):
        return [ord(c) % 97 for c in text]

    def decode(self, ids, **kw):
        return "".join(chr(i + 97) for i in ids)


def _stack():
    """GymEnvAdapter over StateRewardWrapper over a gym env -- the real arrangement."""
    inner = _Env()
    wrapped = StateRewardWrapper(
        env=inner, spec=_spec(), judge=_Judge(), enabled={"state_estimation": 0.5}, format_reward=0.1
    )
    return inner, wrapped, GymEnvAdapter(env=wrapped, env_name="Sokoban", kwargs={})


@pytest.mark.asyncio
async def test_a_step_through_the_real_stack_does_not_blow_up():
    inner, _, adapter = _stack()
    result = await adapter.step("<observation>[]</observation><answer>Up</answer>")

    assert len(result) == 5, "adapter must still return BaseEnv's five values"
    assert inner.steps == 1, "the inner env never ran"
    _obs, _reward, _term, _trunc, info = result
    assert not info.get("env_error"), f"the step raised and was swallowed: {info}"


@pytest.mark.asyncio
async def test_the_error_path_is_not_how_a_good_action_ends():
    """The regression's signature: a valid action reported as an environment error.

    The adapter catches everything so one bad action cannot kill a batch, which is right
    -- and which is also why an arity mismatch presented as zero reward rather than as a
    crash. This asserts the catch is not firing on the happy path.
    """
    _, _, adapter = _stack()
    _obs, reward, terminated, _trunc, info = await adapter.step("<answer>Up</answer>")
    assert info.get("env_error") is None
    assert not terminated, "a good action ended the episode"
    assert reward != 0.0, "a good action earned nothing"


@pytest.mark.asyncio
async def test_the_response_reaches_the_wrapper_so_scores_land_on_tokens():
    """Forwarding is the difference between token-level credit and a scalar.

    Dropping these does not fail; it quietly pays one number at the last token, which is
    the thing the whole state reward exists to avoid.
    """
    _, _, adapter = _stack()
    ids = list(range(40))
    _obs, reward, *_ = await adapter.step(
        "<observation>[]</observation><answer>Up</answer>", response_token_ids=ids, tokenizer=_Tok()
    )
    assert isinstance(reward, list), "reward came back scalar; the response was not forwarded"
    assert len(reward) == len(ids), "reward vector does not line up with the response"


@pytest.mark.asyncio
async def test_a_plain_env_is_not_handed_arguments_it_never_asked_for():
    """The adapter also wraps bare gym envs, whose step takes only an action."""
    inner = _Env()
    adapter = GymEnvAdapter(env=inner, env_name="Sokoban", kwargs={})
    _obs, _reward, _term, _trunc, info = await adapter.step("Up", response_token_ids=[1, 2], tokenizer=_Tok())
    assert not info.get("env_error"), "forwarded kwargs to an env that cannot take them"
    assert inner.steps == 1


def test_the_forwarding_predicate_reads_the_signature():
    assert _accepts_response(StateRewardWrapper.step) is True
    assert _accepts_response(_Env.step) is False
    assert _accepts_response(None) is False       # never raise while deciding
