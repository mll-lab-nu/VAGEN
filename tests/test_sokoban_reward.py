"""Sokoban uses a sparse success reward and ignores gym shaping."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from vagen.envs.sokoban.sokoban_env import Sokoban, SokobanEnvConfig


ACTION = (
    "<perception>state</perception><reasoning>move</reasoning>"
    "<prediction>next</prediction><answer>Right</answer>"
)


class _Gym:
    def __init__(self):
        self.player_position = np.array([1, 1])
        self.boxes_on_target = 0
        self.num_boxes = 1
        self.steps = 0

    def step(self, _action):
        self.steps += 1
        self.player_position = np.array([1, 2])
        self.boxes_on_target = 1
        return None, 10.9, True, {}


def test_numeric_reward_config_accepts_environment_resolver_strings():
    config = SokobanEnvConfig(format_reward="0.03", success_reward="1.0")

    assert config.format_reward == pytest.approx(0.03)
    assert config.success_reward == pytest.approx(1.0)


@pytest.mark.asyncio
async def test_native_gym_shaping_is_ignored_and_success_is_paid_once():
    env = object.__new__(Sokoban)
    env.config = SimpleNamespace(
        action_sep=",",
        max_actions_per_step=3,
        prompt_format="wm",
        strict_format=True,
        success_reward=1.0,
        format_reward=0.5,
    )
    env.env = _Gym()
    env.total_reward = 0.0
    env.valid_actions = []

    async def render(init_obs=False):
        return {"obs_str": str(init_obs)}

    env._render_async = render

    _, reward, done, _ = await env.step(ACTION)

    assert done is True
    assert reward == pytest.approx(1.5)


@pytest.mark.asyncio
async def test_over_budget_wm_answer_is_not_executed_or_rewarded():
    env = object.__new__(Sokoban)
    env.config = SimpleNamespace(
        action_sep=",",
        max_actions_per_step=3,
        prompt_format="wm",
        strict_format=True,
        success_reward=1.0,
        format_reward=0.5,
    )
    env.env = _Gym()
    env.total_reward = 0.0
    env.valid_actions = []

    async def render(init_obs=False):
        return {"obs_str": str(init_obs)}

    env._render_async = render
    over_budget = ACTION.replace(
        "<answer>Right</answer>", "<answer>Right,Up,Left,Down</answer>"
    )

    _, reward, done, info = await env.step(over_budget)

    assert env.env.steps == 0
    assert done is False
    assert reward == 0.0
    assert info["format_correct"] is False
