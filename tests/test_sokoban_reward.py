"""Sokoban uses a sparse success reward and ignores gym shaping."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from vagen.envs.sokoban.sokoban_env import Sokoban


ACTION = (
    "<perception>state</perception><reasoning>move</reasoning>"
    "<prediction>next</prediction><answer>Right</answer>"
)


class _Gym:
    def __init__(self):
        self.player_position = np.array([1, 1])
        self.boxes_on_target = 0
        self.num_boxes = 1

    def step(self, _action):
        self.player_position = np.array([1, 2])
        self.boxes_on_target = 1
        return None, 10.9, True, {}


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
