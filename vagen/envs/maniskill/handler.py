"""
Handler for the ManiSkill environment.

ManiSkill uses GPU rendering (render_backend="gpu"), so envs need
to be created on the server side. The handler is straightforward —
just creates ManiSkillEnv instances (no shared models to manage,
unlike SVG's DINO/DreamSim).
"""

from __future__ import annotations

import logging
from typing import Any, Dict

from vagen.envs_remote.handler import BaseGymHandler
from .maniskill_env import ManiSkillEnv

LOGGER = logging.getLogger(__name__)


class ManiSkillHandler(BaseGymHandler):
    """
    Handler for ManiSkill environments.

    Usage:
        handler = ManiSkillHandler()
        app = GymService(handler).build()
    """

    def __init__(
        self,
        session_timeout: float = 3600.0,
        max_sessions: int = 256,
    ):
        super().__init__(session_timeout=session_timeout, max_sessions=max_sessions)

    async def create_env(self, env_config: Dict[str, Any]) -> ManiSkillEnv:
        return ManiSkillEnv(env_config)
