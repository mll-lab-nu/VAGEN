"""
Handler for the SVG environment.

Key design: DINO and DreamSim models are loaded once in the handler
and shared across all SVGEnv sessions, avoiding per-instance GPU waste.
The dataset is also loaded once and shared.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from vagen.envs_remote.handler import BaseGymHandler
from .svg_env import SVGEnv
from .svg_config import SvgEnvConfig
from .svg_utils import load_svg_dataset

LOGGER = logging.getLogger(__name__)


class SVGHandler(BaseGymHandler):
    """
    Handler with shared DINO/DreamSim models and dataset.

    Usage:
        handler = SVGHandler(
            dino_device="cuda:0",
            dreamsim_device="cuda:0",
            model_size="small",
        )
        app = GymService(handler).build()
    """

    def __init__(
        self,
        session_timeout: float = 3600.0,
        max_sessions: int = 0,
        # Model config
        model_size: str = "small",
        dino_device: str = "cuda:0",
        dreamsim_device: str = "cuda:0",
        preload_models: bool = True,
        # Dataset config
        dataset_name: str = "starvector/svg-icons-simple",
        data_dir: str = "data",
        split: str = "train",
    ):
        super().__init__(session_timeout=session_timeout, max_sessions=max_sessions)
        self._model_size = model_size
        self._dino_device = dino_device
        self._dreamsim_device = dreamsim_device

        # Lazy-loaded shared resources
        self._dino_model = None
        self._dreamsim_model = None
        self._dataset = None

        self._dataset_name = dataset_name
        self._data_dir = data_dir
        self._split = split

        if preload_models:
            self._ensure_models()
            self._ensure_dataset()

    # ------------------------------------------------------------------
    # Lazy init
    # ------------------------------------------------------------------

    def _ensure_models(self):
        if self._dino_model is None:
            LOGGER.info(f"Loading DINO ({self._model_size}) on {self._dino_device} ...")
            from .dino import DINOScoreCalculator
            self._dino_model = DINOScoreCalculator(
                model_size=self._model_size, device=self._dino_device
            )
            LOGGER.info("DINO loaded.")

        if self._dreamsim_model is None:
            LOGGER.info(f"Loading DreamSim on {self._dreamsim_device} ...")
            from .dreamsim import DreamSimScoreCalculator
            self._dreamsim_model = DreamSimScoreCalculator(device=self._dreamsim_device)
            LOGGER.info("DreamSim loaded.")

    def _ensure_dataset(self):
        if self._dataset is None:
            LOGGER.info(f"Loading dataset {self._dataset_name} ({self._split}) ...")
            self._dataset = load_svg_dataset(
                self._data_dir, self._dataset_name, self._split
            )
            LOGGER.info(f"Dataset loaded: {len(self._dataset)} samples.")

    # ------------------------------------------------------------------
    # BaseGymHandler abstract method
    # ------------------------------------------------------------------

    async def create_env(self, env_config: Dict[str, Any]) -> SVGEnv:
        """Create an SVGEnv with shared models and dataset injected."""
        self._ensure_models()
        self._ensure_dataset()
        return SVGEnv(
            env_config=env_config,
            dataset=self._dataset,
            dino_model=self._dino_model,
            dreamsim_model=self._dreamsim_model,
        )

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    def get_session_stats(self) -> Dict[str, Any]:
        stats = super().get_session_stats()
        stats["model_size"] = self._model_size
        stats["dino_device"] = self._dino_device
        stats["dreamsim_device"] = self._dreamsim_device
        stats["dataset_size"] = len(self._dataset) if self._dataset else 0
        return stats
