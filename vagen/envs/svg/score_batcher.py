"""
Async score batcher for SVG environment.

Collects scoring requests from concurrent SVGEnv.step() calls and
batches them together for efficient GPU inference. Each request
submits (gt_image, gen_image, score_config) and awaits its result.

The batcher flushes when either:
  - batch reaches max_batch_size, or
  - max_wait_seconds elapsed since first pending request
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from PIL import Image

from .score import calculate_structural_accuracy

LOGGER = logging.getLogger(__name__)


@dataclass
class _ScoreRequest:
    gt_image: Image.Image
    gen_image: Image.Image
    score_config: Dict[str, Any]
    future: asyncio.Future


class ScoreBatcher:
    """
    Batches DINO/DreamSim scoring requests across concurrent sessions.

    Usage:
        batcher = ScoreBatcher(dino_model, dreamsim_model)
        # In SVGEnv.step():
        scores = await batcher.submit(gt_image, gen_image, score_config)
    """

    def __init__(
        self,
        dino_model=None,
        dreamsim_model=None,
        max_batch_size: int = 32,
        max_wait_seconds: float = 0.05,
    ):
        self._dino = dino_model
        self._dreamsim = dreamsim_model
        self._max_batch_size = max_batch_size
        self._max_wait_seconds = max_wait_seconds

        self._queue: List[_ScoreRequest] = []
        self._lock = asyncio.Lock()
        self._flush_task: Optional[asyncio.Task] = None
        self._first_enqueue_time: Optional[float] = None

    async def submit(
        self,
        gt_image: Image.Image,
        gen_image: Image.Image,
        score_config: Dict[str, Any],
    ) -> Dict[str, float]:
        """Submit a scoring request and await the result."""
        loop = asyncio.get_running_loop()
        future = loop.create_future()

        async with self._lock:
            self._queue.append(_ScoreRequest(
                gt_image=gt_image,
                gen_image=gen_image,
                score_config=score_config,
                future=future,
            ))

            if self._first_enqueue_time is None:
                self._first_enqueue_time = time.monotonic()

            # Flush immediately if batch is full
            if len(self._queue) >= self._max_batch_size:
                self._cancel_flush_timer()
                batch = self._take_batch()
            else:
                # Start timer if not already running
                if self._flush_task is None or self._flush_task.done():
                    self._flush_task = asyncio.create_task(self._flush_after_timeout())
                batch = None

        if batch is not None:
            await self._process_batch(batch)

        return await future

    async def _flush_after_timeout(self):
        """Wait up to max_wait_seconds then flush whatever is queued."""
        await asyncio.sleep(self._max_wait_seconds)
        async with self._lock:
            if not self._queue:
                return
            batch = self._take_batch()
        if batch:
            await self._process_batch(batch)

    def _take_batch(self) -> List[_ScoreRequest]:
        """Take all pending requests (must be called under lock)."""
        batch = list(self._queue)
        self._queue.clear()
        self._first_enqueue_time = None
        return batch

    def _cancel_flush_timer(self):
        if self._flush_task is not None and not self._flush_task.done():
            self._flush_task.cancel()
            self._flush_task = None

    async def _process_batch(self, batch: List[_ScoreRequest]):
        """Run batch scoring and resolve futures."""
        n = len(batch)
        LOGGER.debug(f"ScoreBatcher: processing batch of {n}")

        gt_images = [r.gt_image for r in batch]
        gen_images = [r.gen_image for r in batch]

        try:
            # Batch DINO scores
            dino_scores = await asyncio.to_thread(
                self._dino.calculate_batch_scores, gt_images, gen_images
            ) if self._dino is not None else [0.0] * n

            # Batch DreamSim scores
            dreamsim_scores = await asyncio.to_thread(
                self._dreamsim.calculate_batch_scores, gt_images, gen_images
            ) if self._dreamsim is not None else [0.0] * n

            # Structural scores (CPU, per-pair)
            structural_scores = await asyncio.to_thread(
                self._compute_structural_batch, gt_images, gen_images
            )

            # Assemble per-request results
            for i, req in enumerate(batch):
                weights = {
                    "dino": req.score_config.get("dino_weight", 0.0),
                    "structural": req.score_config.get("structural_weight", 0.0),
                    "dreamsim": req.score_config.get("dreamsim_weight", 0.0),
                }
                total = (
                    dino_scores[i] * weights["dino"]
                    + structural_scores[i] * weights["structural"]
                    + dreamsim_scores[i] * weights["dreamsim"]
                )
                result = {
                    "dino_score": dino_scores[i],
                    "structural_score": structural_scores[i],
                    "dreamsim_score": dreamsim_scores[i],
                    "total_score": max(0.0, total),
                }
                if not req.future.done():
                    req.future.set_result(result)

        except Exception as e:
            LOGGER.error(f"ScoreBatcher: batch scoring failed: {e}")
            for req in batch:
                if not req.future.done():
                    req.future.set_exception(e)

    @staticmethod
    def _compute_structural_batch(gt_images, gen_images) -> List[float]:
        return [
            max(0.0, float(calculate_structural_accuracy(gt, gen)))
            for gt, gen in zip(gt_images, gen_images)
        ]
