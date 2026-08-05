"""Publishing validation episodes to wandb, off the training thread.

Naming, since the three levels are easy to conflate:

* **episode** -- one whole agent/environment interaction, start to terminal or turn cap.
  Identified by ``(group_idx, traj_idx)``.
* **conversation** -- one continuous exchange with the model. An episode is usually one;
  compaction ends the current conversation and opens the next, so an episode can span
  several. This is the level "trajectory" would name, except that ``traj_idx`` already
  means something else upstream -- *which sampled rollout of a prompt* -- so reusing the
  word would collide with the axis that identifies the episode in the first place.
* **turn** -- one model call inside a conversation.

The table is one row per validation step, with N episodes side by side, so scrubbing the
step slider plays the same slots forward through training.

Encoding a table of transcripts costs real time -- base64 for every frame of every turn
-- and it happens on the driver, between two training steps. It runs in a Ray actor on
one CPU instead, and the training loop never waits for the upload.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__file__)

# What the table shows per episode. Keep in step with `_row_for`.
_PER_EPISODE = ("rollout", "success", "score", "turns")


def build_table(episodes: list[dict], step: int, previous=None):
    """One row: this step, and the episodes chosen for it.

    ``previous`` is the table logged last time. wandb tables are immutable once logged,
    so history is carried by rebuilding from the old rows -- the documented workaround.
    """
    import wandb

    n = len(episodes)
    columns = ["step"] + [f"ep{i}_{f}" for i in range(n) for f in _PER_EPISODE]
    table = wandb.Table(columns=columns, data=list(previous.data) if previous is not None else [])

    row: list[Any] = [step]
    for e in episodes:
        row += [
            wandb.Html(e["html"]),
            e.get("success"),
            e.get("score"),
            e.get("turns"),
        ]
    table.add_data(*row)
    return table


class _Uploader:
    """Holds the growing table between steps and does the encoding."""

    def __init__(self):
        self._table = None

    def log(self, episodes: list[dict], step: int) -> int:
        import wandb

        if wandb.run is None:  # nothing to publish into
            return 0
        # Column count is fixed by the first row, so a step that happened to select
        # fewer episodes cannot be appended to a wider table. Log it on its own key
        # rather than dropping it or crashing the run over a logging concern.
        if self._table is not None and len(self._table.columns) != 1 + len(episodes) * len(_PER_EPISODE):
            wandb.log({f"val/episodes_{len(episodes)}": build_table(episodes, step)}, step=step)
            return len(episodes)
        self._table = build_table(episodes, step, previous=self._table)
        wandb.log({"val/episodes": self._table}, step=step)
        return len(episodes)


def make_logger(use_ray: bool = True):
    """An episode logger. A Ray actor when Ray is up, otherwise a plain in-process one.

    The actor gets one CPU and no GPU: this is base64 and JSON, and it must not be
    scheduled against the workers that are training.
    """
    if not use_ray:
        return _Uploader()
    try:
        import ray

        if not ray.is_initialized():
            return _Uploader()
        return ray.remote(num_cpus=1)(_Uploader).remote()
    except Exception as exc:  # noqa: BLE001 - logging must never take the run down
        logger.warning("episode logger falling back to in-process: %s", exc)
        return _Uploader()


def log_episodes(handle, episodes: list[dict], step: int, pending: list):
    """Hand the episodes off. Returns without waiting for the upload.

    ``pending`` accumulates in-flight futures. One is drained per call rather than all
    of them, so a slow upload cannot pile up unboundedly and a fast one costs nothing.
    """
    if not episodes:
        return
    if not hasattr(handle, "log") or not hasattr(handle.log, "remote"):
        handle.log(episodes, step)  # in-process fallback
        return

    import ray

    pending.append(handle.log.remote(episodes, step))
    if len(pending) > 2:
        done, rest = ray.wait(pending, num_returns=1)
        try:
            ray.get(done)  # surface a failure here rather than never
        except Exception as exc:  # noqa: BLE001
            logger.warning("episode upload failed: %s", exc)
        pending[:] = rest
