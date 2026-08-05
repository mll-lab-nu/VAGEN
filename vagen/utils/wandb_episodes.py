"""Publishing validation episodes to wandb without stalling the training loop.

Naming, since the three levels are easy to conflate:

* **episode** -- one whole agent/environment interaction, ``(group_idx, traj_idx)``.
* **conversation** -- one continuous exchange with the model.
* **turn** -- one model call within a conversation.

The context policy decides the shape: concat is one conversation of many turns, no_concat
many conversations of one turn each, compact several conversations of many turns.

The split of work matters, and getting it wrong is silent. Rendering -- base64 for every
frame of every turn -- runs in a Ray actor on one CPU. ``wandb.log`` runs on the driver,
because the run lives there: an actor sees ``wandb.run is None`` and publishes nothing,
which is exactly what happened when the whole upload was moved off-process. Nothing
failed; three validation runs simply logged no table at all.

So the driver never waits for a render, and never delegates the log.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__file__)

# Fields shown per episode, in order. Keep in step with `build_table`.
_PER_EPISODE = ("rollout", "success", "score", "turns", "conversations")


class _Renderer:
    """Turns rows into episode records with HTML. No wandb here -- see the module note."""

    def render(self, rows: list[dict], n: int, step: int) -> tuple[list[dict], int]:
        from vagen.utils.episode_log import episode_rows, select_episodes

        return select_episodes(episode_rows(rows), n), step


def make_renderer(use_ray: bool = True):
    """A renderer. A Ray actor with one CPU when Ray is up, otherwise in-process."""
    if not use_ray:
        return _Renderer()
    try:
        import ray

        if not ray.is_initialized():
            return _Renderer()
        return ray.remote(num_cpus=1)(_Renderer).remote()
    except Exception as exc:  # noqa: BLE001 - logging must never take the run down
        logger.warning("episode renderer falling back to in-process: %s", exc)
        return _Renderer()


def build_table(episodes: list[dict], step: int, previous=None):
    """One row: this step, and the episodes chosen for it, side by side.

    ``previous`` carries the history. wandb tables are immutable once logged, so a
    growing table has to be rebuilt from the old rows -- the documented workaround.
    """
    import wandb

    columns = ["step"] + [f"ep{i}_{f}" for i in range(len(episodes)) for f in _PER_EPISODE]
    table = wandb.Table(columns=columns, data=list(previous.data) if previous is not None else [])
    row: list[Any] = [step]
    for e in episodes:
        row += [
            wandb.Html(e["html"]),
            e.get("success"),
            e.get("score"),
            e.get("turns"),
            e.get("conversations"),
        ]
    table.add_data(*row)
    return table


class EpisodeTableLogger:
    """Renders off-thread, logs on the driver, and never blocks on either."""

    def __init__(self, use_ray: bool = True):
        self._renderer = make_renderer(use_ray)
        self._pending: list = []
        self._table = None

    def submit(self, rows: list[dict], n: int, step: int) -> None:
        """Queue a render, and publish whatever finished since last time."""
        self._drain(block=False)
        if not rows or n <= 0:
            return
        if hasattr(self._renderer, "render") and hasattr(self._renderer.render, "remote"):
            self._pending.append(self._renderer.render.remote(rows, n, step))
            # One render in flight is enough; a second means the driver is outrunning
            # the actor, and waiting here is cheaper than an unbounded queue.
            if len(self._pending) > 1:
                self._drain(block=True)
        else:
            self._publish(*self._renderer.render(rows, n, step))

    def flush(self) -> None:
        """Publish everything outstanding. For the end of a run."""
        self._drain(block=True)

    # ------------------------------------------------------------------ internals
    def _drain(self, block: bool) -> None:
        if not self._pending:
            return
        import ray

        ready, rest = (self._pending, []) if block else ray.wait(self._pending, timeout=0)
        if not block:
            ready, rest = ready, rest
        for fut in ready:
            try:
                self._publish(*ray.get(fut))
            except Exception as exc:  # noqa: BLE001
                logger.warning("episode render failed: %s", exc)
        self._pending = rest

    def _publish(self, episodes: list[dict], step: int) -> None:
        import wandb

        if wandb.run is None or not episodes:
            return
        # Column count is fixed by the first row, so a step that selected fewer episodes
        # cannot extend the same table. Give it its own key rather than dropping it.
        if self._table is not None and len(self._table.columns) != 1 + len(episodes) * len(_PER_EPISODE):
            wandb.log({f"val/episodes_{len(episodes)}": build_table(episodes, step)}, step=step)
            return
        self._table = build_table(episodes, step, previous=self._table)
        wandb.log({"val/episodes": self._table}, step=step)
