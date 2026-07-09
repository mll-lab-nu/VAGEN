"""Best-validation checkpoint tracking.

Keeps a single ``best_val/`` directory under ``trainer.default_local_dir``
that mirrors the actor's HuggingFace model at the validation step that
produced the highest ``val-core`` score so far. Only the HF model is
retained (no optimizer / extra state), matching the user request to
"只保留 model". Optionally triggers an HF upload of the new best.

Lives outside ray_trainer.py to keep that file's surface small — the
trainer just constructs one tracker and calls :meth:`maybe_save` after
every in-loop validation.
"""
from __future__ import annotations

import json
import os
import shutil
from typing import Optional


class BestValTracker:
    """Owns the best-validation state machine and on-disk side effects.

    Wire-up in the trainer::

        self._best_val_tracker = BestValTracker(config)
        ...
        # After each in-loop ``_validate()``:
        self._best_val_tracker.maybe_save(
            val_metrics=val_metrics,
            global_steps=self.global_steps,
            actor_rollout_wg=self.actor_rollout_wg,
            hf_upload_manager=self._hf_upload_manager,
        )
    """

    def __init__(self, config):
        self.enabled: bool = bool(config.trainer.get("save_best_val", False))
        self.default_local_dir: str = config.trainer.default_local_dir
        self.best_score: Optional[float] = None
        self.best_step: Optional[int] = None

    # ------------------------------------------------------------------
    # Score aggregation
    # ------------------------------------------------------------------
    @staticmethod
    def aggregate_val_core_score(val_metrics: dict) -> Optional[float]:
        """Reduce the trainer's ``val-core/...`` metrics to one scalar.

        ``_validate()`` emits keys shaped like
        ``val-core/<data_source>/<var>/<metric>@<N>``. Higher = better.
        We average all ``val-core`` numeric scalars so the score works
        for single- and multi-dataset configs alike. Returns ``None``
        when no ``val-core`` scalars exist.
        """
        scores = [
            float(v) for k, v in val_metrics.items()
            if k.startswith("val-core/") and isinstance(v, (int, float))
        ]
        if not scores:
            return None
        return sum(scores) / len(scores)

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------
    def maybe_save(self, *, val_metrics: dict, global_steps: int,
                   actor_rollout_wg, hf_upload_manager) -> None:
        """Save + optionally upload a new best_val checkpoint, if this
        validation beats the current best.

        Args:
            val_metrics: dict returned by ``RayPPOTrainer._validate``.
            global_steps: current trainer step (used for metadata + HF
                commit message).
            actor_rollout_wg: the actor worker group, used to write the
                FSDP/HF checkpoint via its ``save_checkpoint`` API.
            hf_upload_manager: ``HFUploadManager`` instance; ``flush`` is
                called before mutating ``best_val/`` and
                ``maybe_upload_best_val`` after. No-op if HF upload is
                disabled.
        """
        if not self.enabled:
            return
        score = self.aggregate_val_core_score(val_metrics)
        if score is None:
            return
        if self.best_score is not None and score <= self.best_score:
            return

        prev = self.best_score
        self.best_score = score
        self.best_step = global_steps
        print(
            f"[BestVal] New best validation score {score:.6f} at step "
            f"{global_steps} (previous: {prev})."
        )

        best_val_root = os.path.join(self.default_local_dir, "best_val")
        actor_local_path = os.path.join(best_val_root, "actor")

        # Flush any in-flight HF upload before mutating best_val/.
        hf_upload_manager.flush()

        # Wipe stale best_val from the previous best step so only the new
        # HF model survives the prune step below.
        if os.path.isdir(best_val_root):
            shutil.rmtree(best_val_root, ignore_errors=True)

        actor_rollout_wg.save_checkpoint(
            actor_local_path, None, global_steps, max_ckpt_to_keep=None
        )
        _prune_to_huggingface(actor_local_path)
        _write_metadata(best_val_root, global_steps, score)
        hf_upload_manager.maybe_upload_best_val(global_steps, best_val_root)


# ----------------------------------------------------------------------
# Module-level helpers (no shared state — easier to unit-test)
# ----------------------------------------------------------------------
def _prune_to_huggingface(actor_local_path: str) -> None:
    """Delete everything under ``actor_local_path`` except ``huggingface/``."""
    hf_subdir = os.path.join(actor_local_path, "huggingface")
    if not os.path.isdir(actor_local_path):
        return
    for entry in os.listdir(actor_local_path):
        p = os.path.join(actor_local_path, entry)
        if p == hf_subdir:
            continue
        if os.path.isdir(p):
            shutil.rmtree(p, ignore_errors=True)
        else:
            try:
                os.remove(p)
            except OSError:
                pass


def _write_metadata(best_val_root: str, global_steps: int, score: float) -> None:
    """Stamp ``best_val/best_val.json`` so consumers know which step won."""
    try:
        with open(os.path.join(best_val_root, "best_val.json"), "w") as f:
            json.dump({"step": global_steps, "score": score}, f, indent=2)
    except OSError:
        pass
