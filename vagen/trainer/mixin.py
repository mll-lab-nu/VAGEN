"""VAGEN's trainer customisations, as mixins over verl's ``_fit_*`` hooks.

Replaces the vendored copy of ``verl/trainer/ppo/ray_trainer.py``. Rationale: v0.8.0's
``SeparateRayPPOTrainer`` breaks ``RayPPOTrainer``'s 410-line monolithic ``fit()`` into
~20 small ``_fit_*`` hooks, so our changes can be overrides instead of a fork -- and the
same mixin then composes with ``OneStepOffRayTrainer`` / ``FullyAsyncTrainer`` for free.

Two layers on purpose:

* ``VagenLogicMixin`` -- what we do. Bound to no verl method name, so migrating to
  verl main's V1 ``PPOTrainer`` (``on_step_end`` etc.) means rewriting only the layer
  below, not the logic.
* ``VagenV0Mixin`` -- where it hooks into v0.8.0's ``_fit_*``.

★ Placement matters. ``fit_step`` runs:

    _fit_compute_advantage -> _fit_update_critic -> _fit_update_actor
      -> _fit_dump_data -> _fit_validate -> _fit_save_checkpoint
      -> _fit_collect_metrics -> _fit_experimental

value_mask, custom metrics and filtering all belong *between* advantage and
update_critic, so all three hang off ``_fit_compute_advantage``'s tail. In particular
the filter must NOT go on ``_fit_experimental``: that runs after the actor update, so
filtering there would train on the unfiltered batch and then discard the result.
"""

from __future__ import annotations

import numpy as np
from verl.protocol import pad_dataproto_to_divisor
from verl.utils.debug import marked_timer

from vagen.custom_advantage import needs_value_mask
from vagen.custom_filter.filter import FILTER_REGISTRY
from vagen.custom_metric.metric import METRIC_REGISTRY
from vagen.trainer.logic import collect_registry_metrics, value_mask_from_returns


class VagenLogicMixin:
    """What VAGEN adds on top of verl's PPO loop. Bound to no verl method name."""

    # ------------------------------------------------------------------ setup
    def _vagen_init(self) -> None:
        """Called from the concrete trainer's ``__init__`` after ``super().__init__``."""
        from vagen.utils.upload_hugging_face import HFUploadManager

        self._hf_upload_manager = HFUploadManager(self.config)

    # -------------------------------------------------------------- advantage
    def _vagen_after_advantage(self, batch):
        """Everything that belongs between advantage computation and update_critic.

        Order is load-bearing: value_mask must exist before the critic runs, and the
        filter must shrink the batch before either update.
        """
        batch = self._vagen_write_value_mask(batch)
        self._vagen_collect_train_metrics(batch)
        batch = self._vagen_filter(batch)
        return batch

    def _vagen_write_value_mask(self, batch):
        """Tell the critic which positions carry return supervision.

        Only for estimators that emit sentinel returns. ``needs_value_mask`` reads the
        registry the estimators themselves populate, so it cannot drift from the set of
        estimators that actually emit sentinels (see custom_advantage/registry.py).
        """
        if needs_value_mask(self.config.algorithm.adv_estimator):
            batch.batch["value_mask"] = value_mask_from_returns(
                batch.batch["returns"], batch.batch["response_mask"]
            )
        return batch

    def _vagen_collect_train_metrics(self, batch) -> None:
        with marked_timer("custom_metrics", self.timing_raw, color="magenta"):
            self.metrics.update(
                collect_registry_metrics(METRIC_REGISTRY, batch, prefix="custom_metrics/train")
            )

    def _vagen_filter(self, batch):
        """STARPO-S / DAPO style batch filtering for effective updates.

        Filtering changes the row count, so when ``balance_batch`` is on the batch has
        to be re-padded to a multiple of the DP world size and re-balanced -- otherwise
        the dispatch split is uneven.
        """
        cfg = self.config.get("filter", None)
        if not (cfg and cfg.get("enable", False)):
            return batch

        batch, self.metrics = FILTER_REGISTRY[cfg.name](batch, self.metrics, **cfg.filter_kwargs)

        if self.config.trainer.balance_batch:
            before = len(batch.batch["attention_mask"])
            divisor = self.actor_rollout_wg.world_size
            batch, pad_size = pad_dataproto_to_divisor(batch, divisor)
            print(f"[vagen] filter: padded {before} -> {before + pad_size} for {divisor} dp workers")
            self._balance_batch(batch, metrics=self.metrics, logging_prefix="filtered_global_seqlen")
        return batch

    # ------------------------------------------------------------ checkpoint
    def _vagen_should_upload_hf(self) -> bool:
        return self._hf_upload_manager.should_upload(self.global_steps)

    def _vagen_flush_hf(self) -> None:
        """Drain pending uploads before a save.

        Must happen *before* ``_save_checkpoint``: with ``max_actor_ckpt_to_keep`` set,
        saving deletes older checkpoints, and an in-flight upload reading one of them
        would race with the delete.
        """
        self._hf_upload_manager.flush()

    def _vagen_upload_hf(self) -> None:
        self._hf_upload_manager.maybe_upload(self.global_steps)

    def _vagen_ckpt_dir_for_step(self) -> str:
        """Where verl writes this step's checkpoint (ray_trainer.py:978)."""
        import os

        return os.path.join(self.config.trainer.default_local_dir, f"global_step_{self.global_steps}")

    def _vagen_ckpt_exists_for_step(self) -> bool:
        import os

        return os.path.isdir(self._vagen_ckpt_dir_for_step())


class VagenV0Mixin(VagenLogicMixin):
    """Binds :class:`VagenLogicMixin` to verl v0.8.0's ``_fit_*`` hook names."""

    def _fit_compute_advantage(self, batch):
        batch = super()._fit_compute_advantage(batch)
        return self._vagen_after_advantage(batch)

    def _fit_save_checkpoint(self, *args, **kwargs):
        """HF Hub upload on its own schedule, independent of ``trainer.save_freq``.

        An upload-only step (hf_save_freq hits, save_freq does not) still needs a
        checkpoint on disk to upload from, so one has to be forced. Rather than
        duplicating verl's save condition (which would silently rot when upstream
        changes it), let ``super()`` decide and then check the filesystem.
        """
        upload = self._vagen_should_upload_hf()
        if upload:
            # Before any save: with max_*_ckpt_to_keep set, saving deletes older
            # checkpoints, which would race an in-flight upload reading one of them.
            self._vagen_flush_hf()

        # *args/**kwargs are forwarded rather than dropped: FullyAsyncTrainer
        # declares `_fit_save_checkpoint(self, force=False)` and calls it with
        # force=True, so a fixed no-arg signature here would break that composition.
        super()._fit_save_checkpoint(*args, **kwargs)

        if upload:
            if not self._vagen_ckpt_exists_for_step():
                with marked_timer("save_checkpoint", self.timing_raw, color="green"):
                    self._save_checkpoint()
            self._vagen_upload_hf()
