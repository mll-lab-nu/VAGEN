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
from vagen.utils.image_token_utils import replace_image_tokens_for_logging
from vagen.utils.episode_log import describe_columns, rows_from_validation
from vagen.utils.wandb_episodes import EpisodeTableLogger


class VagenLogicMixin:
    """What VAGEN adds on top of verl's PPO loop. Bound to no verl method name."""

    # ------------------------------------------------------------------ setup
    def _vagen_init(self) -> None:
        """Called from the concrete trainer's ``__init__`` after ``super().__init__``."""
        from vagen.utils.upload_hugging_face import HFUploadManager

        self._hf_upload_manager = HFUploadManager(self.config)
        self._vagen_image_actors: dict = {}
        self._vagen_image_futures: list = []

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

    # ----------------------------------------------------------- image dumps
    def _vagen_dump_images(self, batch) -> None:
        """Write this step's environment frames alongside verl's JSONL dump.

        verl dumps text only. Images arrive as an ``image_data`` column, put there by
        the gym loops via ``extra_fields``.

        Writing happens in a Ray actor because encoding PNGs on the driver would stall
        the training loop, and the number of in-flight writes is capped: an environment
        that renders every turn can otherwise queue frames faster than they are written.
        """
        cfg = self.config.trainer.get("log_image", {})
        if not cfg.get("enable", False):
            return
        dump_path = self.config.trainer.get("rollout_data_dir", None)
        images = batch.non_tensor_batch.get("image_data") if dump_path else None
        if not dump_path or images is None:
            return

        import ray

        from vagen.utils.image_dump_actor import ImageDumpActor

        actor = self._vagen_image_actors.get(dump_path)
        if actor is None:
            actor = ImageDumpActor.remote(base_dir=dump_path)
            self._vagen_image_actors[dump_path] = actor

        self._vagen_image_futures.append(
            actor.dump_images.remote(
                step=self.global_steps,
                images=list(images),
                compress_level=cfg.get("png_compress_level", 0),
            )
        )

        max_pending = cfg.get("max_pending", 2)
        if max_pending > 0 and len(self._vagen_image_futures) > max_pending:
            done, rest = ray.wait(self._vagen_image_futures, num_returns=1)
            ray.get(done)  # re-raises if the write failed
            self._vagen_image_futures = rest

    def _vagen_flush_images(self) -> None:
        """Drain pending writes. Called before a checkpoint save, which may delete the
        directory an in-flight write is still targeting."""
        if not self._vagen_image_futures:
            return
        import ray

        ray.get(self._vagen_image_futures)
        self._vagen_image_futures = []

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

    def _dump_generations(self, inputs, outputs, *args, **kwargs):
        """Shorten image placeholder runs before verl writes the JSONL.

        One frame expands to hundreds of repeats of the same token, which buries the
        prompt in the dump. Overriding here rather than inside `_log_rollout_data`
        covers the validation dump too, since both funnel through this method.
        """
        if self.config.trainer.get("replace_image_tokens_for_logging", True):
            processor = getattr(self, "processor", None)
            inputs = replace_image_tokens_for_logging(inputs, processor)
            outputs = replace_image_tokens_for_logging(outputs, processor)
        return super()._dump_generations(inputs, outputs, *args, **kwargs)

    #: What the episode log needs from a validation batch. Named here rather than in
    #: verl: these are our columns, produced by our agent loop and our validation merge,
    #: and adding one should not touch the dependency. Eight separate upstream commits
    #: went into this list before it moved.
    val_log_columns = (
        "image_data",        # the frames the model was shown
        "episode_id",        # identity: episode > conversation > turn
        "group_idx",
        "traj_idx",
        "turn_idx",
        "conversation_id",
        "data_source",
        "episode_turns",     # counts the merge computes while it still can
        "n_conversations",
        "conversations",     # the transcript, laid out as it was spoken
    )

    def _fit_dump_data(self, batch):
        super()._fit_dump_data(batch)
        self._vagen_dump_images(batch)

    def _fit_validate(self, *args, **kwargs):
        """Validate, then make sure whatever it queued for wandb actually goes out.

        The episode table is rendered off-thread, so a submit only queues. Without a
        drain here a run whose validation happens once -- every short run, and the last
        validation of every long one -- finishes with the table still in the queue and
        nothing logged.
        """
        out = super()._fit_validate(*args, **kwargs)
        logger_ = getattr(self, "_vagen_val_logger", None)
        if logger_ is not None:
            logger_.flush()
        return out

    def _maybe_log_val_generations(self, inputs, outputs, scores, extras=None):
        """Hand validation episodes to the logger. The assembling lives in utils.

        verl's table is one row per model call, which is the wrong unit for looking at an
        agent: an episode is several calls, and once the context is compacted it is
        several conversations. Regrouping, selecting and rendering are in
        ``vagen/utils/episode_log.py`` and ``vagen/utils/wandb_episodes.py``; what belongs
        here is only the decision to log and where the columns come from.

        An override rather than a copy of ``_validate``, which is 150 lines that would
        then rot against upstream -- the reason the vendored trainer was deleted at all.
        """
        n = self.config.trainer.get("log_val_generations", 0)
        if not n:
            return
        extras = extras or {}
        images = extras.get("image_data")
        # Each checked on its own. `a or b` picks a list of Nones over a good list,
        # because a non-empty list is truthy whatever is in it -- which sent this down
        # the fallback path and logged verl's flat table instead of the episode one.
        print(f"[vagen] val episodes <- {describe_columns(extras, len(outputs))}")
        has_id = any(v is not None for v in (extras.get("episode_id") or []))
        has_group = any(v is not None for v in (extras.get("group_idx") or []))
        if not (has_id or has_group):
            # Nothing published episode ids, so there is nothing to regroup.
            return super()._maybe_log_val_generations(inputs, outputs, scores, extras=extras)

        if self.config.trainer.get("replace_image_tokens_for_logging", True):
            processor = getattr(self, "processor", None)
            inputs = replace_image_tokens_for_logging(inputs, processor)
            outputs = replace_image_tokens_for_logging(outputs, processor)

        if "wandb" not in self.config.trainer.logger:
            return
        if getattr(self, "_vagen_val_logger", None) is None:
            self._vagen_val_logger = EpisodeTableLogger()
        # Grouping, balancing and rendering all happen inside; the driver hands over the
        # raw rows and moves on.
        self._vagen_val_logger.submit(
            rows_from_validation(inputs, outputs, scores, images, extras),
            n,
            self.global_steps,
            self.config.trainer.get("val_log_select", "balanced"),
            float(self.config.trainer.get("val_log_success_ratio", 0.5)),
        )

    def _fit_save_checkpoint(self, *args, **kwargs):
        """HF Hub upload on its own schedule, independent of ``trainer.save_freq``.

        An upload-only step (hf_save_freq hits, save_freq does not) still needs a
        checkpoint on disk to upload from, so one has to be forced. Rather than
        duplicating verl's save condition (which would silently rot when upstream
        changes it), let ``super()`` decide and then check the filesystem.
        """
        # Both the HF upload and the image writer read from the checkpoint directory,
        # which a save may delete entries from.
        self._vagen_flush_images()

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
