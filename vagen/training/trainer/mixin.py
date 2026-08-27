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

import json
import logging
import os

import numpy as np
import torch
from verl import DataProto
from verl.utils.debug import marked_timer

from vagen.algorithms import needs_value_mask
from vagen.training.filters import FILTER_REGISTRY
from vagen.training.metrics import METRIC_REGISTRY
from vagen.training.trainer.logic import (
    collect_registry_metrics,
    pad_to_multiple,
    row_multiple_required,
    value_mask_from_returns,
)
from vagen.models import replace_image_tokens_for_logging
from vagen.utils.episode_log import describe_columns, rows_from_validation
from vagen.utils.wandb_episodes import EpisodeTableLogger

logger = logging.getLogger(__name__)


class VagenLogicMixin:
    """What VAGEN adds on top of verl's PPO loop. Bound to no verl method name."""

    # ------------------------------------------------------------------ setup
    def _vagen_init(self) -> None:
        """Called from the concrete trainer's ``__init__`` after ``super().__init__``."""
        from vagen.utils.upload_hugging_face import HFUploadManager

        self._vagen_check_estimator_spans_the_layout()
        self._vagen_check_estimator_has_its_critic()
        self._vagen_check_estimator_is_undiscounted()
        self._vagen_check_turn_level_loss_has_what_it_needs()
        self._hf_upload_manager = HFUploadManager(self.config)
        self._vagen_image_actors: dict = {}
        self._vagen_image_futures: list = []

    def _vagen_harness_splits_rows(self) -> bool:
        """Whether this run's context policy puts one episode in several rows.

        ★ Asked of the harness class rather than matched against a tuple kept here --
        the same argument the docstring below makes for estimators. An unregistered name
        is treated as splitting, matching the base class default.
        """
        from vagen.harness import resolve_harness

        try:
            cls = resolve_harness(self._vagen_harness_mode())
        except (ValueError, TypeError):
            # Unresolvable here is not fatal -- build_harness will say so properly when the
            # rollout starts. Assume it splits, which is the conservative answer: it makes
            # the estimator check demand a trajectory estimator rather than waving through
            # a per-row one that would score a fraction of an episode.
            return True
        return cls.splits_episode_across_rows

    def _vagen_harness_mode(self) -> str:
        """Which context policy the agent loop will actually run.

        ★ Resolved the same way ``GymLoop._harness_mode`` resolves it, not read straight
        off the key: the yaml may ship ``harness: null``, and a guard that compares
        ``None`` against the splitting policies accepts everything.
        """
        return self.config.trainer.get("harness", None) or "concat"

    def _vagen_check_estimator_spans_the_layout(self) -> None:
        """Refuse a row-local estimator under a policy that splits episodes across rows.

        ★ This is a config error, so it stops the run rather than costing an episode.
        The pairing it rejects does not fail on its own: verl's ``gae`` and ``grpo``
        score each row independently and open every one with ``nextvalues=0``, which
        under ``no_concat`` or ``compact`` means turn *t+1* is never credited to turn
        *t*. Training runs, the curves look ordinary, and the multi-turn credit
        assignment the harness exists to provide is simply absent -- there is no
        downstream check that would notice.

        Which estimators are safe is read from the registry the estimators populate
        themselves (``custom_advantage/registry.py``), not from a list kept here, so a
        new estimator cannot be forgotten in one place and remembered in the other.
        """
        from vagen.algorithms import TRAJECTORY_ESTIMATORS, spans_rows

        harness = self._vagen_harness_mode()
        if not self._vagen_harness_splits_rows():
            return
        estimator = self.config.algorithm.adv_estimator
        if spans_rows(estimator):
            return
        raise ValueError(
            f"algorithm.adv_estimator={str(getattr(estimator, 'value', estimator))!r} scores one "
            f"row at a time, but trainer.harness={harness!r} splits an episode across rows -- "
            "every turn after the first would be dropped from its predecessors' returns, "
            "silently. Use one of "
            f"{sorted(TRAJECTORY_ESTIMATORS)}, or trainer.harness=concat."
        )

    def _vagen_check_estimator_has_its_critic(self) -> None:
        """Refuse a value-based estimator when no critic will be built.

        ★ Also a config error, and also silent. verl builds a critic when
        ``critic.enable`` is set; when it is *unset* it falls back to
        ``adv_estimator == "gae"`` -- the literal string. Every estimator in this repo
        fails that test, so an unset flag disables the critic, ``values`` reads as zeros,
        and GAE degenerates into a whitened discounted reward sum. The run starts, uses
        half the memory, trains faster, and the only evidence is a driver warning saying
        "Disabled critic as algorithm.adv_estimator != gae" -- which is true of the string
        and false of the algorithm.

        The exposure grew when the PPO scripts moved off the name ``gae``: before, the
        fallback happened to do the right thing.
        """
        from vagen.algorithms import needs_critic

        estimator = self.config.algorithm.adv_estimator
        # `self.use_critic` directly: verl's RayPPOTrainer.__init__ sets it well before
        # `_vagen_init` runs, so a missing attribute means the call moved somewhere it
        # should not be -- an AttributeError says that, a `getattr` default of False turns
        # it into "every value-based run refuses to start".
        if not needs_critic(estimator) or self.use_critic:
            return
        raise ValueError(
            f"algorithm.adv_estimator={str(getattr(estimator, 'value', estimator))!r} reads the "
            "critic's values, but no critic will be built. verl only infers one for the "
            "literal estimator name 'gae', so this needs critic.enable=True explicitly -- "
            "without it `values` is all zeros and the advantage silently becomes a "
            "discounted reward sum with no baseline."
        )

    def _vagen_check_estimator_is_undiscounted(self) -> None:
        """Refuse a two-clock estimator when ``gamma != 1``.

        ★ A two-clock estimator runs one recursion per token and switches lambda at turn
        boundaries, so a single turn is discounted twice over by two different clocks:
        the turn level pays one ``gamma`` to cross it, the token level pays
        ``gamma ** (tokens in the turn)``. They agree only at ``gamma == 1``.

        The size of the disagreement is set by how much the model wrote, not by anything
        in the config: at ``gamma = 0.99`` a 200-token turn is over-weighted 7.5x and a
        500-token turn 152x, so the effective horizon becomes a function of the policy's
        verbosity -- which the policy changes as it trains. Measured relative error
        against an exact policy gradient on a tabular multi-turn MDP: 0.11% at 0.999,
        1.06% at 0.99, 4.9% at 0.95.

        ``gamma`` defaults to something reasonable and every curve keeps its shape, so
        this has to be refused at startup or it is never noticed.
        """
        from vagen.algorithms import requires_undiscounted

        estimator = self.config.algorithm.adv_estimator
        gamma = float(self.config.algorithm.gamma)
        if not requires_undiscounted(estimator) or gamma == 1.0:
            return
        raise ValueError(
            f"algorithm.adv_estimator={str(getattr(estimator, 'value', estimator))!r} combines a "
            f"per-token recursion with a per-turn one, which is only defined at "
            f"algorithm.gamma=1.0, but gamma={gamma}. Crossing one turn costs the turn "
            f"level one gamma and the token level gamma**(turn length), so the two "
            f"disagree by a factor that grows with how much the model writes "
            f"({gamma}**500 = {gamma ** 500:.4g}). Set algorithm.gamma=1.0, or use "
            f"token_level_gae, which has one clock and is defined at any gamma."
        )

    def _vagen_check_turn_level_loss_has_what_it_needs(self) -> None:
        """``turn_gspo`` and ``turn_ppo`` each need two things the config can silently fail to provide.

        ★ It runs in the **actor worker**, a different process from this one. Registering
        it here registers it nowhere that matters: the worker builds its own registry from
        whatever it imported, and ours is not on that list unless
        ``actor_rollout_ref.model.external_lib`` names it. verl calls
        ``import_external_libs`` on that field while constructing the model config in the
        worker, which is the hook meant for exactly this.

        ★ And it needs ``turn_id``, which only the trajectory advantage estimators
        publish. Paired with verl's own ``gae`` or ``grpo`` there is no column saying
        where a turn starts, and a loss that guessed would be guessing "the row" -- which
        is verl's ``gspo``, the thing ``turn_gspo`` exists not to be.

        Both are refused here rather than in the worker: the worker's version of the first
        failure arrives several minutes into a run as ``Unsupported loss mode``, and its
        version of the second is a ``ValueError`` from inside the first backward pass.
        """
        from vagen.algorithms import PUBLISHES_TURN_ID, publishes_turn_id
        from vagen.training.losses import TURN_LEVEL_LOSSES

        actor = self.config.get("actor_rollout_ref", {}).get("actor", {})
        loss_mode = (actor.get("policy_loss", {}) or {}).get("loss_mode", "vanilla")
        if loss_mode not in TURN_LEVEL_LOSSES:
            return

        estimator = self.config.algorithm.adv_estimator
        if not publishes_turn_id(estimator):
            raise ValueError(
                f"actor.policy_loss.loss_mode={loss_mode!r} needs a `turn_id` column, and "
                f"algorithm.adv_estimator={str(getattr(estimator, 'value', estimator))!r} "
                f"does not publish one. Use one of {sorted(PUBLISHES_TURN_ID)}, which "
                f"locate the turns while computing the advantage. Nothing else in the "
                f"batch says where a turn starts, and the only fallback would be to treat "
                f"a row as a turn -- which is verl's own `gspo`, and is an entire episode "
                f"under concat."
            )

        external = self.config.get("actor_rollout_ref", {}).get("model", {}).get("external_lib", None)
        named = [external] if isinstance(external, str) else list(external or [])
        if "vagen.training.losses" not in named:
            raise ValueError(
                f"actor.policy_loss.loss_mode={loss_mode!r} but the actor worker will not "
                "import it. The policy loss runs in a separate process and builds its own "
                "registry, so add:\n\n"
                "    actor_rollout_ref.model.external_lib=vagen.training.losses\n\n"
                f"Without it the worker raises 'Unsupported loss mode: {loss_mode}' at the "
                f"first update step. Currently external_lib={external!r}."
            )

    # -------------------------------------------------------------- advantage
    def _vagen_after_advantage(self, batch):
        """Everything that belongs between advantage computation and update_critic.

        Order is load-bearing: value_mask must exist before the critic runs, and the
        filter must shrink the batch before either update.
        """
        batch = self._vagen_write_value_mask(batch)
        self._vagen_collect_train_metrics(batch)
        batch = self._vagen_filter(batch)
        batch = self._vagen_pad_rows_for_split(batch)
        return batch

    #: Zeroed on a padding row so it cannot reach a loss. ``response_mask`` is what the
    #: policy and value losses average over; the rest are zeroed so a row that somehow
    #: escapes the mask still contributes nothing rather than a copy of row 0's reward.
    _NEUTRALISED_ON_PAD = (
        "response_mask",
        "loss_mask",
        "advantages",
        "returns",
        "token_level_scores",
        "token_level_rewards",
        "value_mask",
    )

    def _vagen_pad_rows_for_split(self, batch):
        """Make the row count divisible by everything that is about to divide it.

        ``DataProto.split`` asserts ``batch_size[0] % mini_batch_size == 0``. Under concat
        an episode is one row and the count is whatever ``train_batch_size * rollout.n``
        says, so it always divides; under no_concat an episode becomes a *variable* number
        of rows and the count is whatever the rollouts happened to produce. That only
        divides by luck, and the luck runs out once the batch is small -- 32 prompts x 4
        rollouts produced 76 rows against a mini batch of 16 and the step died on
        ``AssertionError: 76 % 16 != 0``.

        Padding existed only inside ``_vagen_filter``, so it never ran with the filter off,
        and it aligned to the DP world size only -- neither constraint being the one that
        failed.

        The filler rows are copies of row 0 with every loss-bearing tensor zeroed, so they
        occupy split slots without contributing gradient. Repeating a real row *unmasked*
        -- which is what ``pad_dataproto_to_divisor`` does -- would quietly weight that
        episode more heavily than the others.
        """
        b = getattr(batch, "batch", None)
        if b is None or "response_mask" not in b.keys():
            return batch

        n = b.batch_size[0]
        # Read defensively: a trainer assembled without one of these sections must not
        # lose its batch padding to an AttributeError on a config key.
        arr = self.config.get("actor_rollout_ref", None) or {}
        actor_mini = (arr.get("actor", None) or {}).get("ppo_mini_batch_size", 0)
        critic_cfg = self.config.get("critic", None) or {}
        critic_mini = critic_cfg.get("ppo_mini_batch_size", 0) if critic_cfg.get("enable", False) else 0
        # ★ Both mini batches are counted in *prompts*; verl multiplies them by rollout.n
        # to get rows (ray_trainer.py: `ppo_mini_batch_size * rollout.n`) and then divides
        # by the DP size. Padding against the unscaled number looks right, divides evenly
        # at the driver, and still leaves each worker's shard indivisible -- which is how
        # the first version of this passed its own arithmetic and the run still died.
        rollout_n = (arr.get("rollout", None) or {}).get("n", 1) or 1
        multiple = row_multiple_required(
            self.actor_rollout_wg.world_size, actor_mini * rollout_n, critic_mini * rollout_n
        )

        extra = pad_to_multiple(n, multiple)
        if not extra:
            return batch

        padded = self._vagen_pad_to_multiple(batch, multiple)
        print(f"[vagen] split-pad: {n} -> {n + extra} rows for multiple {multiple} "
              f"(dp={self.actor_rollout_wg.world_size}, actor_mini={actor_mini}, critic_mini={critic_mini})")
        self.metrics["custom_metrics/train/split_pad_rows"] = float(extra)
        return padded

    def _vagen_pad_to_multiple(self, batch, multiple: int):
        """Pad the row count to a multiple of ``multiple`` with rows that carry no gradient.

        The filler is a copy of row 0 with every loss-bearing tensor zeroed, so it occupies
        a split slot and contributes nothing. verl's ``pad_dataproto_to_divisor`` repeats
        real rows *unmasked* instead, which weights those episodes twice.
        """
        n = len(batch.batch["attention_mask"])
        extra = (-n) % max(1, int(multiple))
        if not extra:
            return batch
        filler = batch.select_idxs([0] * extra)
        for key in self._NEUTRALISED_ON_PAD:
            if key in filler.batch.keys():
                filler.batch[key] = torch.zeros_like(filler.batch[key])
        return DataProto.concat([batch, filler])

    def _vagen_write_value_mask(self, batch):
        """Tell the critic which positions carry return supervision.

        Only for estimators that emit sentinel returns. ``needs_value_mask`` reads the
        registry the estimators themselves populate, so it cannot drift from the set of
        estimators that actually emit sentinels (see custom_advantage/registry.py).

        ★ A fallback, not the main path. An estimator built on ``AdvantageOutputs``
        states its own ``value_mask`` and has already written it here -- that is one
        source of truth instead of two that can disagree. Recovering the mask by
        scanning for the sentinel is what an estimator returning a bare tuple gets.
        """
        if "value_mask" in batch.batch:
            return batch
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

    def _vagen_rescope_row_metrics_to_episodes(self) -> None:
        """Report episode-level quantities per episode, not per row.

        In concat an episode is one row and the two agree. In no_concat and compact an
        episode becomes several rows, and an episode's reward divided over them reads as
        ``episode_reward / rows_per_episode`` -- so the headline training reward is low by
        the split factor, and only for the harnesses that split.

        On the 2026-08-12 sweep this inverted the ranking outright. The concat arms matched
        validation (0.571 vs 0.573, 0.843 vs 0.823) while compact read 0.539 against 0.776
        and no_concat read 0.375 against 0.918 -- each low by exactly its rows-per-episode
        factor. By the row metric no_concat was the worst arm; by episode it was the best.

        Only ``critic/score`` is rescoped, because ``episode_score`` recomputes exactly that
        quantity per episode. ``critic/rewards`` differs from it by the KL penalty and has
        no episode-grouped counterpart yet, so it is left alone rather than rescaled by a
        factor that would only be a guess.

        The row-scoped values are kept under ``.../by_row`` rather than dropped: they are
        what verl computed, and a number that changes meaning without changing name is how
        this went unnoticed for a whole sweep.
        """
        episode = self.metrics.get("custom_metrics/train/episode_score/mean")
        if episode is None:
            return  # the metric failed or episode_id was absent; leave verl's numbers alone
        if "critic/score/mean" not in self.metrics:
            return
        for stat in ("mean", "max", "min"):
            value = self.metrics.pop(f"critic/score/{stat}", None)
            if value is not None:
                self.metrics[f"critic/score/by_row/{stat}"] = value
        self.metrics["critic/score/mean"] = episode

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
            # ★ Neutralised filler, not `pad_dataproto_to_divisor`. That repeats real rows
            # verbatim, and this runs *after* advantage -- so up to world_size-1 real rows
            # would carry their advantages and response_mask twice: double gradient weight
            # and double weight in critic/score, for whichever episodes happen to sit at
            # the front of the batch. `_vagen_pad_rows_for_split`'s own docstring says so
            # thirty lines up; this call site was the one that had not caught up.
            batch = self._vagen_pad_to_multiple(batch, divisor)
            print(f"[vagen] filter: padded {before} -> {len(batch.batch['attention_mask'])} "
                  f"for {divisor} dp workers")
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

    def _fit_collect_metrics(self, *args, **kwargs):
        """Rescope after verl has computed its data metrics, not before.

        ``critic/score/*`` is produced by ``compute_data_metrics`` inside this hook, which
        runs several hooks *after* ``_fit_compute_advantage``. Rescoping from the earlier
        hook -- where the custom metrics are collected -- meant the key did not exist yet,
        the rescope took its "leave verl's numbers alone" branch every step, and the fix
        was inert. Caught by a live no_concat run reporting
        ``episode_score/mean 0.456`` beside ``critic/score/mean 0.027``.
        """
        out = super()._fit_collect_metrics(*args, **kwargs)
        self._vagen_rescope_row_metrics_to_episodes()
        return out

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
    #
    # ``turn_idx`` and ``conversation_id`` are deliberately absent. They identify a *row*,
    # and the validation merge folds an episode's rows into one -- so an episode has many
    # of each and no single value, and the merge drops them. Requesting them anyway made
    # the diagnostic report "turn_idx=0/256 conversation_id=0/256" every validation, which
    # reads as a failure and is not one: the per-turn structure is inside ``conversations``,
    # which is 256/256. Where they still identify one thing is the training path, and that
    # does not come through here.
    val_log_columns = (
        "image_data",        # the frames the model was shown
        "episode_id",        # identity: episode > conversation > turn
        "group_idx",
        "traj_idx",
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
        self._vagen_maybe_save_best_actor()
        return out

    #: Where the best-scoring actor is kept, under `trainer.default_local_dir`.
    VAGEN_BEST_ACTOR_DIR = "best_actor"

    def _vagen_best_val_score(self):
        """The validation score to select on: mean ``val-core/<env>/reward/mean@1``.

        Averaged across environments when a run validates on several, so a two-env run
        does not silently select on whichever key happens to sort first. Returns None
        when validation produced no such key, which is the honest answer for a step that
        did not validate -- `_fit_validate` is called every step and only sometimes runs.
        """
        metrics = getattr(self, "metrics", None) or {}
        keys = [k for k in metrics
                if k.startswith("val-core/") and k.endswith("/reward/mean@1")]
        if not keys:
            return None
        try:
            return sum(float(metrics[k]) for k in keys) / len(keys)
        except (TypeError, ValueError):
            return None

    def _vagen_maybe_save_best_actor(self) -> None:
        """Keep a copy of the actor from the best-validating step.

        ★ The actor only, and in its own directory. The periodic checkpoint exists to
        *resume* -- it carries the critic and the optimiser and it is what
        `latest_checkpointed_iteration.txt` points at. This one exists to be *used*, so it
        carries none of that and, crucially, does not touch that file: writing it would
        make a resume rewind to whichever step happened to validate best, which is not
        where training was.

        Selection is on reward rather than success rate because that is what was asked
        for. Worth knowing which you are getting: `val-core/.../reward` includes the
        format bonus (0.10 of a 1.20 maximum at the shipped weights), so it prefers a
        checkpoint that is slightly better at writing tags over one slightly better at
        solving. `trainer.save_best_actor: false` turns the whole thing off.
        """
        if not bool(self.config.trainer.get("save_best_actor", True)):
            return
        score = self._vagen_best_val_score()
        if score is None:
            return

        best = getattr(self, "_vagen_best_val", None)
        if best is not None and score <= best:
            return
        self._vagen_best_val = score

        path = os.path.join(self.config.trainer.default_local_dir, self.VAGEN_BEST_ACTOR_DIR)
        try:
            self.actor_rollout_wg.save_checkpoint(path, None, self.global_steps)
            os.makedirs(path, exist_ok=True)
            with open(os.path.join(path, "best.json"), "w") as fh:
                json.dump({"global_step": self.global_steps,
                           "val_core_reward": score}, fh, indent=1)
        except Exception as exc:  # noqa: BLE001
            # A failed bookkeeping save must not end a training run that is going fine.
            logger.warning("[vagen] could not save the best actor at step %s: %s",
                           self.global_steps, exc)
            return
        logger.info("[vagen] new best val-core reward %.4f at step %s -> %s",
                    score, self.global_steps, path)

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
