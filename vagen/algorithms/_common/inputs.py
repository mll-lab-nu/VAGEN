"""The interface an advantage estimator is written against.

Writing a new algorithm should be: say what a step is, say what the advantage is, return
it. Everything else -- which key holds which tensor, which rows belong to which episode,
how padding duplicates are handled, how a mask reaches the loss -- is the same every
time, and getting any of it wrong is silent.

    from vagen.algorithms import AdvantageInputs, AdvantageOutputs, advantage_estimator

    @advantage_estimator("my_algo")
    def my_algo(inputs: AdvantageInputs):
        adv = inputs.zeros()
        for rows in inputs.view.trajectories:      # one episode, its rows in turn order
            ...
        return AdvantageOutputs(advantages=adv, returns=adv + inputs.values)

Run it with ``algorithm.adv_estimator=my_algo``. Registering through this decorator is
also what tells the trainer the estimator scores whole episodes, so it is allowed under
``no_concat`` and ``compact`` -- see ``registry.py``.

--------------------------------------------------------------------------------------
What verl gives you, and what it does with what you return
--------------------------------------------------------------------------------------

verl 0.8.0 calls an estimator with keyword arguments and expects exactly two tensors
back. ``compute_advantage`` (``verl/trainer/ppo/ray_trainer.py``) does::

    advantages, returns = adv_estimator_fn(**adv_kwargs)
    data.batch["advantages"] = advantages
    data.batch["returns"] = returns

and that is the whole output contract. There is no separate actor/critic advantage and
no mask in the return value:

* the **actor** reads ``advantages`` and masks with ``response_mask``
* the **critic** reads ``returns`` and ``values`` and masks with ``response_mask``

So anything beyond two tensors has to travel as a side channel in the batch. One such
channel exists: ``value_mask``, a VAGEN addition to ``verl/workers/utils/losses.py``,
narrows the critic's supervision. :class:`AdvantageOutputs` writes it for you;
``inputs.batch`` is the same object verl holds, so writing into it is what makes it
visible downstream.

★ There is deliberately **no actor-side equivalent, and zeroing an advantage is not a
substitute for one.** To stop the policy learning a token, take it out of
``response_mask`` -- that is the complete lever and it is already the one this repo uses:
``loss_mask=response_mask`` (``verl/experimental/separation/ray_trainer.py``), and
``batch_num_tokens = data["loss_mask"].sum()`` all-reduced across ranks
(``verl/workers/engine/fsdp/transformer_impl.py``), so removing a token there removes it
from the numerator *and* the denominator *and* the entropy and KL terms.

Setting its advantage to 0.0 does none of that:

* it kills only the policy-gradient term. The entropy bonus and the KL loss are both
  aggregated over ``response_mask`` alone (``verl/workers/utils/losses.py``), so a
  zero-advantage token still gets pulled toward the reference policy and still gets an
  entropy gradient.
* **it does not even stay zero.** Every estimator here finishes with
  ``masked_whiten(advantages, mask)``, which subtracts the batch mean and divides by the
  std -- so an advantage deliberately set to 0.0 at a mask-1 position comes out as
  ``-mean/std``, an ordinary non-zero coefficient. A mask is a 0/1 declaration; an
  advantage is a float magnitude, and only the first survives an affine renormalisation.

A loss-side ``policy_mask`` would mean something narrower -- "this *is* an action, keep it
in the denominator, but do not apply the policy/entropy/KL terms to it this step". Nothing
needs that yet, and it would be another patch to verl, so it does not exist.

Which inputs arrive is decided by **signature introspection**: verl passes ``batch`` and
``non_tensor_batch`` only to estimators that declare those parameters. The adapter this
module builds declares them, which is why ``functools.wraps`` is not used on it --
``wraps`` sets ``__wrapped__``, ``inspect.signature`` follows it to the inner function,
and the containers would silently stop being passed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Optional

import torch

from vagen.algorithms._common import params
from vagen.algorithms._common.registry import (
    register_sentinel_adv_est,
    register_trajectory_adv_est,
)
from vagen.algorithms._common.trajectory import TrajectoryView


class AdvantageInputs:
    """Everything known about the batch at the moment advantages are computed.

    Tensors are ``(n_rows, response_length)`` and indexed by *row*, which is one
    conversation -- under ``concat`` that is a whole episode, under ``no_concat`` one
    turn, under ``compact`` one compaction window. Identity columns are one entry per
    row. :attr:`view` is what turns rows back into episodes.

    Anything not surfaced here is still reachable through :attr:`batch` and
    :attr:`non_tensor_batch`; a property is added when a second algorithm wants it.
    """

    def __init__(self, batch, non_tensor_batch, config, estimator: str):
        self.batch = batch
        self.non_tensor_batch = non_tensor_batch
        self.config = config
        self.estimator = estimator
        self._view: Optional[TrajectoryView] = None

    # ------------------------------------------------------------ per-token tensors

    @property
    def response_mask(self) -> torch.Tensor:
        """1 where the model emitted the token. Everything else -- observations, the
        chat template's scaffolding, padding -- is 0 and is state, not action."""
        return self.batch["response_mask"]

    @property
    def scores(self) -> torch.Tensor:
        """Raw per-token reward from the environment, before any KL penalty."""
        return self.batch["token_level_scores"]

    @property
    def rewards(self) -> torch.Tensor:
        """The reward to build the advantage from: ``token_level_rewards`` when it
        exists, ``token_level_scores`` otherwise.

        ★ Use this, not :attr:`scores`. verl writes the KL-penalised reward into the
        former and leaves the latter untouched, so reading the scores directly makes
        ``algorithm.use_kl_in_reward=True`` a silent no-op -- the penalty is computed,
        stored, and never read.
        """
        rewards = self.batch.get("token_level_rewards")
        return self.batch["token_level_scores"] if rewards is None else rewards

    @property
    def values(self) -> torch.Tensor:
        """Critic output, already aligned so that ``values[:, i]`` is ``V(s_i)`` -- the
        value of the state the token at ``i`` was emitted from. verl does the shift in
        ``verl/workers/utils/padding.py``; the critic's own output at ``i`` is conditioned on
        tokens up to and including ``i``, which is a different quantity.

        Zeros when there is no critic, so a critic-free estimator can read it safely.
        """
        values = self.batch.get("values")
        return torch.zeros_like(self.scores) if values is None else values

    @property
    def old_log_probs(self) -> Optional[torch.Tensor]:
        """Actor log-probs recomputed at training time. ``None`` if not computed."""
        return self.batch.get("old_log_probs")

    @property
    def ref_log_probs(self) -> Optional[torch.Tensor]:
        """Reference-policy log-probs, or ``None`` when no reference model is used.

        Note verl spells this key ``ref_log_prob``, singular, unlike ``old_log_probs``.
        """
        return self.batch.get("ref_log_prob")

    @property
    def rollout_log_probs(self) -> Optional[torch.Tensor]:
        """Log-probs reported by the inference engine, or ``None``. Differs from
        :attr:`old_log_probs` whenever rollout and training numerics diverge."""
        return self.batch.get("rollout_log_probs")

    def kl(self, penalty: str = "kl") -> Optional[torch.Tensor]:
        """Per-token KL between the actor and the reference policy, masked to responses.

        ``None`` unless both log-prob tensors are present. verl does not store this: it
        folds ``- beta * kl`` into ``token_level_rewards`` and keeps only a scalar
        metric, so an estimator that wants the per-token quantity has to recompute it.
        """
        old, ref = self.old_log_probs, self.ref_log_probs
        if old is None or ref is None:
            return None
        from verl.trainer.ppo.core_algos import kl_penalty

        return kl_penalty(old, ref, kl_penalty=penalty) * self.response_mask

    # ------------------------------------------------------------------- identity

    def column(self, name: str, default=None):
        """A non-tensor column, one entry per row, or ``default`` if absent."""
        value = self.non_tensor_batch.get(name)
        return default if value is None else value

    @property
    def group_idx(self):
        """Prompt group. Every rollout of the same prompt shares it -- the axis GRPO
        and any other group-relative baseline normalises over. A uuid string."""
        return self.non_tensor_batch["group_idx"]

    @property
    def traj_idx(self):
        """Which rollout within the prompt group."""
        return self.non_tensor_batch["traj_idx"]

    @property
    def episode_id(self):
        """One episode, minted per ``run_episode``. ``None`` if the loop did not emit it.

        ★ Not the same as ``(group_idx, traj_idx)`` in general -- that pair identifies a
        *rollout*, and a rollout is one episode only because the loop makes it so.
        :attr:`view` prefers this column and falls back to the pair only when the loop
        did not publish one.
        """
        return self.column("episode_id")

    @property
    def conversation_id(self):
        """Which conversation of the episode this row is: 0 under ``concat``, the turn
        number under ``no_concat``, the compaction window under ``compact``."""
        return self.column("conversation_id", self.column("turn_idx"))

    @property
    def turn_idx(self):
        """Ordering key within an episode. Equals :attr:`conversation_id`."""
        return self.column("turn_idx", self.column("conversation_id"))

    @property
    def ends_with_summary(self):
        """Whether this row's last model emission is a compaction summary.

        ★ A seam is not a transition. Under ``compact`` the row ends because the context
        filled up: the model was asked to summarise, wrote one, and the next conversation
        opens on the *same* world state. No environment step separates the summary from
        the next action, so an estimator that spends a lambda crossing a turn must not
        spend one here -- if it does, the amount of credit that survives a compaction is
        set by how often the policy compacts, which is set by how much it writes.

        ``None`` when the column is absent, which is every harness but ``compact`` and
        every batch produced before the column existed.
        """
        return self.column("ends_with_summary")

    @property
    def last_turn(self):
        """Whether this row is the episode's final conversation."""
        return self.column("last_turn")

    @property
    def uid(self):
        """verl's own grouping key, one uuid per prompt before the rollout repeat. Its
        built-in GRPO normalises over this; under a splitting harness every row of an
        episode shares it, which is why those estimators are refused there."""
        return self.column("uid")

    # ------------------------------------------------------------------ structure

    @property
    def view(self) -> TrajectoryView:
        """Rows deduplicated and grouped into episodes, each in turn order.

        Built once and cached. This is what makes an estimator layout-independent:
        under ``concat`` every trajectory's row list has length one, and the same code
        runs unchanged.
        """
        if self._view is None:
            self._view = TrajectoryView.build(self.response_mask, self.non_tensor_batch)
        return self._view

    def zeros(self) -> torch.Tensor:
        """A ``(n_rows, response_length)`` zero tensor matching the batch."""
        return torch.zeros_like(self.scores)

    # --------------------------------------------------------------------- config

    def param(self, name: str, default):
        """An ``algorithm.*`` hyperparameter that means something specific when unset."""
        return params.optional(self.config, name, default)

    def required_param(self, name: str, why: str):
        """An ``algorithm.*`` hyperparameter with no sensible default.

        Raises with instructions rather than defaulting: verl's ``AlgoConfig`` has a
        fixed field list, so a new knob needs hydra's ``+algorithm.name=...``, and
        forgetting it would otherwise silently select a different algorithm.
        """
        return params.required(self.config, name, self.estimator, why)


@dataclass
class AdvantageOutputs:
    """What an estimator produces.

    ``advantages`` and ``returns`` are the two tensors verl itself consumes.
    ``value_mask`` is an optional narrowing, written into the batch as a side channel
    because verl's return contract has no room for it. There is no actor-side equivalent:
    to stop the policy learning a token, take it out of ``response_mask``. Zeroing its
    advantage is not the same thing -- see this module's docstring.
    """

    #: Coefficient on each token's log-prob in the policy gradient. Actor.
    advantages: torch.Tensor
    #: Regression target for the critic at each token. Critic.
    returns: torch.Tensor
    #: Positions where ``returns`` carries real supervision. ``None`` means all of them.
    #: Needed by estimators that anchor one return per turn and leave the rest at a
    #: sentinel -- without it the critic regresses towards the sentinel and its loss
    #: *falls*, so nothing looks wrong.
    value_mask: Optional[torch.Tensor] = None
    #: Anything else to publish into the batch, e.g. a diagnostic the metrics read.
    extra: dict[str, torch.Tensor] = field(default_factory=dict)

    def write_side_channels(self, batch) -> tuple[torch.Tensor, torch.Tensor]:
        """Put the masks where the losses look for them, and hand back verl's pair."""
        if self.value_mask is not None:
            batch["value_mask"] = self.value_mask
        for key, tensor in self.extra.items():
            batch[key] = tensor
        return self.advantages, self.returns


def advantage_estimator(
    name: str,
    *,
    needs_critic: bool = False,
    sentinel_returns: bool = False,
    undiscounted: bool = False,
    turn_lumped_reward: bool = False,
    publishes_turn_id: bool = True,
) -> Callable:
    """Register a function of :class:`AdvantageInputs` as an advantage estimator.

    Args:
        name: what ``algorithm.adv_estimator`` selects.
        needs_critic: set when the estimator reads :attr:`AdvantageInputs.values`. Without
            it, a run with no critic silently trains a different algorithm -- see
            ``registry.needs_critic``. ``tests/test_advantage_registry.py`` fails any
            estimator that reads ``values`` without declaring this.
        sentinel_returns: set when the estimator leaves some positions of ``returns``
            unsupervised. Only affects the trainer's fallback ``value_mask``; prefer
            returning ``value_mask`` on :class:`AdvantageOutputs`, which is explicit.
        publishes_turn_id: whether the estimator emits the ``turn_id`` column, which the
            turn-level policy losses read. True for every estimator that goes through
            ``_Packed.emit``; set False for one returning a bare ``AdvantageOutputs``.
            Keyed off ``spans_rows`` instead, the turn-loss guard accepted an estimator
            that has no turn_id and then raised inside the first backward pass.
        undiscounted: set when the estimator mixes a per-token recursion with a per-turn
            one, which is only defined at ``gamma == 1`` -- see
            ``registry.requires_undiscounted`` for why, and for the measured error at
            gamma < 1. The trainer refuses the pairing at startup.
        turn_lumped_reward: set when the estimator reads a turn's reward only at the
            turn's last token, so the reward wrapper must lump it there rather than pay
            it where it was earned -- see ``registry.wants_turn_lumped_reward``.

    The estimator may return an :class:`AdvantageOutputs` or a plain
    ``(advantages, returns)`` tuple.
    """

    def decorator(fn):
        def adapter(*, batch, non_tensor_batch, config=None, **kwargs):
            result = fn(AdvantageInputs(batch, non_tensor_batch, config, name))
            if isinstance(result, AdvantageOutputs):
                return result.write_side_channels(batch)
            return result

        # Deliberately not functools.wraps: it sets __wrapped__, inspect.signature
        # follows it, and verl decides whether to pass `batch`/`non_tensor_batch` by
        # looking at exactly that signature -- so the estimator would be handed neither.
        adapter.__name__ = getattr(fn, "__name__", name)
        # Without this the qualname still says `<locals>.adapter`, which plain pickle
        # refuses. Nothing pickles it today (the registry is rebuilt by import in each
        # process), but the cost of not lying about it is one line.
        adapter.__qualname__ = adapter.__name__
        adapter.__doc__ = fn.__doc__
        adapter.__module__ = getattr(fn, "__module__", __name__)
        adapter.vagen_estimator_fn = fn

        register = register_sentinel_adv_est if sentinel_returns else register_trajectory_adv_est
        register(
            name,
            needs_critic=needs_critic,
            undiscounted=undiscounted,
            turn_lumped_reward=turn_lumped_reward,
            publishes_turn_id=publishes_turn_id,
        )(adapter)
        return fn

    return decorator
