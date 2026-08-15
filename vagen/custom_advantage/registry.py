"""What the trainer needs to know about an estimator, declared where it registers.

Two properties, both of which used to be -- or would otherwise be -- a hard-coded list
of names somewhere else in the tree. A list like that drifts from the names actually
registered, and both failure modes here are silent.

**Sentinel returns.** Turn-level estimators such as ``turn_level_gae`` write a real
return at one anchor token per turn and leave every other position at ``IGNORE_RETURN``
(-100.0). The critic must be told which positions carry supervision, via ``value_mask``;
without it, it is trained to regress towards the sentinel almost everywhere. The symptom
is a *falling* value loss and a healthy-looking explained variance, so nothing fails.

**Spanning rows.** An episode is one row only under ``concat``; ``no_concat`` gives each
turn its own row and ``compact`` starts a new one at every compaction. An estimator that
scores one row at a time -- which is every estimator verl ships -- then opens each row
with ``nextvalues=0``, asserting that nothing after the row boundary is worth anything.
Training proceeds, the loss curves look ordinary, and the agent is simply never credited
across a turn boundary. ``spans_rows`` is what lets the trainer refuse that pairing at
startup instead.

**Needing a critic.** An estimator that reads ``values`` produces something entirely
different when there is no critic -- not an error, a different algorithm. verl only
builds one automatically for the literal estimator name ``"gae"``, so every estimator
here has to say so and the trainer has to check.

An estimator declares all three at the point where it registers itself, so the declaration
and the registration cannot disagree. ``tests/test_advantage_registry.py`` additionally
asserts that every estimator whose implementation mentions ``IGNORE_RETURN`` has
actually declared it.
"""

from __future__ import annotations

from typing import Callable

from verl.trainer.ppo.core_algos import register_adv_est

# Estimator names whose `returns` contain IGNORE_RETURN at unsupervised positions.
SENTINEL_RETURN_ESTIMATORS: set[str] = set()

# Estimator names that stitch an episode's rows back together before scoring it.
TRAJECTORY_ESTIMATORS: set[str] = set()

# Estimator names that read the critic's values and are meaningless without them.
CRITIC_ESTIMATORS: set[str] = set()

# Estimator names that combine a per-token recursion with a per-turn one, and so are only
# defined at gamma == 1. See `requires_undiscounted`.
UNDISCOUNTED_ESTIMATORS: set[str] = set()

# Estimator names whose outer chain has exactly one reward slot per turn, and which
# therefore require a turn's reward to sit on the turn's last token.
TURN_LUMPED_REWARD_ESTIMATORS: set[str] = set()

# Estimator names that publish a `turn_id` column alongside the advantage.
#
# ★ Not the same set as TRAJECTORY_ESTIMATORS, though it was assumed to be. Stitching an
# episode's rows together and *locating its turns* are different jobs: `trajectory_grpo`
# does the first and not the second -- it returns a bare AdvantageOutputs rather than
# going through `_Packed.emit`. The turn-level losses need the column, so keyed off
# `spans_rows` they accepted trajectory_grpo at startup and then raised inside the first
# backward pass, which is the failure that guard exists to pre-empt.
PUBLISHES_TURN_ID: set[str] = set()


def register_trajectory_adv_est(
    name: str,
    *,
    needs_critic: bool = False,
    undiscounted: bool = False,
    turn_lumped_reward: bool = False,
    publishes_turn_id: bool = True,
) -> Callable:
    """Register an estimator that scores a whole episode, however its rows are laid out.

    ``publishes_turn_id`` defaults True because all but one do; set it False for an
    estimator that returns a bare ``AdvantageOutputs`` instead of going through
    ``_Packed.emit``.
    """

    def decorator(fn):
        TRAJECTORY_ESTIMATORS.add(name)
        if publishes_turn_id:
            PUBLISHES_TURN_ID.add(name)
        if needs_critic:
            CRITIC_ESTIMATORS.add(name)
        if undiscounted:
            UNDISCOUNTED_ESTIMATORS.add(name)
        if turn_lumped_reward:
            TURN_LUMPED_REWARD_ESTIMATORS.add(name)
        return register_adv_est(name)(fn)

    return decorator


def register_sentinel_adv_est(
    name: str,
    *,
    needs_critic: bool = False,
    undiscounted: bool = False,
    turn_lumped_reward: bool = False,
    publishes_turn_id: bool = True,
) -> Callable:
    """Register a trajectory estimator that additionally writes sentinel returns.

    Same contract as verl's ``register_adv_est``, and additionally records the name so
    the trainer can decide to compute ``value_mask`` without hard-coding anything.
    """

    def decorator(fn):
        SENTINEL_RETURN_ESTIMATORS.add(name)
        return register_trajectory_adv_est(
            name,
            needs_critic=needs_critic,
            undiscounted=undiscounted,
            turn_lumped_reward=turn_lumped_reward,
        )(fn)

    return decorator


def _name_of(adv_estimator) -> str:
    """Accepts a str or verl's ``AdvantageEstimator`` enum (whose members are str-valued)."""
    return str(getattr(adv_estimator, "value", adv_estimator))


def needs_value_mask(adv_estimator) -> bool:
    """Whether this estimator's returns require a ``value_mask``."""
    return _name_of(adv_estimator) in SENTINEL_RETURN_ESTIMATORS


def needs_critic(adv_estimator) -> bool:
    """Whether this estimator is meaningless without a critic.

    ★ verl decides whether to build one from ``critic.enable``, and when that is unset it
    falls back to ``adv_estimator == "gae"`` -- the *literal string*. Every estimator here
    fails that test, so an unset ``critic.enable`` disables the critic, ``values`` becomes
    zeros, and GAE quietly degenerates into a whitened discounted reward sum. The run
    comes up, uses half the memory, trains faster, and the only evidence is a warning that
    reads "Disabled critic as algorithm.adv_estimator != gae".
    """
    return _name_of(adv_estimator) in CRITIC_ESTIMATORS


def requires_undiscounted(adv_estimator) -> bool:
    """Whether this estimator is only defined at ``algorithm.gamma == 1``.

    ★ An estimator that runs one recursion per token and another per turn discounts the
    same span of trajectory twice over, by two different clocks. Crossing one turn costs
    the turn-level chain a single ``gamma``; it costs the token-level chain
    ``gamma ** (tokens in that turn)``. The two agree only at ``gamma == 1``.

    The divergence is not a rounding error and it is not bounded by the turn count -- it
    is set by how much the model wrote. At ``gamma = 0.99`` a 200-token turn bootstraps
    ``0.99 ** 200 = 0.134`` where the turn level uses ``0.99``, an over-weighting of 7.5x;
    a 500-token turn gives ``0.0066``, over-weighted 152x. The *effective* horizon
    therefore becomes a function of the policy's verbosity, which the policy is free to
    change during training. Measured relative gradient error against an exact policy
    gradient: 0.11% at gamma 0.999, 1.06% at 0.99, 4.9% at 0.95.

    Nothing about this fails loudly, which is why it is a startup assertion rather than a
    documented caveat: ``gamma`` has a perfectly ordinary default and every curve keeps
    its shape.
    """
    return _name_of(adv_estimator) in UNDISCOUNTED_ESTIMATORS


def wants_turn_lumped_reward(adv_estimator) -> bool:
    """Whether this estimator reads a turn's reward only at the turn's last token.

    ★ Declarative, not a lever. An estimator whose outer chain has a single reward slot
    per turn -- ``bi_level_gae`` reads the reward only at each turn-final token -- would
    otherwise credit a mid-turn reward once through the inner token chain and again
    through the outer turn chain: measured bias 0.177 against an exact policy gradient,
    and a critic fixed-point error of exactly the misplaced weight. The estimators with a
    reward slot per *token* want the opposite, because a lumped score has to be remembered
    by ``V`` for the rest of the turn (measured -28% variance at lam 0.9 from per-span,
    -45% at 0.8).

    Neither failure raises. An estimator that needs the lumped shape therefore builds it
    itself, from the per-span rewards the environment paid -- see
    ``_Packed.rewards_lumped_to_turn_end``. Environments do not know which estimator will
    read them, and nothing asks them to.
    """
    return _name_of(adv_estimator) in TURN_LUMPED_REWARD_ESTIMATORS


def publishes_turn_id(adv_estimator) -> bool:
    """Whether this estimator emits the `turn_id` column the turn-level losses read."""
    return _name_of(adv_estimator) in PUBLISHES_TURN_ID


def spans_rows(adv_estimator) -> bool:
    """Whether this estimator carries its recursion across an episode's rows.

    False for everything verl ships: those score one row and stop, which is correct only
    when a row is a whole episode.
    """
    return _name_of(adv_estimator) in TRAJECTORY_ESTIMATORS
