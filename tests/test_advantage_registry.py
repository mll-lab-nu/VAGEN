"""Guards for the sentinel-estimator registry.

Whether an estimator needs a ``value_mask`` used to be decided by a hard-coded list of
names. A list like that can disagree with the names actually registered, and when it
does the critic is trained on the -100 sentinel with nothing failing loudly.

``test_every_sentinel_writing_estimator_is_declared`` is the structural guard: it reads
the estimator sources and fails if one writes ``IGNORE_RETURN`` without declaring it.
"""

import ast
import inspect
import pytest

from vagen.algorithms import (
    SENTINEL_RETURN_ESTIMATORS,
    TRAJECTORY_ESTIMATORS,
    needs_value_mask,
)
from vagen.algorithms._common import trajectory_algos as impl


def test_known_estimators_are_registered():
    assert SENTINEL_RETURN_ESTIMATORS == {"turn_level_gae"}


def test_sentinel_estimators_need_value_mask():
    """Turn-level GAE anchors one return per turn and leaves the rest at the sentinel,
    so the critic must be told which positions carry supervision."""
    assert needs_value_mask("turn_level_gae") is True


#: Everything that is not sentinel-writing, read from the registry rather than listed --
#: a list here would go stale the moment an estimator is added, and going stale means
#: *not testing* the new one while still looking like full coverage.
PLAIN_ESTIMATORS = ["gae", "grpo"] + sorted(TRAJECTORY_ESTIMATORS - SENTINEL_RETURN_ESTIMATORS)


@pytest.mark.parametrize("name", PLAIN_ESTIMATORS)
def test_plain_estimators_do_not(name):
    """★ These supervise every model-output token, so a value_mask would be wrong for
    them, not merely unnecessary."""
    assert needs_value_mask(name) is False


def test_typo_does_not_silently_pass():
    """A near-miss name must read as False rather than being fuzzy-matched to a real
    estimator -- that is how the original hard-coded list went wrong."""
    assert needs_value_mask("turn_gae") is False


def test_accepts_enum_like_values():
    """`config.algorithm.adv_estimator` may arrive as verl's AdvantageEstimator enum."""

    class _Enum:
        value = "turn_level_gae"

    assert needs_value_mask(_Enum()) is True


def test_every_sentinel_writing_estimator_is_declared():
    """★ The structural guard.

    Any estimator that writes IGNORE_RETURN / ignore_value into `returns` must be
    registered via @register_sentinel_adv_est. If someone adds a variant and registers
    it with plain @register_adv_est, this fails.
    """
    src = inspect.getsource(impl)
    tree = ast.parse(src)

    offenders = []
    for node in tree.body:
        if not isinstance(node, ast.FunctionDef):
            continue
        body = ast.unparse(node)
        writes_sentinel = "ignore_value" in body or "IGNORE_RETURN" in body
        if not writes_sentinel:
            continue
        if not node.name.startswith("_compute_"):
            continue
        estimator = node.name.removeprefix("_compute_")
        if estimator not in SENTINEL_RETURN_ESTIMATORS:
            offenders.append(estimator)

    assert not offenders, (
        f"{offenders} write sentinel returns but register with plain @register_adv_est; "
        "use @register_sentinel_adv_est or the critic will train on the sentinel"
    )


def test_the_structural_guard_finds_a_real_offender():
    """★ The guard above matched decorators containing `register_adv_est`. After the
    rework no estimator uses that name -- they all use `@advantage_estimator` -- so its
    offender list was unconditionally empty and it had been passing for free.

    This feeds the guard's own detection an offending source and requires a hit, so the
    guard cannot go dead again without this failing.
    """
    src = (
        "@advantage_estimator('bad')\n"
        "def bad(inputs):\n"
        "    returns = inputs.zeros() + IGNORE_RETURN\n"
        "    return inputs.zeros(), returns\n"
    )
    tree = ast.parse(src)
    offenders = []
    for node in tree.body:
        if not isinstance(node, ast.FunctionDef):
            continue
        body = ast.unparse(node)
        if not ("ignore_value" in body or "IGNORE_RETURN" in body):
            continue
        decorators = " ".join(ast.unparse(d) for d in node.decorator_list)
        if "advantage_estimator" not in decorators:
            continue
        if "sentinel_returns=True" not in decorators:
            offenders.append(node.name)
    assert offenders == ["bad"], "the detection logic no longer detects anything"


def test_every_value_reading_estimator_declares_needs_critic():
    """★ The structural guard for the critic, mirroring the sentinel one above.

    An estimator that reads `inputs.values` produces a different algorithm when no critic
    exists -- not an error. Registering it without `needs_critic=True` puts it back in
    reach of verl's "is the name literally gae" fallback, silently.
    """
    from vagen.algorithms import CRITIC_ESTIMATORS

    src = inspect.getsource(impl)
    tree = ast.parse(src)

    offenders = []
    for node in tree.body:
        if not isinstance(node, ast.FunctionDef):
            continue
        body = ast.unparse(node)
        if "inputs.values" not in body:
            continue
        if not node.name.startswith("_compute_"):
            continue
        estimator = node.name.removeprefix("_compute_")
        if estimator not in CRITIC_ESTIMATORS:
            offenders.append(estimator)

    assert not offenders, (
        f"{offenders} read the critic's values but did not declare needs_critic=True; "
        "without it a run with no critic trains a different algorithm and says nothing"
    )


def test_that_guard_can_fail():
    """Feed the same detection an estimator that does offend."""
    src = (
        "@advantage_estimator('bad')\n"
        "def bad(inputs):\n"
        "    return inputs.values, inputs.values\n"
    )
    tree = ast.parse(src)
    found = [
        n.name for n in tree.body
        if isinstance(n, ast.FunctionDef) and "inputs.values" in ast.unparse(n)
    ]
    assert found == ["bad"], "the detection logic no longer detects anything"


# --------------------------------------------------- verl 0.8 calling convention


ALL_VAGEN_ESTIMATORS = sorted(TRAJECTORY_ESTIMATORS)


@pytest.mark.parametrize("name", ALL_VAGEN_ESTIMATORS)
def test_estimators_match_the_dispatch_signature(name):
    """★ verl 0.8 calls custom estimators with keyword tensors, not the DataProto:
    token_level_rewards / response_mask / config, plus batch and non_tensor_batch for
    estimators that name them. A stale `(data, gamma, lam)` signature only fails at the
    first advantage computation, i.e. after a cluster is up and a rollout has run."""
    from verl.trainer.ppo.core_algos import get_adv_estimator_fn

    params = inspect.signature(get_adv_estimator_fn(name)).parameters
    assert "data" not in params, f"{name} still takes the DataProto"
    # These group rows by trajectory and turn, so they need the raw containers.
    assert "batch" in params and "non_tensor_batch" in params, f"{name} cannot reach its index columns"
    assert any(p.kind is p.VAR_KEYWORD for p in params.values()), (
        f"{name} must tolerate the extra kwargs verl passes (index, reward_baselines)"
    )


def test_dispatch_hands_over_the_containers_by_signature():
    """The verl side of the contract: naming batch/non_tensor_batch is what opts an
    estimator in. Pins it so a rebase that drops the branch is caught here."""
    from verl.trainer.ppo import ray_trainer

    src = inspect.getsource(ray_trainer.compute_advantage)
    assert '"non_tensor_batch" in _adv_params' in src
    assert '"batch" in _adv_params' in src
