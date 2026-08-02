# All comments are in English.
"""Guards against the class of bug that silently broke every no-concat PPO run.

The original code decided whether to compute ``value_mask`` from a hard-coded list::

    if self.config.algorithm.adv_estimator in ["no_concat_gae_last", "no_concat_gae_first"]:

``no_concat_gae_first`` never existed -- the registered name is ``no_concat_gae``, and
that is what every no-concat script passes. So ``value_mask`` was never produced and
the critic regressed towards the -100 sentinel on nearly every token. Measured on
sokoban: ``critic/vf_loss`` 568 -> ... -> 1.2e-4 (fitting the constant), against ~0.5
for the concat control.

``test_every_sentinel_writing_estimator_is_declared`` is the test that would have
caught it: it reads the estimator sources and fails if one writes IGNORE_RETURN
without declaring itself.
"""

import ast
import inspect

import pytest

from vagen.custom_advantage import SENTINEL_RETURN_ESTIMATORS, needs_value_mask
from vagen.custom_advantage import no_concat_gae as impl


def test_known_estimators_are_registered():
    assert SENTINEL_RETURN_ESTIMATORS == {"no_concat_gae", "no_concat_gae_last"}


@pytest.mark.parametrize("name", ["no_concat_gae", "no_concat_gae_last"])
def test_sentinel_estimators_need_value_mask(name):
    """★ Regression test for the actual bug: `no_concat_gae` -- the name every script
    uses -- must be recognised."""
    assert needs_value_mask(name) is True


@pytest.mark.parametrize("name", ["gae", "grpo", "reinforce_plus_plus"])
def test_plain_estimators_do_not(name):
    assert needs_value_mask(name) is False


def test_typo_does_not_silently_pass():
    """The stale name from the original list must read as False, not as a near-miss
    that some fuzzy match would accept."""
    assert needs_value_mask("no_concat_gae_first") is False


def test_accepts_enum_like_values():
    """`config.algorithm.adv_estimator` may arrive as verl's AdvantageEstimator enum."""

    class _Enum:
        value = "no_concat_gae"

    assert needs_value_mask(_Enum()) is True


def test_every_sentinel_writing_estimator_is_declared():
    """★ The structural guard.

    Any function in no_concat_gae.py that writes IGNORE_RETURN / ignore_value into
    `returns` must be registered via @register_sentinel_adv_est. If someone adds a
    third variant and registers it with plain @register_adv_est, this fails.
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
        decorators = {ast.unparse(d) for d in node.decorator_list}
        declared = any("register_sentinel_adv_est" in d for d in decorators)
        registered_plain = any(
            "register_adv_est" in d and "sentinel" not in d for d in decorators
        )
        if registered_plain and not declared:
            offenders.append(node.name)

    assert not offenders, (
        f"{offenders} write sentinel returns but register with plain @register_adv_est; "
        "use @register_sentinel_adv_est or the critic will train on the sentinel"
    )
