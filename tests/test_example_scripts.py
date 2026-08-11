"""Every launch script under examples/train must name things that exist.

These are the entry points a new user copies, and nothing else checks them: a renamed
estimator, a required hyperparameter left unset, or a state reward switched on for an
environment that has no spec all fail only once a cluster is up and a rollout has run.
"""

from __future__ import annotations

import glob
import os
import re

import pytest

import vagen.custom_advantage  # noqa: F401  registers the estimators
from vagen.custom_advantage import TRAJECTORY_ESTIMATORS

SCRIPTS = sorted(glob.glob("examples/train/*/*.sh"))
VERL_OWN = {"gae", "grpo", "reinforce_plus_plus", "rloo", "remax"}


def _flag(text, name):
    m = re.search(rf"{re.escape(name)}=([^\s\\]+)", text)
    return m.group(1) if m else None


def test_there_are_scripts_to_check():
    assert SCRIPTS, "found no example scripts; this test is silently vacuous"


@pytest.mark.parametrize("path", SCRIPTS, ids=lambda p: "/".join(p.split("/")[-2:]))
def test_the_estimator_exists(path):
    """★ A renamed estimator turns every script that used it into a run that dies at
    startup. episode_gae -> vanilla_gae and removed_estimator_gae_paper -> removed_estimator_gae both
    happened; this is what would have caught a missed one."""
    est = _flag(open(path).read(), "algorithm.adv_estimator")
    if est is None:
        pytest.skip("script sets no estimator")
    assert est in TRAJECTORY_ESTIMATORS or est in VERL_OWN, (
        f"{path} selects adv_estimator={est!r}, which is neither registered by VAGEN "
        f"{sorted(TRAJECTORY_ESTIMATORS)} nor one of verl's own"
    )


@pytest.mark.parametrize("path", SCRIPTS, ids=lambda p: "/".join(p.split("/")[-2:]))
def test_estimators_with_a_required_hyperparameter_set_it(path):
    """`removed_estimator_gae` reads `removed_estimator`, and at removed_estimator == gamma the two
    passes telescope and it *is* token_level_gae. A script that omits it reproduces the
    wrong algorithm silently, so the scripts state it."""
    text = open(path).read()
    if _flag(text, "algorithm.adv_estimator") != "removed_estimator_gae":
        return
    assert "removed_estimator" in text, (
        f"{path} runs removed_estimator_gae without removed_estimator; it degenerates to "
        "token_level_gae when the two clocks agree"
    )


@pytest.mark.parametrize("path", SCRIPTS, ids=lambda p: "/".join(p.split("/")[-2:]))
def test_state_reward_is_only_switched_on_where_a_spec_exists(path):
    """★ STATE_REWARD_SPECS has Sokoban and nothing else, and `_maybe_state_reward`
    raises on a missing spec. Turning it on elsewhere is a run that dies on the first
    step."""
    from vagen.agent_loop.gym_loop import STATE_REWARD_SPECS

    text = open(path).read()
    if "state_reward" not in text or ".enable=True" not in text:
        return
    env_dir = os.path.basename(os.path.dirname(path))
    ok = {name.lower() for name in STATE_REWARD_SPECS}
    assert env_dir.replace("_", "") in {n.replace("_", "") for n in ok}, (
        f"{path} enables a state reward, but {env_dir!r} has no entry in "
        f"STATE_REWARD_SPECS ({sorted(STATE_REWARD_SPECS)})"
    )


@pytest.mark.parametrize("path", SCRIPTS, ids=lambda p: "/".join(p.split("/")[-2:]))
def test_the_yaml_it_points_at_exists(path):
    """A script referring to a data yaml that was renamed fails at load, not at parse."""
    text = open(path).read()
    for key in ("data.train_files", "data.val_files"):
        val = _flag(text, key)
        if not val:
            continue
        val = val.strip("\"'")
        if "$" in val:      # built from a variable; not resolvable here
            continue
        assert os.path.exists(val), f"{path}: {key}={val} does not exist"
