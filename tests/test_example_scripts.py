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
    startup. episode_gae -> default_gae and removed_estimator_gae_paper -> removed_estimator_gae both
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
    """★ Only Sokoban declares a STATE_REWARD_SPEC, and `_maybe_state_reward`
    raises when the environment declares none. Turning it on elsewhere is a run that dies on the first
    step."""
    import vagen.envs.registry as R
    from vagen.envs.state_reward import state_reward_spec_of

    R._load_registry()
    STATE_REWARD_SPECS = {n: c for n, c in R._ENV_REGISTRY.items() if state_reward_spec_of(c)}

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


# --------------------------------------------- an environment declares its own capability


def test_the_capability_lives_on_the_environment_not_in_a_table():
    """★ The point of moving the spec next to the environment.

    `STATE_REWARD_SPECS = {"Sokoban": ...}` in the agent loop meant the loop had to be
    edited whenever an environment gained the capability, the environment could not be
    read to find out whether it had it, and a name spelled one way in the registry and
    another in the table failed only once a run was up. Sokoban now declares
    `STATE_REWARD_SPEC` and the loop asks the class.
    """
    import inspect

    import vagen.agent_loop.gym_loop as loop
    from vagen.envs.sokoban.sokoban_env import Sokoban
    from vagen.envs.state_reward import state_reward_spec_of, supports_state_reward

    assert supports_state_reward(Sokoban), "Sokoban no longer declares its spec"
    assert state_reward_spec_of(Sokoban).object_weights, "the declared spec is empty"

    assert not hasattr(loop, "STATE_REWARD_SPECS"), (
        "the central table is back; an environment's capability belongs on the environment"
    )
    src = inspect.getsource(loop.GymLoop._maybe_state_reward)
    assert "state_reward_spec_of" in src, "the loop no longer asks the environment class"


def test_env_specific_reward_code_is_not_in_the_generic_package():
    """`vagen/rewards/` is the machinery -- judge client, F1, spans, wrapper. Anything
    that knows what a *box* is belongs next to the environment that has boxes."""
    import os

    generic = set(os.listdir("vagen/rewards"))
    for env_name in ("sokoban", "frozenlake", "navigation", "primitive_skill", "spatial_gym"):
        assert f"{env_name}.py" not in generic, (
            f"vagen/rewards/{env_name}.py is environment-specific; move it under "
            f"vagen/envs/{env_name}/"
        )
    assert os.path.exists("vagen/envs/sokoban/state_reward_spec.py")


# ------------------------------------------- a seed that indexes data, not a generator


#: Envs whose seed selects a fixed on-disk sample rather than seeding a generator, with
#: the directory each seed resolves to. frozenlake and sokoban generate their layouts, so
#: any seed is valid for them and none of this applies.
DATASET_BACKED = {"SpatialGym": lambda seed: f"run{seed:02d}"}

DATA_YAMLS = sorted(glob.glob("examples/train/*/*.yaml"))


def _dataset_specs(path):
    import yaml

    for spec in (yaml.safe_load(open(path)) or {}).get("envs", []) or []:
        if spec.get("name") in DATASET_BACKED:
            yield spec


@pytest.mark.parametrize("path", DATA_YAMLS, ids=lambda p: "/".join(p.split("/")[-2:]))
def test_every_seed_a_config_asks_for_resolves_to_data_that_exists(path):
    """★ For a dataset-backed env the seed is an array index, so a range wider than the
    dataset is a crash rather than more variety.

    val_spatial_gym_vision.yaml asked for [1000,1049] to be disjoint from train, and the
    training half asked for [0,49] against a 20-room download. ImageHandler.load_data
    asserts the directory exists, so both failed -- but only once a rollout had started,
    on a box that took minutes to reach that point.
    """
    import random

    from vagen.gym_agent_dataset import _generate_from_len_three

    for spec in _dataset_specs(path):
        data_dir = spec["config"]["data_dir"]
        if not os.path.isdir(data_dir):
            pytest.skip(f"{data_dir} not downloaded; see the env README")
        lo, hi, limit = spec["seed"]
        naming = DATASET_BACKED[spec["name"]]
        # Check the whole declared range, not a sample of it: the sampler is random, so
        # sampling would make this test flaky in exactly the case it is meant to catch.
        missing = [s for s in range(lo, hi + 1) if not os.path.isdir(os.path.join(data_dir, naming(s)))]
        assert not missing, (
            f"{path}: seed range [{lo},{hi}] includes {missing[:5]} which have no data "
            f"under {data_dir}"
        )
        assert (hi - lo + 1) * limit >= spec["n_envs"], (
            f"{path}: {hi - lo + 1} seeds used {limit}x cannot supply n_envs="
            f"{spec['n_envs']}; the dataset builder raises rather than sampling fewer"
        )


def test_train_and_val_do_not_share_a_dataset_sample():
    """★ Held-out means held out. These were identical ranges once, and validation was
    reporting training performance -- a number that looks like generalisation and is not.
    """
    import collections

    by_env = collections.defaultdict(dict)
    for path in DATA_YAMLS:
        kind = "val" if os.path.basename(path).startswith("val") else "train"
        for spec in _dataset_specs(path):
            lo, hi, _ = spec["seed"]
            by_env[(os.path.dirname(path), spec["name"])][kind] = set(range(lo, hi + 1))

    checked = 0
    for (where, env), halves in by_env.items():
        if len(halves) < 2:
            continue
        checked += 1
        overlap = sorted(halves["train"] & halves["val"])
        assert not overlap, f"{where}: {env} trains and validates on samples {overlap[:5]}"
    assert checked, "no env had both a train and a val yaml; this test checked nothing"
