"""Every launch script under examples/train must name things that exist.

These are the entry points a new user copies, and nothing else checks them: a renamed
estimator, a required hyperparameter left unset, or a state reward switched on for an
environment that has no spec all fail only once a cluster is up and a rollout has run.
"""

from __future__ import annotations

import glob
import os
import re
import subprocess

import pytest

import vagen.algorithms  # noqa: F401  registers the estimators
from vagen.algorithms import TRAJECTORY_ESTIMATORS

SCRIPTS = sorted(glob.glob("examples/train/*/*.sh"))
ALL_SHELL_SCRIPTS = sorted(
    glob.glob("examples/train/**/*.sh", recursive=True)
    + glob.glob("examples/evaluate/**/*.sh", recursive=True)
)
VERL_OWN = {"gae", "grpo", "reinforce_plus_plus", "rloo", "remax"}


def _flag(text, name):
    """The value a script actually passes for `name`, ignoring anything commented out.

    Comment lines are stripped first because these scripts document their flags. Reading
    a comment as configuration can report a documented alternative as the live value, or
    treat a flag someone commented out as still enabled.
    """
    live = "\n".join(ln for ln in text.splitlines() if not ln.lstrip().startswith("#"))
    m = re.search(rf"{re.escape(name)}=([^\s\\]+)", live)
    return m.group(1) if m else None


def test_there_are_scripts_to_check():
    assert SCRIPTS, "found no example scripts; this test is silently vacuous"


def test_sglang_is_the_default_rollout_engine():
    defaults = open("vagen/configs/training_defaults.flags").read()
    installer = open("scripts/install.sh").read()
    assert "actor_rollout_ref.rollout.name=sglang" in defaults
    assert "actor_rollout_ref.rollout.free_cache_engine=False" in defaults
    assert 'export BACKEND=${BACKEND:-sglang}' in installer
    for path in SCRIPTS:
        text = open(path).read()
        assert "vagen/configs/training_defaults.flags" in text
        assert "actor_rollout_ref.rollout.free_cache_engine=True" not in text


@pytest.mark.parametrize(
    "environment,launcher",
    [
        ("frozenlake", "eval_qwen25_vl_3b.sh"),
        ("navigation", "eval_qwen25_vl_7b.sh"),
        ("primitive_skill", "eval_qwen25_vl_3b.sh"),
        ("sokoban", "eval_qwen25_vl_3b.sh"),
        ("spatial_gym", "eval_qwen25_vl_3b.sh"),
    ],
)
def test_each_environment_has_both_local_engine_launchers(environment, launcher):
    for backend in ("sglang", "vllm"):
        assert os.path.isfile(f"examples/evaluate/{environment}/{backend}/{launcher}")


def test_sglang_eval_launchers_are_seeded_and_deterministic():
    launchers = glob.glob("examples/evaluate/*/sglang/eval_*.sh")
    assert len(launchers) == 5
    for path in launchers:
        text = open(path).read()
        assert "--random-seed" in text, path
        assert "--enable-deterministic-inference" in text, path


@pytest.mark.parametrize("path", ALL_SHELL_SCRIPTS)
def test_every_example_shell_script_parses(path):
    subprocess.run(["bash", "-n", path], check=True)


@pytest.mark.parametrize("path", SCRIPTS, ids=lambda p: "/".join(p.split("/")[-2:]))
def test_training_examples_use_canonical_entrypoints(path):
    text = open(path).read()
    assert "-m vagen.training.main" in text
    assert "vagen/training/dataset.py" in text


def test_examples_do_not_reference_removed_packages():
    legacy = re.compile(
        r"vagen(?:\.|/)(?:core|agent_loop|trainer|custom_advantage|custom_filter|"
        r"custom_metric|custom_loss|evaluate|envs_remote|main_ppo|gym_agent_dataset)"
    )
    offenders = []
    shipped = glob.glob("examples/train/**/*", recursive=True)
    shipped += glob.glob("examples/evaluate/**/*", recursive=True)
    for path in shipped:
        if not os.path.isfile(path):
            continue
        try:
            text = open(path).read()
        except UnicodeDecodeError:
            continue
        if legacy.search(text):
            offenders.append(path)
    assert not offenders, f"examples reference removed package paths: {offenders}"


@pytest.mark.parametrize("path", SCRIPTS, ids=lambda p: "/".join(p.split("/")[-2:]))
def test_the_estimator_exists(path):
    """A renamed estimator turns every script that used it into a startup failure."""
    est = _flag(open(path).read(), "algorithm.adv_estimator")
    if est is None:
        pytest.skip("script sets no estimator")
    assert est in TRAJECTORY_ESTIMATORS or est in VERL_OWN, (
        f"{path} selects adv_estimator={est!r}, which is neither registered by VAGEN "
        f"{sorted(TRAJECTORY_ESTIMATORS)} nor one of verl's own"
    )


@pytest.mark.parametrize("path", SCRIPTS, ids=lambda p: "/".join(p.split("/")[-2:]))
def test_state_reward_is_only_switched_on_where_a_spec_exists(path):
    """Every dataset that enables state reward must name a capable environment."""
    import vagen.envs.registry as R
    from vagen.envs._common.rewards import state_reward_spec_of
    import yaml

    R._load_registry()
    STATE_REWARD_SPECS = {n: c for n, c in R._ENV_REGISTRY.items() if state_reward_spec_of(c)}

    text = open(path).read()
    for match in re.finditer(r'data\.(?:train|val)_files="?([^"\s\\]+)', text):
        value = match.group(1)
        if value.startswith("$SCRIPTDIR/"):
            yaml_path = os.path.join(os.path.dirname(path), value.removeprefix("$SCRIPTDIR/"))
        elif "$" not in value:
            yaml_path = value
        else:
            continue
        if not os.path.exists(yaml_path):
            continue
        for spec in (yaml.safe_load(open(yaml_path)) or {}).get("envs", []):
            settings = (spec.get("config") or {}).get("state_reward") or {}
            enabled = any((settings.get(n) or {}).get("enable", False)
                          for n in ("state_estimation", "transition_prediction"))
            if enabled:
                assert spec["name"] in STATE_REWARD_SPECS, (
                    f"{yaml_path} enables state reward for {spec['name']!r}, which declares "
                    f"no STATE_REWARD_SPEC; capable envs: {sorted(STATE_REWARD_SPECS)}"
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
    `STATE_REWARD_SPEC` and the shared environment factory asks the class.
    """
    import inspect

    from vagen.envs.sokoban.sokoban_env import Sokoban
    import vagen.envs._common.rewards.factory as state_reward
    from vagen.envs._common.rewards import state_reward_spec_of, supports_state_reward

    assert supports_state_reward(Sokoban), "Sokoban no longer declares its spec"
    assert state_reward_spec_of(Sokoban).object_weights, "the declared spec is empty"

    assert not hasattr(state_reward, "STATE_REWARD_SPECS"), (
        "the central table is back; an environment's capability belongs on the environment"
    )
    src = inspect.getsource(state_reward._with_state_reward)
    assert "state_reward_spec_of" in src, "the factory no longer asks the environment class"


def test_env_specific_reward_code_is_not_in_the_generic_package():
    """The env axis owns shared reward machinery; implementations own their specs."""
    import os

    generic_dir = "vagen/envs/_common/rewards"
    generic = set(os.listdir(generic_dir))
    for env_name in ("sokoban", "frozenlake", "navigation", "primitive_skill", "spatial_gym"):
        assert f"{env_name}.py" not in generic, (
            f"{generic_dir}/{env_name}.py is environment-specific; move it under "
            f"vagen/envs/{env_name}/"
        )
    legacy_dir = "vagen/rewards"
    assert not os.path.isdir(legacy_dir) or not any(
        name.endswith(".py") for name in os.listdir(legacy_dir)
    )
    assert os.path.exists("vagen/envs/sokoban/state_reward_spec.py")


def test_the_state_reward_example_owns_the_judge_lifecycle():
    text = open("examples/train/sokoban/train_default_gae_sr_qwen25vl3b.sh").read()
    assert "scripts/launch_judge.sh" in text
    assert "/health" in text
    assert "trap cleanup_judge EXIT" in text
    assert "train_sokoban_vision_sr.yaml" in text
    assert "val_sokoban_vision_sr.yaml" in text


def test_sokoban_bi_level_state_reward_script_uses_validated_defaults():
    text = open("examples/train/sokoban/train_bi_level_gae_sr_qwen25vl3b.sh").read()
    for setting in (
        "PROJECT_NAME=${PROJECT_NAME:-vagen-experiments}",
        "BI_LEVEL_MIX=${BI_LEVEL_MIX:-0.75}",
        "GAMMA_TURN=${GAMMA_TURN:-0.95}",
        "LAMBDA_TURN=${LAMBDA_TURN:-0.95}",
        "LAMBDA_TOKEN=${LAMBDA_TOKEN:-1.0}",
        "STATE_REWARD_CREDIT_SITE=${STATE_REWARD_CREDIT_SITE:-turn_end}",
        "STATE_REWARD_SCORE_BASE=${STATE_REWARD_SCORE_BASE:-0.625}",
        "STATE_REWARD_AGGREGATION=${STATE_REWARD_AGGREGATION:-episode_mean}",
        "STATE_REWARD_SCORER=${STATE_REWARD_SCORER:-exact}",
        "STATE_ESTIMATION_REWARD=${STATE_ESTIMATION_REWARD:-0.006}",
        "TRANSITION_PREDICTION_REWARD=${TRANSITION_PREDICTION_REWARD:-0.006}",
        "+algorithm.bi_level_mix=\"$BI_LEVEL_MIX\"",
    ):
        assert setting in text


def test_sokoban_state_reward_example_uses_the_small_shaping_budget():
    """State and format shaping stay small relative to the success reward."""
    import yaml

    for name in ("train_sokoban_vision_sr.yaml", "val_sokoban_vision_sr.yaml"):
        path = os.path.join("examples/train/sokoban", name)
        config = yaml.safe_load(open(path))["envs"][0]
        env = config["config"]
        assert config["max_turns"] == 5
        assert config["response_length_per_turn"] == 512
        assert "reward_mode" not in env
        assert env["strict_format"] is True
        assert env["format_reward"] == "${oc.env:SOKOBAN_FORMAT_REWARD,0.03}"
        assert env["state_reward"]["state_estimation"]["reward"] == (
            "${oc.env:STATE_ESTIMATION_REWARD,0.03}"
        )
        assert env["state_reward"]["transition_prediction"]["reward"] == (
            "${oc.env:TRANSITION_PREDICTION_REWARD,0.03}"
        )
        assert env["state_reward"]["score_base"] == (
            "${oc.env:STATE_REWARD_SCORE_BASE,0.334}"
        )
        shaping_cap = config["max_turns"] * (0.03 + 0.03 + 0.03)
        assert shaping_cap == pytest.approx(0.45)


def test_the_judge_launcher_uses_a_toolkit_that_really_has_nvcc():
    """A conda CUDA runtime is not necessarily a compiler toolkit."""
    text = open("scripts/launch_judge.sh").read()
    assert 'torch.version.cuda' in text
    assert '[ ! -x "$CUDA_HOME/bin/nvcc" ]' in text
    assert 'CUDA toolkit with nvcc not found' in text


def test_the_sglang_judge_enables_deterministic_inference():
    text = open("scripts/launch_judge.sh").read()
    sglang_branch = text.split("  sglang)", 1)[1].split("    ;;", 1)[0]
    vllm_branch = text.split("  vllm)", 1)[1].split("    ;;", 1)[0]

    assert '--random-seed "$SEED"' in sglang_branch
    assert "--enable-deterministic-inference" in sglang_branch
    assert "--enable-deterministic-inference" not in vllm_branch


def test_the_sglang_installer_does_not_reference_downstream_repositories():
    text = open("scripts/install_sglang.sh").read().lower()
    assert "viewagent" not in text
    assert "viewsuite" not in text


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

    from vagen.training.dataset import _generate_from_len_three

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


@pytest.mark.parametrize("path", SCRIPTS, ids=lambda p: "/".join(p.split("/")[-2:]))
def test_no_comment_hides_inside_a_line_continuation(path):
    """★ A `#` line after a trailing `\\` is not a comment -- it is arguments.

    These scripts are one long backslash-continued command, so a comment written between
    two flags is passed to hydra word by word. `bash -n` accepts it, the script is still
    executable, and the failure arrives as an unparseable override from a run that has
    already allocated GPUs. Caught while explaining why a flag had been turned off.
    """
    lines = open(path).read().splitlines()
    offenders = [
        (i + 1, ln.strip()[:60])
        for i, ln in enumerate(lines)
        if i and ln.lstrip().startswith("#") and lines[i - 1].rstrip().endswith("\\")
    ]
    assert not offenders, (
        f"{path}: these follow a line continuation, so they are arguments and not "
        f"comments: {offenders}"
    )
