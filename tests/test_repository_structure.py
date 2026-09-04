"""Structural contracts for the package layout."""

from __future__ import annotations

import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "vagen"


def test_extension_axes_have_facades_common_code_and_implementation_directories():
    expected = {
        "algorithms": [
            "default_gae",
            "token_level_gae",
            "trajectory_grpo",
            "turn_level_gae",
        ],
        "harness": ["concat", "no_concat", "compact"],
        "envs": ["sokoban", "frozenlake", "navigation", "primitive_skill", "spatial_gym", "remote"],
    }
    for axis, implementations in expected.items():
        axis_dir = PACKAGE / axis
        assert (axis_dir / "__init__.py").is_file()
        assert (axis_dir / "_common" / "__init__.py").is_file()
        for implementation in implementations:
            assert (axis_dir / implementation / "__init__.py").is_file()


def test_invariant_packages_are_explicitly_named():
    for package in ("rollout", "training", "evaluation", "models"):
        assert (PACKAGE / package / "__init__.py").is_file()


LEGACY_PATHS = (
    "core",
    "agent_loop",
    "trainer",
    "custom_advantage",
    "custom_filter",
    "custom_metric",
    "custom_loss",
    "evaluate",
    "envs_remote",
    "main_ppo.py",
    "gym_agent_dataset.py",
    "envs/gym_base_env.py",
    "envs/gym_image_env.py",
    "envs/state_reward.py",
    "envs/turn_limit.py",
    "harness/budget.py",
    "utils/image_token_utils.py",
)


def test_legacy_compatibility_paths_are_removed():
    remaining = [path for path in LEGACY_PATHS if (PACKAGE / path).exists()]
    assert not remaining, f"legacy compatibility paths remain: {remaining}"


def test_reward_sources_live_under_the_environment_axis():
    legacy = PACKAGE / "rewards"
    assert not list(legacy.glob("*.py"))
    assert (PACKAGE / "envs" / "_common" / "rewards" / "state.py").is_file()


def test_canonical_code_does_not_import_legacy_packages():
    legacy = (
        "vagen.core",
        "vagen.agent_loop",
        "vagen.trainer",
        "vagen.custom_advantage",
        "vagen.custom_filter",
        "vagen.custom_metric",
        "vagen.custom_loss",
        "vagen.evaluate",
        "vagen.envs_remote",
    )
    canonical = ("algorithms", "envs", "evaluation", "harness", "models", "rollout", "training")
    offenders = []
    for package in canonical:
        for path in (PACKAGE / package).rglob("*.py"):
            tree = ast.parse(path.read_text(), filename=str(path))
            for node in ast.walk(tree):
                modules = []
                if isinstance(node, ast.ImportFrom) and node.module:
                    modules.append(node.module)
                elif isinstance(node, ast.Import):
                    modules.extend(alias.name for alias in node.names)
                for module in modules:
                    if module.startswith(legacy):
                        offenders.append(f"{path.relative_to(ROOT)} imports {module}")
    assert not offenders, "\n".join(offenders)


def test_dynamic_training_paths_use_the_canonical_training_package():
    agent_config = (PACKAGE / "configs" / "agent_v2.yaml").read_text()
    flags = (PACKAGE / "configs" / "baseline_vllm.flags").read_text()
    run_config = (PACKAGE / "configs" / "vagen_multiturn.yaml").read_text()
    assert "vagen.training.agent_loop.gym_loop.GymLoop" in agent_config
    assert "vagen.training.agent_loop.multi_output.MultiOutputAgentLoopManager" in flags
    assert "vagen/training/dataset.py" in run_config


def test_algorithm_registry_points_to_implementation_packages():
    from vagen.algorithms import ALGORITHMS, registered_algorithms

    expected = {
        "bi_level_gae",
        "default_gae",
        "token_level_gae",
        "trajectory_grpo",
        "turn_level_gae",
    }
    assert set(registered_algorithms()) == expected
    for name, spec in ALGORITHMS.items():
        assert spec.implementation.__module__.startswith(f"vagen.algorithms.{name}.")


def test_algorithm_directories_own_their_implementations():
    """Concrete packages must contain the algorithm, not a forwarding wrapper."""
    assert not (PACKAGE / "algorithms" / "_common" / "trajectory_algos.py").exists()
    assert (PACKAGE / "algorithms" / "_common" / "packing.py").is_file()

    for path in (PACKAGE / "algorithms").glob("*/*.py"):
        if path.parent.name == "_common" or path.name == "__init__.py":
            continue
        tree = ast.parse(path.read_text(), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module and "._common" in node.module:
                delegated = [alias.name for alias in node.names if alias.name.startswith("_compute_")]
                assert not delegated, f"{path.relative_to(ROOT)} delegates to _common: {delegated}"
