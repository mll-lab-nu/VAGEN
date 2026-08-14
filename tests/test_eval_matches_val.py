"""An eval number is read against the val curve, so the two configs have to agree.

Nothing pinned this. `test_train_and_val_agree_on_rewards` globs only `examples/train/`,
so every drift between an eval config and the val config it will be compared with passed
the suite: primitive_skill evaluated with a 67% larger turn budget, sokoban let the policy
write twice the tokens it was trained to, and both frozenlake and primitive_skill drew
their eval seeds from the training range -- 127 of 128 and 16 of 16 respectively, which
makes the reported number training-set accuracy.
"""

from __future__ import annotations

import glob
import os

import pytest
from omegaconf import OmegaConf

#: Anchored to the repo. A relative glob at module scope silently matched nothing from any
#: other working directory, so the whole file passed while checking zero configs.
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EVAL = sorted(glob.glob(os.path.join(_ROOT, "examples/evaluate/*/config.yaml")))
assert EVAL, "no eval configs found; the glob is wrong, not the repo empty"
#: eval directory -> the val yaml its numbers are read against.
PAIRS = {
    "sokoban": "examples/train/sokoban/val_sokoban_vision.yaml",
    "frozenlake": "examples/train/frozenlake/val_frozenlake_vision.yaml",
    "primitive_skill": "examples/train/primitive_skill/val_primitive_skill.yaml",
}

#: Eval directories deliberately not paired, with the reason. An allowlist with no
#: completeness check is how navigation and spatial_gym went unexamined while both drew
#: their eval seeds from the training range.
UNPAIRED = {
    # Trains on `base_train` (1200 tasks) and scores on `base` (60): the seed numbers
    # index different datasets, so a range comparison is meaningless.
    "navigation": "train and val use different eval_set datasets",
    # The seed is a room index into a fixed download, and eval deliberately covers all 20
    # rooms rather than only the held-out four.
    "spatial_gym": "seed indexes a fixed room set, not a generator",
}


def test_every_eval_config_is_either_paired_or_deliberately_not():
    """So a new eval directory cannot quietly escape the checks below."""
    seen = {os.path.basename(os.path.dirname(p)) for p in EVAL}
    unexplained = seen - set(PAIRS) - set(UNPAIRED)
    assert not unexplained, (
        f"{sorted(unexplained)} have no val counterpart and no stated reason; add them to "
        f"PAIRS or to UNPAIRED with why")


def _envs(path):
    if not os.path.isabs(path):
        path = os.path.join(_ROOT, path)
    return OmegaConf.to_container(OmegaConf.load(path), resolve=False).get("envs", []) or []


def _seed_values(spec):
    seed = spec.get("seed")
    if not isinstance(seed, list) or len(seed) < 2:
        return set()
    return set(range(int(seed[0]), int(seed[1]) + 1))


@pytest.mark.parametrize("name,val_path", sorted(PAIRS.items()))
def test_eval_turn_budget_matches_the_val_config(name, val_path):
    """max_turns is the dominant driver of success rate on these tasks; a different budget
    makes the two numbers different measurements, not a comparison."""
    val = _envs(val_path)[0]
    for spec in _envs(f"examples/evaluate/{name}/config.yaml"):
        assert spec.get("max_turns") == val.get("max_turns"), (
            f"{name} eval runs {spec.get('max_turns')} turns against val's "
            f"{val.get('max_turns')}")


@pytest.mark.parametrize("name,val_path", sorted(PAIRS.items()))
def test_eval_seeds_do_not_overlap_the_training_seeds(name, val_path):
    """★ Same seed, same instance. An eval range that reaches into the train range reports
    training-set accuracy under an evaluation heading, and nothing about the run says so."""
    train_path = os.path.join(_ROOT, val_path.replace("/val_", "/train_"))
    if not os.path.exists(train_path):
        pytest.skip(f"no train counterpart for {name}")
    train = set().union(*(_seed_values(s) for s in _envs(train_path)))
    for spec in _envs(f"examples/evaluate/{name}/config.yaml"):
        overlap = _seed_values(spec) & train
        assert not overlap, (
            f"{name} eval draws {len(overlap)} seed(s) the training config also uses, "
            f"e.g. {sorted(overlap)[:5]}")


def test_every_eval_seed_range_covers_itself_exactly():
    """`[a,b,1]` samples `n_envs` from the INCLUSIVE range, so n_envs short of the range
    size drops a task at random on every run -- and where the env indexes its dataset as
    `seed % len`, the surplus wraps onto task 0 and scores it twice."""
    for path in EVAL:
        for spec in _envs(path):
            seed = spec.get("seed")
            if isinstance(seed, list) and len(seed) == 3 and seed[2] == 1:
                size = int(seed[1]) - int(seed[0]) + 1
                assert size == int(spec["n_envs"]), (
                    f"{path}: seed range holds {size} values for n_envs={spec['n_envs']}")
