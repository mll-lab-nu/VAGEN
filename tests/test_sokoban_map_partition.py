from pathlib import Path

import numpy as np
import pytest
import yaml
from gym_sokoban.envs.room_utils import generate_room

from vagen.envs.sokoban.patch_sokoban_env import (
    _room_matches_partition,
    _room_partition_bucket,
    get_shortest_action_path,
)
from vagen.envs.sokoban.sokoban_env import SokobanEnvConfig
from vagen.envs.sokoban.utils.seeding import set_seed


ROOT = Path(__file__).resolve().parents[1]
TRAIN_CONFIGS = (
    ROOT / "examples/train/sokoban/train_sokoban_vision.yaml",
    ROOT / "examples/train/sokoban/train_sokoban_vision_sr.yaml",
)
VAL_CONFIGS = (
    ROOT / "examples/train/sokoban/val_sokoban_vision.yaml",
    ROOT / "examples/train/sokoban/val_sokoban_vision_sr.yaml",
)


def _spec(path):
    return yaml.safe_load(path.read_text())["envs"][0]


def test_room_partition_is_stable_and_complementary():
    fixed = np.arange(36, dtype=np.uint8).reshape(6, 6) % 3
    state = np.arange(36, dtype=np.uint8).reshape(6, 6) % 7

    bucket = _room_partition_bucket(fixed, state, 4)
    assert bucket == _room_partition_bucket(fixed.copy(), state.copy(), 4)
    assert _room_matches_partition(fixed, state, "train", 4, 0) != (
        _room_matches_partition(fixed, state, "eval", 4, 0)
    )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"map_partition": "unknown"},
        {"map_partition_modulus": 1},
        {"map_partition_modulus": 4, "map_partition_eval_bucket": 4},
        {"map_partition_modulus": 4.5},
        {"map_partition_eval_bucket": 1.5},
        {"map_partition_modulus": "4.5"},
        {"map_partition_eval_bucket": "1.5"},
    ],
)
def test_invalid_map_partition_config_is_rejected(kwargs):
    with pytest.raises(ValueError):
        SokobanEnvConfig(**kwargs)


def test_integral_float_partition_values_are_accepted():
    config = SokobanEnvConfig(
        map_partition_modulus="4",
        map_partition_eval_bucket=1.0,
    )
    assert config.map_partition_modulus == 4
    assert config.map_partition_eval_bucket == 1


def test_train_and_validation_configs_use_complementary_map_partitions():
    for path in TRAIN_CONFIGS:
        config = _spec(path)["config"]
        assert config["map_partition"] == "train"
        assert config["map_partition_modulus"] == 4
        assert config["map_partition_eval_bucket"] == 0

    validation_seeds = []
    for path in VAL_CONFIGS:
        spec = _spec(path)
        config = spec["config"]
        assert config["map_partition"] == "eval"
        assert config["map_partition_modulus"] == 4
        assert config["map_partition_eval_bucket"] == 0
        assert len(spec["seed_list"]) == spec["n_envs"] == 256
        validation_seeds.append(spec["seed_list"])

    assert validation_seeds[0] == validation_seeds[1]
    assert len(set(validation_seeds[0])) == 256
    assert set(validation_seeds[0]).isdisjoint(range(1, 10001))


def test_validation_seed_manifest_is_unique_and_in_the_eval_partition():
    seeds = _spec(VAL_CONFIGS[0])["seed_list"]
    fingerprints = set()
    for seed in seeds:
        with set_seed(seed):
            fixed, state, _ = generate_room(
                dim=(6, 6), num_steps=20, num_boxes=1, second_player=False
            )
        assert 1 <= len(get_shortest_action_path(fixed, state, MAX_DEPTH=200)) <= 5
        assert _room_matches_partition(fixed, state, "eval", 4, 0)
        fingerprints.add(fixed.tobytes() + state.tobytes())

    assert len(fingerprints) == 256
