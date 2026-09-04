"""Sokoban dataset seeds must identify the same map in every worker process."""

import os
import subprocess
import sys

from vagen.envs.sokoban.patch_sokoban_env import _next_retry_seed


def test_retry_seed_is_a_stable_32_bit_sequence():
    values = [7]
    for _ in range(5):
        values.append(_next_retry_seed(values[-1]))

    assert values == [
        7,
        1025555898,
        3923423697,
        2630631676,
        3981355051,
        211918734,
    ]
    assert len(set(values)) == len(values)


def test_retry_seed_does_not_depend_on_python_hash_randomization():
    code = (
        "from vagen.envs.sokoban.patch_sokoban_env import _next_retry_seed; "
        "print(_next_retry_seed(10001))"
    )
    outputs = []
    for hash_seed in ("1", "937"):
        env = {**os.environ, "PYTHONHASHSEED": hash_seed}
        outputs.append(
            subprocess.check_output([sys.executable, "-c", code], env=env, text=True)
            .strip()
            .splitlines()[-1]
        )
    assert outputs == [outputs[0], outputs[0]]
