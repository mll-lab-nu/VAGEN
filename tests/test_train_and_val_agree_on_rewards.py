"""Train and validation must shape reward the same way.

`val_sokoban_vision.yaml` omitted `format_reward`, so validation silently fell back to
`SokobanEnvConfig`'s default of 0.1 a turn while training used 0.02. A solved episode
therefore scored 1.50 in validation and 1.10 in training, and `val-core/.../reward` --
the number you would naturally read as "how well is it doing" -- was 45% formatting.

Neither file is wrong on its own, which is why this has to be a test across the pair.
Structural differences (`n_envs`, `seed`, `max_turns`, split sizes) are the point of
having two files; reward *weights* are not.
"""

from __future__ import annotations

import glob
import os

import pytest
import yaml

#: Keys that define what the agent is being paid for. These must not differ.
REWARD_KEYS = (
    "format_reward", "success_reward", "per_turn_format_reward", "penalty", "state_reward",
)


def _pairs():
    out = []
    for d in sorted(glob.glob("examples/train/*/")):
        tr = sorted(glob.glob(os.path.join(d, "train_*.yaml")))
        va = sorted(glob.glob(os.path.join(d, "val_*.yaml")))
        for t in tr:
            stem = os.path.basename(t)[len("train_"):]
            match = [v for v in va if os.path.basename(v) == f"val_{stem}"]
            if match:
                out.append((t, match[0]))
    return out


def test_there_are_pairs_to_check():
    assert _pairs(), "found no train/val yaml pairs; this test is silently vacuous"


@pytest.mark.parametrize("train_path,val_path", _pairs(), ids=lambda p: os.path.basename(p))
def test_reward_weights_match(train_path, val_path):
    t = (yaml.safe_load(open(train_path))["envs"][0].get("config") or {})
    v = (yaml.safe_load(open(val_path))["envs"][0].get("config") or {})

    differing = {
        k: (t.get(k, "<absent -> env default>"), v.get(k, "<absent -> env default>"))
        for k in REWARD_KEYS
        if (k in t or k in v) and t.get(k) != v.get(k)
    }
    assert not differing, (
        f"{os.path.basename(train_path)} and {os.path.basename(val_path)} disagree on what "
        f"the agent is paid for: {differing}. An absent key is not neutral -- it takes the "
        "env dataclass default, which is how sokoban ended up validating at format_reward "
        "0.1 while training at 0.02."
    )
