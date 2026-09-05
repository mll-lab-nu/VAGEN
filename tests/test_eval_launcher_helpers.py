"""`examples/evaluate/common.sh`, which decides where an eval writes its results.

The dump directory carries the model name, and summary.json is per-directory. Two
checkpoints that derive the same name therefore land in one directory and the second
overwrites the first's summary -- reporting one checkpoint's numbers under the other's
name, exit 0, nothing in the output saying so.

Every launcher got this wrong in one of two ways: `basename`, which is the literal string
`huggingface` for every verl checkpoint, or a hardcoded constant. It is shell, so nothing
in the suite could have caught either.
"""

from __future__ import annotations

import os
import subprocess

import pytest

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_COMMON = os.path.join(_ROOT, "examples/evaluate/common.sh")


def _model_name(path: str) -> str:
    #: `set -eo pipefail` as the launchers run it -- `grep -oE` exits 1 on no match, and a
    #: helper that dies on the hub-id case would take the launcher with it.
    out = subprocess.run(
        ["bash", "-c", f'set -eo pipefail; source "{_COMMON}"; vagen_model_name "$1"',
         "_", path],
        capture_output=True, text=True, cwd=_ROOT,
    )
    assert out.returncode == 0, f"vagen_model_name exited {out.returncode}: {out.stderr}"
    return out.stdout.strip()


@pytest.mark.parametrize("path,expected", [
    # What trainer.default_local_dir produces, and what docs/evaluation.md points at.
    ("exps/p/sokoban_default_gae/verl_checkpoints/global_step_200/actor/huggingface",
     "sokoban_default_gae-global_step_200"),
    # ★ The best-validating actor. docs/evaluation.md names this directory too, and it has
    # no step number -- so matching only `global_step_` sent every run's best actor to the
    # same `huggingface` dump directory.
    ("exps/p/sokoban_default_gae/verl_checkpoints/best_actor/actor/huggingface",
     "sokoban_default_gae-best_actor"),
    # verl's own default layout: checkpoints/<project>/<exp>/<step>/..., no `verl_` prefix
    # and one level shallower. Without the `checkpoints` case the run name is lost and two
    # different runs' step 400 collide.
    ("checkpoints/proj/run_y/global_step_400/actor/huggingface", "run_y-global_step_400"),
    ("/abs/checkpoints/proj/run_z/best_actor/actor/huggingface", "run_z-best_actor"),
    ("/a/exps/run_x/verl_checkpoints/global_step_400/actor/huggingface/",
     "run_x-global_step_400"),
    # A hub id is not a path: `/` would open a directory level under rollouts/.
    ("Qwen/Qwen2.5-VL-3B-Instruct", "Qwen_Qwen2.5-VL-3B-Instruct"),
])
def test_the_dump_directory_name_distinguishes_two_checkpoints(path, expected):
    assert _model_name(path) == expected


def test_two_checkpoints_of_one_run_never_share_a_name():
    """The property, stated directly -- the parametrized cases are how it is spelled today."""
    base = "exps/p/my_run/verl_checkpoints"
    names = {_model_name(f"{base}/{c}/actor/huggingface")
             for c in ("global_step_100", "global_step_200", "best_actor")}
    assert len(names) == 3, f"two checkpoints collapsed to one dump directory: {names}"


def test_a_local_directory_keeps_its_own_name(tmp_path):
    d = tmp_path / "my-local-model"
    d.mkdir()
    assert _model_name(str(d)) == "my-local-model"


def test_every_launcher_derives_the_name_rather_than_hardcoding_one():
    """★ The reason common.sh exists. The original sglang launchers hardcoded a constant,
    so every checkpoint scored through them landed in one directory."""
    import glob
    import re

    launchers = sorted(glob.glob(os.path.join(_ROOT, "examples/evaluate/*/*/eval_*.sh")))
    assert launchers, "no launchers found; the glob is wrong, not the repo empty"
    # The ASSIGNMENT, not the substring: every one of these files mentions
    # `vagen_model_name` in the comment explaining why, so a plain `in text` passes against
    # a launcher that has gone back to a hardcoded constant.
    assign = re.compile(r'^\s*MODEL_NAME=.*\$\(vagen_model_name\s', re.M)
    source = re.compile(r'^\s*source\s+.*common\.sh"', re.M)
    for path in launchers:
        text = open(path).read()
        rel = os.path.relpath(path, _ROOT)
        assert assign.search(text), (
            f"{rel} does not set MODEL_NAME from vagen_model_name, so two checkpoints "
            f"scored through it share a dump directory")
        assert source.search(text), f"{rel} does not source the shared helpers"
