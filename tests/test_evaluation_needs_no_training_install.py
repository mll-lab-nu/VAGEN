"""The dependency rule VAGEN_ARCH.md states, as a gate rather than a claim.

Nothing on the evaluation path may import verl, torch or ray. That is what lets a user
score a checkpoint against a hosted API without a training install, and it is the reason
`vagen/core/env_adapter.py` exists at all -- `GymEnvAdapter` was moved out of
`agent_loop/gym_loop.py`, which does import verl, purely so evaluation could reach it.

The rule was written down and enforced by review. A single `import torch` at the top of any
of these modules breaks it, costs a multi-gigabyte install for an eval run, and nothing
would say so: torch is present in every environment this repo is developed in.
"""

from __future__ import annotations

import ast
import os

import pytest

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

#: Packages that must import none of FORBIDDEN.
CLEAN_PACKAGES = ["vagen/core", "vagen/harness", "vagen/evaluate", "vagen/envs_remote",
                  "vagen/envs"]

FORBIDDEN = {"verl", "torch", "ray"}

#: ManiSkill's own simulation code, vendored under primitive_skill. It is torch-native --
#: the simulator returns tensors -- and it is reached only by the primitive_skill
#: environment, which already needs its own requirements file. The rule is about VAGEN's
#: own layering, not about what a third-party simulator is written in.
EXEMPT_PREFIXES = ("vagen/envs/primitive_skill/maniskill/",)


def _module_roots(path: str):
    """Every module imported by `path`, at any nesting depth, as its top-level name.

    ast rather than a grep: a `from torch import ...` inside a function body is exactly the
    kind of import that gets added to "avoid the dependency" and then makes the dependency
    mandatory anyway the moment the code path runs.
    """
    with open(path, encoding="utf-8") as f:
        tree = ast.parse(f.read(), filename=path)
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                yield alias.name.split(".")[0]
        elif isinstance(node, ast.ImportFrom):
            # level > 0 is a relative import, which cannot reach outside the package.
            if node.module and node.level == 0:
                yield node.module.split(".")[0]


def _python_files():
    out = []
    for pkg in CLEAN_PACKAGES:
        for dirpath, dirnames, filenames in os.walk(os.path.join(_ROOT, pkg)):
            dirnames[:] = [d for d in dirnames if d != "__pycache__"]
            for name in filenames:
                if not name.endswith(".py"):
                    continue
                rel = os.path.relpath(os.path.join(dirpath, name), _ROOT)
                if rel.startswith(EXEMPT_PREFIXES):
                    continue
                out.append(rel)
    return sorted(out)


FILES = _python_files()
assert FILES, "no files collected; CLEAN_PACKAGES is wrong, not the repo empty"


@pytest.mark.parametrize("rel", FILES)
def test_the_evaluation_path_imports_no_training_dependency(rel):
    found = sorted(set(_module_roots(os.path.join(_ROOT, rel))) & FORBIDDEN)
    assert not found, (
        f"{rel} imports {found}. Evaluation is meant to run without a training install, "
        f"so this makes `pip install` for an eval-only user pull in {found[0]}. If the "
        f"code genuinely belongs on the training side, it belongs under vagen/agent_loop "
        f"or vagen/trainer."
    )
