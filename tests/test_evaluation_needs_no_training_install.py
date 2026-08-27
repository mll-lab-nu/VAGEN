"""Nothing on the evaluation path may import verl, torch or ray. That is what lets a user
score a checkpoint against a hosted API without a training install. Shared environment
contracts live in ``vagen/envs/_common`` so evaluation never reaches into training code.

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
CLEAN_PACKAGES = ["vagen/rollout", "vagen/harness", "vagen/evaluation", "vagen/envs"]

FORBIDDEN = {"verl", "torch", "ray"}

#: ★ Also forbidden: reaching into VAGEN's own training packages. Pure-looking helpers
#: there can gain a heavyweight dependency later and silently pull it into evaluation.
FORBIDDEN_PACKAGES = {"vagen.training", "vagen.algorithms"}

#: ManiSkill's own simulation code, vendored under primitive_skill. It is torch-native --
#: the simulator returns tensors -- and it is reached only by the primitive_skill
#: environment, which already needs its own requirements file. The rule is about VAGEN's
#: own layering, not about what a third-party simulator is written in.
EXEMPT_PREFIXES = ("vagen/envs/primitive_skill/maniskill/",)


def _imports(path: str):
    """Every module imported by `path`, at any nesting depth, with its full dotted name.

    ast rather than a grep: a `from torch import ...` inside a function body is exactly the
    kind of import that gets added to "avoid the dependency" and then makes the dependency
    mandatory anyway the moment the code path runs.
    """
    with open(path, encoding="utf-8") as f:
        tree = ast.parse(f.read(), filename=path)
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                yield alias.name
        elif isinstance(node, ast.ImportFrom):
            # level > 0 is a relative import, which cannot reach outside the package.
            if node.module and node.level == 0:
                yield node.module


def _offenders(path: str):
    bad = []
    for mod in _imports(path):
        if mod.split(".")[0] in FORBIDDEN:
            bad.append(mod.split(".")[0])
        elif any(mod == pkg or mod.startswith(pkg + ".") for pkg in FORBIDDEN_PACKAGES):
            bad.append(mod)
    return sorted(set(bad))


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
    found = _offenders(os.path.join(_ROOT, rel))
    assert not found, (
        f"{rel} imports {found}. Evaluation is meant to run without a training install, so "
        f"this makes `pip install` for an eval-only user pull in a training dependency -- "
        f"directly, or transitively through a VAGEN training package that is free of one "
        f"only until someone adds it. Shared helpers belong in the owning axis's _common package."
    )
