"""`docs/issues.md` tells a user which prompt format to set. Pin it to the dataclasses.

The table there is load-bearing: on Qwen3-VL, Qwen3.5 and GLM, `<think>` is a single
reserved control token the model will not emit as text, so sokoban's `wm` format -- which
requires it -- scores exactly zero while every other metric looks healthy. Measured on
Qwen3-VL-4B: `wm` gave format 0.000 / score 0.000 against `free_wm`'s 0.969 / 0.602.

Two environments *default* to `wm`. A reader who trusts a stale table there loses a run and
has nothing in the logs pointing at the cause, so the table has to track the code rather
than the other way round.
"""

from __future__ import annotations

import os
import re

import pytest

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_DOC = os.path.join(_ROOT, "docs/issues.md")

#: env -> (config dataclass module path, attribute holding the default)
ENVS = {
    "sokoban": "vagen.envs.sokoban.sokoban_env",
    "frozenlake": "vagen.envs.frozenlake.frozenlake_env",
    "primitive_skill": "vagen.envs.primitive_skill.primitive_skill_env",
    "navigation": "vagen.envs.navigation.navigation_env",
}


def _documented_table():
    """The `| env | formats | default |` table in section 4, as {env: (formats, default)}."""
    rows = {}
    for line in open(_DOC, encoding="utf-8"):
        m = re.match(r"^\|\s*([a-z_]+)\s*\|\s*(.+?)\s*\|\s*`([a-z_]+)`\s*\|\s*$", line)
        if m and m.group(1) in ENVS:
            formats = set(re.findall(r"`([a-z_]+)`", m.group(2)))
            rows[m.group(1)] = (formats, m.group(3))
    return rows


def _config_default(module_path: str) -> str:
    import importlib

    mod = importlib.import_module(module_path)
    # The env's config dataclass is the one carrying prompt_format.
    for obj in vars(mod).values():
        if isinstance(obj, type) and hasattr(obj, "__dataclass_fields__"):
            f = obj.__dataclass_fields__.get("prompt_format")
            if f is not None:
                return f.default
    raise AssertionError(f"no dataclass with prompt_format in {module_path}")


def test_the_table_covers_every_environment_that_has_a_prompt_format():
    documented = _documented_table()
    assert documented, "the prompt_format table in docs/issues.md was not found or moved"
    missing = set(ENVS) - set(documented)
    assert not missing, (
        f"{sorted(missing)} take a prompt_format but are absent from the table in "
        f"docs/issues.md, so nothing tells a reader whether their default is reachable "
        f"on a model that reserves <think>")


@pytest.mark.parametrize("env,module_path", sorted(ENVS.items()))
def test_the_documented_default_is_the_dataclass_default(env, module_path):
    documented = _documented_table()
    assert env in documented, env
    assert documented[env][1] == _config_default(module_path), (
        f"docs/issues.md says {env} defaults to {documented[env][1]!r}, the config "
        f"dataclass says {_config_default(module_path)!r}")


@pytest.mark.parametrize("env", sorted(ENVS))
def test_the_documented_formats_are_the_ones_the_env_implements(env):
    """★ Not a superset check. A format listed but unimplemented sends a user to a value
    the env ignores or rejects, which is the failure this whole section exists to prevent
    -- and `free_wm`/`answer` really are sokoban-only, so the table must not level them up
    to every environment."""
    import glob

    documented = _documented_table()[env][0]
    sources = glob.glob(os.path.join(_ROOT, f"vagen/envs/{env}/**/*.py"), recursive=True)
    text = "".join(open(p, encoding="utf-8").read() for p in sources)
    # The formats are dict keys in the env's prompt module and its parser.
    implemented = set(re.findall(r'"(wm|free_wm|free_think|answer|no_think|eval_mode)"\s*:',
                                 text))
    assert documented == implemented, (
        f"docs/issues.md lists {sorted(documented)} for {env}; the code implements "
        f"{sorted(implemented)}")


def test_every_yaml_block_in_the_docs_is_valid_yaml():
    """★ A ```yaml block is something a reader pastes into a config.

    `docs/configuration.md` and the README both shipped blocks that listed two values for
    one key to show the alternatives -- `harness: mine` / `harness: pkg:Class` under one
    `trainer:`. PyYAML silently keeps the last; OmegaConf, which is what actually loads
    these files, raises `ConstructorError: found duplicate key`. So the block was neither
    readable as documentation nor usable as config, and nothing said so.

    Blocks that show alternatives rather than a config now use ```text.
    """
    import glob
    import re

    from omegaconf import OmegaConf

    files = ([os.path.join(_ROOT, "README.md")]
             + sorted(glob.glob(os.path.join(_ROOT, "docs/*.md")))
             + sorted(glob.glob(os.path.join(_ROOT, "vagen/**/README.md"), recursive=True)))
    bad = []
    for path in files:
        for block in re.findall(r"```yaml\n(.*?)```", open(path, encoding="utf-8").read(),
                                re.S):
            # `<env>` / `<path>` placeholders are prose, not values.
            if "<" in block and ">" in block:
                continue
            try:
                OmegaConf.create(block)
            except Exception as exc:
                bad.append(f"{os.path.relpath(path, _ROOT)}: {type(exc).__name__}: "
                           f"{str(exc).splitlines()[0]}")
    assert not bad, "yaml blocks a reader would paste, that do not parse:\n" + "\n".join(bad)
