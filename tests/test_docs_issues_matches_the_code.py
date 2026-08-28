"""The documented response protocols must track environment capabilities and defaults."""

from __future__ import annotations

import os
import re

import pytest

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_DOC = os.path.join(_ROOT, "docs/issues.md")

#: env -> (config dataclass module path, attribute holding the default)
ENVS = {
    "sokoban": ("vagen.envs.sokoban.sokoban_env", "vagen.envs.sokoban.utils.utils"),
    "frozenlake": ("vagen.envs.frozenlake.frozenlake_env", "vagen.envs.frozenlake.utils.utils"),
    "primitive_skill": (
        "vagen.envs.primitive_skill.primitive_skill_env",
        "vagen.envs.primitive_skill.utils.parse",
    ),
    "navigation": ("vagen.envs.navigation.navigation_env", "vagen.envs.navigation.utils.parse"),
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


@pytest.mark.parametrize("env,modules", sorted(ENVS.items()))
def test_the_documented_default_is_the_dataclass_default(env, modules):
    module_path, _ = modules
    documented = _documented_table()
    assert env in documented, env
    assert documented[env][1] == _config_default(module_path), (
        f"docs/issues.md says {env} defaults to {documented[env][1]!r}, the config "
        f"dataclass says {_config_default(module_path)!r}")


@pytest.mark.parametrize("env", sorted(ENVS))
def test_the_documented_formats_are_the_ones_the_env_implements(env):
    """Compatibility aliases are intentionally excluded from the public format set."""
    import importlib

    _, parser_module = ENVS[env]
    documented = _documented_table()[env][0]
    implemented = set(importlib.import_module(parser_module).PROMPT_FORMATS)
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
