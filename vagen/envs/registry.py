# Dynamic environment registry that loads from env_registry.yaml
from __future__ import annotations
import importlib
import logging
import os
from typing import Any, Dict, Type

import yaml

logger = logging.getLogger(__name__)

_ENV_REGISTRY: Dict[str, Type] = {}
_FAILED_ENVS: Dict[str, str] = {}
_LOADED = False


def _load_registry():
    global _ENV_REGISTRY, _FAILED_ENVS, _LOADED
    if _LOADED:
        return

    # Find env_registry.yaml relative to this file
    config_path = os.path.join(
        os.path.dirname(__file__),
        "..",
        "configs",
        "env_registry.yaml"
    )
    config_path = os.path.abspath(config_path)

    if not os.path.exists(config_path):
        raise FileNotFoundError(f"env_registry.yaml not found at {config_path}")

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    env_registry = config.get("env_registry", {})

    for env_name, module_path in env_registry.items():
        try:
            module_name, class_name = module_path.rsplit(".", 1)
            module = importlib.import_module(module_name)
            _ENV_REGISTRY[env_name] = getattr(module, class_name)
        except Exception as e:
            _FAILED_ENVS[env_name] = str(e)
            logger.warning(f"Failed to load env '{env_name}' from '{module_path}': {e}")

    _LOADED = True


def get_env_cls(name: str) -> Type:
    """Resolve environment class from env_registry.yaml."""
    _load_registry()
    if name not in _ENV_REGISTRY:
        raise KeyError(f"Unknown env name: {name}. Available: {list(_ENV_REGISTRY.keys())}")
    return _ENV_REGISTRY[name]


def register_env(name: str, env_cls: Type) -> None:
    """Manually register an environment class."""
    _load_registry()
    _ENV_REGISTRY[name] = env_cls


def list_envs() -> list:
    """List all registered environment names."""
    _load_registry()
    return list(_ENV_REGISTRY.keys())
