# Make browser_env / evaluation_harness / agent / llms importable as
# top-level modules. The upstream VAB-WebArena-Lite code does absolute
# imports (`from browser_env.constants import ...`), so we expose this
# directory on sys.path rather than rewriting every import in the tree.
import os as _os
import sys as _sys
_WA_DIR = _os.path.dirname(_os.path.abspath(__file__))
if _WA_DIR not in _sys.path:
    _sys.path.insert(0, _WA_DIR)

from .webarena_env import WebArenaEnv, WebArenaEnvConfig
from .handler import WebArenaHandler
from .browser_pool import BrowserPool, BrowserSlot

__all__ = [
    "WebArenaEnv",
    "WebArenaEnvConfig",
    "WebArenaHandler",
    "BrowserPool",
    "BrowserSlot",
]
