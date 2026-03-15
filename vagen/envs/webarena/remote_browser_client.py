"""
Remote browser client for WebArena.

Drop-in replacement for ScriptBrowserEnv that talks to
remote_browser_server.py via HTTP instead of running
Playwright locally. This eliminates network latency
caused by Playwright's many sub-requests going through
SSH tunnels.
"""

import json
import logging
import os
from pathlib import Path
from typing import Any, Optional

import requests

logger = logging.getLogger(__name__)


class RemoteBrowserEnv:
    """
    Drop-in replacement for ScriptBrowserEnv that delegates
    browser operations to a remote server.
    """

    def __init__(
        self,
        server_url: str = "http://localhost:5100",
        observation_type: str = "accessibility_tree",
        current_viewport_only: bool = True,
        viewport_size: Optional[dict] = None,
        timeout: float = 120.0,
        **kwargs,  # absorb extra args for compatibility
    ):
        self.server_url = server_url.rstrip("/")
        self.observation_type = observation_type
        self.current_viewport_only = current_viewport_only
        self.viewport_size = viewport_size or {"width": 1280, "height": 720}
        self.timeout = timeout
        self.session_id: Optional[str] = None
        self.reset_finished = False

        # Mimic ScriptBrowserEnv attributes used by WebArenaEnv
        self.page = _FakePage()

    def setup(self, config_file: Path | None = None) -> None:
        """Not used directly — reset() handles everything."""
        pass

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, str] | None = None,
    ) -> tuple[dict, dict]:
        # Close previous session if any
        if self.session_id:
            self._close_session()

        # Load config from file
        config = {}
        storage_state_dict = None
        if options and "config_file" in options:
            config_path = Path(options["config_file"])
            with open(config_path) as f:
                config = json.load(f)

            # Load storage_state file content if specified
            ss_path = config.get("storage_state")
            if ss_path:
                if not os.path.isabs(ss_path):
                    webarena_root = os.path.dirname(os.path.dirname(config_path))
                    ss_path = os.path.join(webarena_root, ss_path)
                if os.path.exists(ss_path):
                    with open(ss_path) as f:
                        storage_state_dict = json.load(f)
                    # Remove from config since we send it separately
                    del config["storage_state"]

        resp = requests.post(
            f"{self.server_url}/reset",
            json={
                "config": config,
                "observation_type": self.observation_type,
                "current_viewport_only": self.current_viewport_only,
                "viewport_width": self.viewport_size["width"],
                "viewport_height": self.viewport_size["height"],
                "storage_state": storage_state_dict,
            },
            timeout=self.timeout,
        )
        resp.raise_for_status()
        data = resp.json()

        self.session_id = data["session_id"]
        self.reset_finished = True
        self.page = _FakePage(url=data.get("url", ""))

        observation = {"text": data["observation"]}
        info = {
            "page": _FakePage(url=data.get("url", ""), content=""),
            "fail_error": "",
            "observation_metadata": {},
        }

        return observation, info

    def step(self, action: Any) -> tuple[dict, float, bool, bool, dict]:
        if not self.session_id:
            raise RuntimeError("Call reset first before calling step.")

        # Convert Action dict to action string
        action_str = self._action_to_string(action)

        resp = requests.post(
            f"{self.server_url}/step/{self.session_id}",
            json={"action": action_str},
            timeout=self.timeout,
        )
        resp.raise_for_status()
        data = resp.json()

        self.page = _FakePage(url=data["info"].get("url", ""))

        observation = {"text": data["observation"]}
        info = {
            "page": _FakePage(url=data["info"].get("url", ""), content=""),
            "fail_error": data["info"].get("fail_error", ""),
            "observation_metadata": {},
        }

        return observation, data["reward"], data["done"], False, info

    def evaluate(self, config: dict, answer: str = "") -> float:
        """Run WebArena evaluator on the remote server."""
        if not self.session_id:
            raise RuntimeError("No active session.")

        resp = requests.post(
            f"{self.server_url}/eval/{self.session_id}",
            json={"config": config, "answer": answer},
            timeout=self.timeout,
        )
        resp.raise_for_status()
        return resp.json()["score"]

    def close(self) -> None:
        if self.session_id:
            self._close_session()
        self.reset_finished = False

    def _close_session(self):
        try:
            requests.delete(
                f"{self.server_url}/session/{self.session_id}",
                timeout=10,
            )
        except Exception:
            pass
        self.session_id = None

    @staticmethod
    def _action_to_string(action: dict) -> str:
        """Convert an Action dict back to action string for the server."""
        from vagen.envs.webarena.browser_env.actions import ActionTypes

        action_type = action.get("action_type")
        if action_type == ActionTypes.CLICK:
            if action.get("element_id"):
                return f"click [{action['element_id']}]"
        elif action_type == ActionTypes.TYPE:
            if action.get("element_id") and action.get("text"):
                return f"type [{action['element_id']}] [{action['text']}]"
        elif action_type == ActionTypes.HOVER:
            if action.get("element_id"):
                return f"hover [{action['element_id']}]"
        elif action_type == ActionTypes.SCROLL:
            direction = action.get("direction", "down")
            return f"scroll [{direction}]"
        elif action_type == ActionTypes.KEY_PRESS:
            return f"press [{action.get('key_comb', '')}]"
        elif action_type == ActionTypes.GOTO:
            return f"goto [{action.get('url', '')}]"
        elif action_type == ActionTypes.NEW_TAB:
            return "new_tab"
        elif action_type == ActionTypes.GO_BACK:
            return "go_back"
        elif action_type == ActionTypes.GO_FORWARD:
            return "go_forward"
        elif action_type == ActionTypes.TAB_FOCUS:
            return f"tab_focus [{action.get('tab_index', 0)}]"
        elif action_type == ActionTypes.CLOSE_TAB:
            return "close_tab"
        elif action_type == ActionTypes.NONE:
            return "noop"

        # Fallback: try pw_code
        if action.get("pw_code"):
            return action["pw_code"]

        return "noop"


class _FakePage:
    """Minimal page-like object for compatibility with WebArenaEnv."""

    def __init__(self, url: str = "", content: str = ""):
        self.url = url
        self._content = content

    def content(self) -> str:
        return self._content

    def set_default_timeout(self, timeout: int):
        pass

    def set_default_navigation_timeout(self, timeout: int):
        pass