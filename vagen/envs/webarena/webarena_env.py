"""WebArena environment for VAGEN.

One env instance == one Playwright browser context == one task rollout.
The handler hands the env a `BrowserSlot` from its pool; the env creates
its own context + page on that slot. Playwright sync calls are dispatched
through `slot.run()` onto the slot's dedicated thread.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from vagen.envs.gym_base_env import GymBaseEnv
from vagen.envs.webarena.browser_pool import BrowserPool, BrowserSlot
from vagen.envs.webarena.utils.parse import parse_response, map_url_to_local
from vagen.envs.webarena.utils.prompt import WEBARENA_SYS_PROMPT, format_task_prompt

LOGGER = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass
class WebArenaEnvConfig:
    env_name: str = "webarena"
    task_config_file: str = ""      # Absolute path to aggregate JSON (list of tasks)
    max_steps: int = 15
    viewport_width: int = 1280
    viewport_height: int = 720
    observation_type: str = "webrl"
    sleep_after_execution: float = 3.0

    # Timeouts (seconds)
    reset_timeout: float = 120.0
    step_timeout: float = 60.0
    eval_timeout: float = 60.0

    # Storage state / auto_login
    auth_cache_dir: str = ""        # Absolute path to shared .auth cache; handler sets this

    # Reward: binary success/fail — no shaping
    # (no knobs here — kept explicit in step())


# ---------------------------------------------------------------------------
# Env
# ---------------------------------------------------------------------------


class WebArenaEnv(GymBaseEnv):
    """One-task WebArena env. Created by handler with an injected BrowserSlot."""

    def __init__(
        self,
        env_config: Dict[str, Any],
        browser_slot: BrowserSlot,
        browser_pool: BrowserPool,
        auth_locks: Dict[str, asyncio.Lock],
        tasks: List[Dict[str, Any]],
    ):
        super().__init__(env_config)
        self.cfg = WebArenaEnvConfig(**env_config)
        self._slot = browser_slot
        self._pool = browser_pool
        self._auth_locks = auth_locks
        self._tasks = tasks

        self._context = None          # Playwright BrowserContext (lives on slot's thread)
        self._page = None             # Playwright Page
        self._obs_handler = None      # ObservationHandler (one per context)
        self._task: Optional[Dict[str, Any]] = None
        self._task_intent: str = ""
        self._steps_used: int = 0
        self._released = False

    # ------------------------------------------------------------------
    # GymBaseEnv abstract methods
    # ------------------------------------------------------------------

    async def system_prompt(self) -> Dict[str, Any]:
        return {"obs_str": WEBARENA_SYS_PROMPT}

    async def reset(self, seed: int) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        if not self._tasks:
            raise RuntimeError("No tasks loaded; set task_config_file.")

        task = self._tasks[seed % len(self._tasks)]
        self._task = task
        self._task_intent = task.get("intent", "")
        self._steps_used = 0

        # 1) Resolve storage_state path (run auto_login subprocess if needed).
        storage_state_path = await self._ensure_storage_state(task)

        # 2) Create context + goto start_url (on slot's dedicated thread).
        viewport = {"width": self.cfg.viewport_width, "height": self.cfg.viewport_height}
        start_url = task.get("start_url") or ""
        obs_type = self.cfg.observation_type
        sleep_after = self.cfg.sleep_after_execution

        from browser_env.processors import ObservationHandler

        def _setup():
            ctx = self._slot.browser.new_context(
                viewport=viewport,
                storage_state=storage_state_path,
                device_scale_factor=1,
            )
            page = ctx.new_page()
            if start_url:
                for url in start_url.split(" |AND| "):
                    _goto_with_retry(page, url)
                page.bring_to_front()
            page.set_default_timeout(120_000)
            page.set_default_navigation_timeout(120_000)
            page.wait_for_timeout(int(sleep_after * 1000))
            obs_handler = ObservationHandler(
                main_observation_type="text",
                text_observation_type=obs_type,
                image_observation_type="",
                current_viewport_only=True,
                viewport_size=viewport,
                captioning_fn=None,
            )
            obs = obs_handler.get_observation(page)
            return ctx, page, obs_handler, obs

        self._context, self._page, self._obs_handler, obs = await self._slot.run(
            _setup, timeout=self.cfg.reset_timeout
        )

        obs_text = obs.get("text", str(obs)) if isinstance(obs, dict) else str(obs)
        obs_str = format_task_prompt(self._task_intent, round_idx=0, observation=obs_text)
        info = {"task_id": task.get("task_id"), "sites": task.get("sites", [])}
        return {"obs_str": obs_str}, info

    async def step(
        self, action_str: str
    ) -> Tuple[Dict[str, Any], float, bool, Dict[str, Any]]:
        if self._context is None:
            raise RuntimeError("reset() must be called before step().")

        self._steps_used += 1
        parsed = parse_response(action_str)
        info: Dict[str, Any] = {
            "format_correct": parsed["format_correct"],
            "answer": parsed["answer"],
            "is_exit": parsed["is_exit"],
        }

        # Format mismatch → no-op turn, no reward, continue until max_steps.
        if not parsed["format_correct"]:
            if self._steps_used >= self.cfg.max_steps:
                return (
                    {"obs_str": "Max steps reached."},
                    0.0,
                    True,
                    {**info, "error": "max_steps", "success": False},
                )
            # Re-return the current observation (agent sees same page again).
            obs = await self._get_obs_safe()
            obs_str = format_task_prompt(self._task_intent, self._steps_used, obs)
            return (
                {"obs_str": obs_str},
                0.0,
                False,
                {**info, "error": "format_invalid"},
            )

        # URL-map the agent's action text (e.g. http://reddit.com -> http://localhost:9999)
        from browser_env import env_config as _env_cfg
        url_mappings = _env_cfg.URL_MAPPINGS
        mapped_answer = map_url_to_local(parsed["answer"], url_mappings)

        # Exit / terminal action
        if parsed["is_exit"]:
            try:
                score = await self._evaluate(exit_message=parsed["exit_message"])
            except Exception as e:
                LOGGER.warning(f"evaluator failed: {e}")
                info["eval_error"] = str(e)
                score = 0.0
            reward = 1.0 if score > 0 else 0.0
            info["eval_score"] = score
            info["success"] = reward > 0
            return (
                {"obs_str": f"Agent exited: {parsed['exit_message']}"},
                reward,
                True,
                info,
            )

        # Regular action: parse → execute → get new obs
        from browser_env.actions import (
            ActionParsingError,
            create_webrl_id_based_action,
            execute_action_webrl,
        )

        try:
            action = create_webrl_id_based_action(mapped_answer)
        except ActionParsingError as e:
            info["error"] = f"action_parse: {e}"
            obs = await self._get_obs_safe()
            obs_str = format_task_prompt(self._task_intent, self._steps_used, obs)
            done = self._steps_used >= self.cfg.max_steps
            if done:
                info["error"] = "max_steps"
                info["success"] = False
            return {"obs_str": obs_str}, 0.0, done, info

        action["raw_prediction"] = parsed["answer"]

        def _do_step():
            self._page = execute_action_webrl(
                action,
                self._page,
                self._context,
                self._obs_handler.action_processor,
                self.cfg.sleep_after_execution,
            )
            return self._obs_handler.get_observation(self._page)

        try:
            obs = await self._slot.run(_do_step, timeout=self.cfg.step_timeout)
        except asyncio.TimeoutError:
            info["error"] = f"step timed out after {self.cfg.step_timeout}s"
            info["success"] = False
            return {"obs_str": "Step timed out."}, 0.0, True, info
        except Exception as e:
            info["error"] = f"step_exception: {e}"
            # Try to recover by getting current obs; if that fails, terminate.
            obs = await self._get_obs_safe()
            obs_str = format_task_prompt(self._task_intent, self._steps_used, obs)
            done = self._steps_used >= self.cfg.max_steps
            if done:
                info["success"] = False
            return {"obs_str": obs_str}, 0.0, done, info

        obs_text = obs.get("text", str(obs)) if isinstance(obs, dict) else str(obs)
        obs_str = format_task_prompt(self._task_intent, self._steps_used, obs_text)
        done = self._steps_used >= self.cfg.max_steps
        if done:
            info["error"] = "max_steps"
            info["success"] = False
        return {"obs_str": obs_str}, 0.0, done, info

    async def close(self) -> None:
        if self._released:
            return
        if self._context is not None:
            try:
                ctx = self._context

                def _close_ctx():
                    try:
                        ctx.close()
                    except Exception:
                        pass

                await self._slot.run(_close_ctx, timeout=30.0)
            except Exception as e:
                LOGGER.warning(f"context close failed: {e}")
            finally:
                self._context = None
                self._page = None
                self._obs_handler = None
        self._pool.release_slot(self._slot)
        self._released = True

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    async def _get_obs_safe(self) -> str:
        """Best-effort fetch of current observation text; returns '' on error."""
        if self._obs_handler is None or self._page is None:
            return ""

        def _get():
            try:
                obs = self._obs_handler.get_observation(self._page)
                return obs.get("text", "") if isinstance(obs, dict) else str(obs)
            except Exception:
                return ""

        try:
            return await self._slot.run(_get, timeout=self.cfg.step_timeout)
        except Exception:
            return ""

    async def _evaluate(self, exit_message: Optional[str]) -> float:
        """Run the WebArena evaluator on the current page. Returns 0.0/1.0."""
        from evaluation_harness.evaluators import evaluator_router

        config_obj = self._task
        eval_types = config_obj.get("eval", {}).get("eval_types", [])
        if not eval_types:
            return 0.0

        evaluator = evaluator_router(config_file="", eval_types=eval_types)

        def _run_eval():
            try:
                return float(evaluator(
                    trajectory=[],
                    config_file="",
                    page=self._page,
                    exit_message=exit_message or "",
                    config_obj=config_obj,
                ))
            except Exception as e:
                LOGGER.warning(f"evaluator exception: {e}")
                return 0.0

        return await self._slot.run(_run_eval, timeout=self.cfg.eval_timeout)

    async def _ensure_storage_state(self, task: Dict[str, Any]) -> Optional[str]:
        """Materialize the storage_state cookie file for this task.

        Looks up the required cookie file (by basename) in `auth_cache_dir`.
        If missing, runs auto_login.py as a subprocess to generate it.
        Uses per-file asyncio locks so concurrent sessions don't race.
        """
        storage_state = task.get("storage_state") or ""
        if not storage_state:
            return None

        cookie_name = os.path.basename(storage_state)
        cache_dir = self.cfg.auth_cache_dir
        if not cache_dir:
            raise RuntimeError("auth_cache_dir not set in env_config")
        os.makedirs(cache_dir, exist_ok=True)
        target = os.path.join(cache_dir, cookie_name)

        if os.path.exists(target):
            return target

        lock = self._auth_locks.setdefault(cookie_name, asyncio.Lock())
        async with lock:
            # Re-check after acquiring
            if os.path.exists(target):
                return target
            from browser_env.auto_login import get_site_comb_from_filepath
            comb = get_site_comb_from_filepath(cookie_name)
            auto_login_path = Path(__file__).parent / "browser_env" / "auto_login.py"

            def _run_subproc():
                return subprocess.run(
                    [
                        sys.executable,
                        str(auto_login_path),
                        "--auth_folder", cache_dir,
                        "--site_list", *comb,
                    ],
                    capture_output=True,
                    text=True,
                    env=os.environ.copy(),
                    timeout=180,
                )

            result = await asyncio.to_thread(_run_subproc)
            if result.returncode != 0 or not os.path.exists(target):
                raise RuntimeError(
                    f"auto_login failed for {cookie_name}: "
                    f"rc={result.returncode}, stdout={result.stdout[-400:]}, stderr={result.stderr[-400:]}"
                )
            return target


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------


def _goto_with_retry(page, url: str, max_attempts: int = 3) -> None:
    """Retry page.goto with progressively looser wait conditions."""
    for attempt in range(max_attempts):
        try:
            if attempt == 0:
                page.goto(url, timeout=60_000, wait_until="load")
            elif attempt == 1:
                page.goto(url, timeout=60_000, wait_until="domcontentloaded")
            else:
                page.goto(url, timeout=60_000, wait_until="commit")
            page.wait_for_load_state("networkidle", timeout=30_000)
            return
        except Exception:
            time.sleep(1)
    # Final fallback
    page.goto(url, timeout=60_000)
    time.sleep(3)


def load_tasks(task_config_file: str) -> List[Dict[str, Any]]:
    """Load aggregate JSON → list of task dicts."""
    with open(task_config_file) as f:
        tasks = json.load(f)
    if not isinstance(tasks, list):
        raise ValueError(f"{task_config_file} must be a JSON array")
    return tasks
