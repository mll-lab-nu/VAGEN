"""WebArena environment for VAGEN.

A multi-turn web navigation environment where the agent interacts with
realistic websites (shopping, forums, CMS, GitLab, map) via browser actions
to complete tasks.

Requires:
    - playwright (pip install playwright && playwright install chromium)
    - WebArena Docker containers running (see README.md)
    - Task config JSON from WebArena (e.g. test.raw.json)
"""

import json
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from vagen.envs.gym_image_env import GymImageEnv

from .action_parser import execute_action, parse_action
from .evaluation import evaluate_task
from .observation import build_observation
from .prompt import get_system_prompt

logger = logging.getLogger(__name__)


@dataclass
class WebArenaEnvConfig:
    """Configuration for the WebArena environment."""

    # Task data
    task_file: str = ""                     # Path to WebArena task JSON
    task_ids: Optional[List[int]] = None    # Subset of task IDs (None = all)
    tasks: Optional[List[Dict]] = None      # Directly provide task list (overrides task_file)

    # WebArena server URLs (used to substitute __SHOP__, etc. in task configs)
    shopping_url: str = "http://localhost:7770"
    shopping_admin_url: str = "http://localhost:7780"
    reddit_url: str = "http://localhost:9999"
    gitlab_url: str = "http://localhost:8023"
    wikipedia_url: str = "http://localhost:8888"
    map_url: str = "http://localhost:443"
    homepage_url: str = "http://localhost:4399"

    # Observation
    render_mode: str = "vision"             # "text" or "vision"
    viewport_width: int = 1280
    viewport_height: int = 720
    image_placeholder: str = "<image>"

    # Episode
    max_steps: int = 15

    # Rewards
    format_reward: float = 0.01
    success_reward: float = 1.0
    step_penalty: float = 0.0

    # Browser
    headless: bool = True
    slow_mo: int = 0                        # Slow down Playwright actions (ms), useful for debugging


class WebArenaEnv(GymImageEnv):
    """
    WebArena environment for VAGEN.

    The agent observes web pages (screenshots or accessibility trees) and
    issues browser actions (click, type, scroll, goto, go_back, stop) to
    complete realistic web tasks.
    """

    def __init__(self, env_config: Dict[str, Any]):
        super().__init__(env_config)
        self.config = WebArenaEnvConfig(**env_config)
        self._tasks: List[Dict[str, Any]] = []
        self._load_tasks()

        # Browser state (lazy init)
        self._playwright = None
        self._browser = None
        self._context = None
        self._page = None

        # Episode state
        self._task: Optional[Dict[str, Any]] = None
        self._steps_used: int = 0
        self._last_answer: Optional[str] = None
        self.total_reward: float = 0.0

    def _load_tasks(self) -> None:
        """Load tasks from file or direct config."""
        if self.config.tasks:
            self._tasks = list(self.config.tasks)
        elif self.config.task_file:
            with open(self.config.task_file, "r") as f:
                all_tasks = json.load(f)
            self._tasks = all_tasks
        else:
            raise ValueError(
                "WebArenaEnv requires either 'task_file' (path to JSON) or 'tasks' (list of dicts) in env_config."
            )

        # Filter by task_ids if specified
        if self.config.task_ids is not None:
            id_set = set(self.config.task_ids)
            self._tasks = [t for t in self._tasks if t.get("task_id") in id_set]

        if not self._tasks:
            raise ValueError("No tasks loaded. Check task_file/tasks and task_ids.")

        # Substitute server URL placeholders in task configs
        self._substitute_urls()

    def _substitute_urls(self) -> None:
        """Replace URL placeholders like __SHOPPING__ with actual server URLs."""
        url_map = {
            "__SHOPPING__": self.config.shopping_url,
            "__SHOPPING_ADMIN__": self.config.shopping_admin_url,
            "__REDDIT__": self.config.reddit_url,
            "__GITLAB__": self.config.gitlab_url,
            "__WIKIPEDIA__": self.config.wikipedia_url,
            "__MAP__": self.config.map_url,
            "__HOMEPAGE__": self.config.homepage_url,
        }

        def _replace(obj):
            if isinstance(obj, str):
                for placeholder, url in url_map.items():
                    obj = obj.replace(placeholder, url)
                return obj
            elif isinstance(obj, dict):
                return {k: _replace(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [_replace(item) for item in obj]
            return obj

        self._tasks = _replace(self._tasks)

    async def _ensure_browser(self) -> None:
        """Lazy-initialize Playwright and browser."""
        if self._browser is None:
            from playwright.async_api import async_playwright
            self._playwright = await async_playwright().start()
            self._browser = await self._playwright.chromium.launch(
                headless=self.config.headless,
                slow_mo=self.config.slow_mo,
            )

    async def close(self) -> None:
        """Clean up browser resources."""
        if self._page:
            try:
                await self._page.close()
            except Exception:
                pass
            self._page = None
        if self._context:
            try:
                await self._context.close()
            except Exception:
                pass
            self._context = None
        if self._browser:
            try:
                await self._browser.close()
            except Exception:
                pass
            self._browser = None
        if self._playwright:
            try:
                await self._playwright.stop()
            except Exception:
                pass
            self._playwright = None

    async def system_prompt(self) -> Dict[str, Any]:
        """Return system-level instructions for web navigation."""
        return {"obs_str": get_system_prompt(self.config.render_mode)}

    async def reset(self, seed: int) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Reset: pick a task by seed, open the starting URL."""
        await self._ensure_browser()

        # Deterministic task selection
        idx = seed % len(self._tasks)
        self._task = self._tasks[idx]

        # Reset episode state
        self._steps_used = 0
        self._last_answer = None
        self.total_reward = 0.0

        # Create fresh browser context and page
        if self._page:
            await self._page.close()
        if self._context:
            await self._context.close()

        self._context = await self._browser.new_context(
            viewport={
                "width": self.config.viewport_width,
                "height": self.config.viewport_height,
            },
        )
        self._page = await self._context.new_page()

        # Navigate to the task's starting URL
        start_url = self._task.get("start_url", self.config.homepage_url)
        try:
            await self._page.goto(start_url, timeout=15000)
            await self._page.wait_for_load_state("domcontentloaded", timeout=10000)
        except Exception as e:
            logger.warning(f"Failed to load start_url {start_url}: {e}")

        # Build initial observation
        task_intent = self._task.get("intent", self._task.get("task", ""))
        remaining = self.config.max_steps - self._steps_used
        obs = await build_observation(
            self._page,
            task_intent,
            remaining,
            self.config.render_mode,
            self.config.image_placeholder,
        )

        info = {"task_id": self._task.get("task_id", idx)}
        return obs, info

    async def step(self, action_str: str) -> Tuple[Dict[str, Any], float, bool, Dict[str, Any]]:
        """Execute one browser action and return the new observation."""
        self._steps_used += 1
        reward = 0.0
        done = False
        info: Dict[str, Any] = {}

        # Parse action from agent output
        parsed = parse_action(action_str)
        info["format_correct"] = parsed["format_correct"]
        info["action_type"] = parsed.get("action_type")

        # Format reward
        if parsed["format_correct"]:
            reward += self.config.format_reward

        # Execute browser action
        action_error = None
        if parsed["format_correct"] and parsed["action_type"] != "stop":
            action_error = await execute_action(self._page, parsed)
            if action_error:
                info["action_error"] = action_error
            else:
                # Wait for page to stabilize after action
                try:
                    await self._page.wait_for_load_state("domcontentloaded", timeout=5000)
                except Exception:
                    pass

        # Handle stop action
        if parsed.get("action_type") == "stop":
            self._last_answer = parsed.get("answer")
            success = await self._evaluate_task()
            reward += self.config.success_reward if success else 0.0
            info["success"] = success
            done = True

        # Step budget exhausted
        if not done and self._steps_used >= self.config.max_steps:
            success = await self._evaluate_task()
            info["success"] = success
            info["error"] = "max_steps_reached"
            done = True

        # Step penalty
        reward += self.config.step_penalty
        self.total_reward += reward

        # Metrics
        info["metrics"] = {
            "turn_metrics": {
                "format_correct": parsed["format_correct"],
                "action_type": parsed.get("action_type"),
                "steps_used": self._steps_used,
                "action_error": action_error,
            },
            "traj_metrics": {
                "success": bool(info.get("success", False)),
            },
        }

        # Build observation
        task_intent = self._task.get("intent", self._task.get("task", ""))
        remaining = max(0, self.config.max_steps - self._steps_used)
        obs = await build_observation(
            self._page,
            task_intent,
            remaining,
            self.config.render_mode,
            self.config.image_placeholder,
        )

        return obs, reward, done, info

    async def _evaluate_task(self) -> bool:
        """Evaluate whether the current task has been completed successfully."""
        try:
            return await evaluate_task(self._page, self._task, self._last_answer)
        except Exception as e:
            logger.warning(f"Evaluation failed: {e}")
            return False


# ---------------------------------------------------------------------------
# Local async test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import asyncio

    TOY_TASKS = [
        {
            "task_id": 0,
            "intent": "Find the price of the cheapest item in the 'Electronics' category.",
            "start_url": "http://localhost:7770",
            "eval": {
                "eval_types": ["string_match"],
                "configs": [{"reference_answers": ["$9.99"]}],
            },
        }
    ]

    async def main():
        env = WebArenaEnv({
            "tasks": TOY_TASKS,
            "render_mode": "text",
            "max_steps": 5,
        })
        print("System prompt:")
        print((await env.system_prompt())["obs_str"])
        print()

        obs, info = await env.reset(seed=0)
        print("Initial observation:")
        print(obs["obs_str"])
        print()

        while True:
            action = input("\nAction (e.g. <action>click[0]</action>): ")
            obs, reward, done, info = await env.step(action)
            print(f"reward={reward} done={done}")
            print(obs["obs_str"][:500])
            if done:
                print(f"TOTAL: {env.total_reward} SUCCESS: {info.get('success', False)}")
                break

        await env.close()

    asyncio.run(main())
