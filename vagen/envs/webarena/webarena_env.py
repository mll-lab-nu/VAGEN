import asyncio
import concurrent.futures
import re
import json
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from vagen.envs.gym_image_env import GymImageEnv


# ------------------------------
# Prompt helpers
# ------------------------------
def _system_prompt() -> str:
    return (
        "You are an autonomous web browsing agent. You interact with web pages "
        "through an accessibility tree representation.\n\n"
        "## Available Actions\n"
        "- click [id]: Click on an element.\n"
        "- type [id] [content]: Type content into an element.\n"
        "- hover [id]: Hover over an element.\n"
        "- press [key_comb]: Press a key combination (e.g., Enter, Ctrl+a).\n"
        "- scroll [direction=down|up]: Scroll the page.\n"
        "- new_tab: Open a new tab.\n"
        "- tab_focus [tab_index]: Focus on a specific tab.\n"
        "- close_tab: Close the current tab.\n"
        "- goto [url]: Navigate to a URL.\n"
        "- go_back: Go back to the previous page.\n"
        "- go_forward: Go forward to the next page.\n"
        "- stop [answer]: Stop and submit your answer.\n\n"
        "## Action Format\n"
        "Wrap your action in <action>...</action> tags. For example:\n"
        "  <action>click [42]</action>\n"
        "  <action>type [15] [Hello World]</action>\n"
        "  <action>stop [answer text]</action>\n\n"
        "## Rules\n"
        "- Issue ONE action per step.\n"
        "- Use the element [id] from the accessibility tree.\n"
        "- When the task is complete, use the stop action.\n"
    )


def _parse_action(action_str: str) -> Dict[str, Any]:
    """Parse LLM output to extract the browser action from <action>...</action> tags."""
    out = {
        "format_correct": False,
        "action": None,
        "raw": action_str,
    }

    m = re.search(r"<action>(.*?)</action>", action_str, re.DOTALL | re.IGNORECASE)
    if m:
        action = m.group(1).strip()
        if action:
            out["format_correct"] = True
            out["action"] = action
    return out


# ------------------------------
# Config
# ------------------------------
@dataclass
class WebArenaEnvConfig:
    """Configuration for WebArena environment."""
    config_files: Optional[List[str]] = None   # List of task config file paths
    config_dir: str = "config_files"           # Directory to auto-discover configs from
    headless: bool = True
    observation_type: str = "accessibility_tree"
    current_viewport_only: bool = True
    viewport_width: int = 1280
    viewport_height: int = 720
    max_steps: int = 30

    # Filtering
    skip_sites: Optional[List[str]] = None  # Skip tasks involving these sites (e.g. ["map"])

    # Rewards
    format_reward: float = 0.01
    success_reward: float = 1.0


# ------------------------------
# Environment
# ------------------------------
class WebArenaEnv(GymImageEnv):
    """
    Minimal WebArena environment wrapper for VAGEN.
    Wraps browser_env.ScriptBrowserEnv with the async GymImageEnv interface.
    """

    def __init__(self, env_config: Dict[str, Any]):
        super().__init__(env_config)
        self.config = WebArenaEnvConfig(**env_config)

        # Resolve config file list
        if self.config.config_files is not None:
            self._config_files = list(self.config.config_files)
        elif os.path.isdir(self.config.config_dir):
            self._config_files = sorted(
                os.path.join(self.config.config_dir, f)
                for f in os.listdir(self.config.config_dir)
                if f.endswith(".json")
            )
        else:
            self._config_files = []

        # Filter out tasks involving skip_sites
        if self.config.skip_sites:
            skip = {s.lower() for s in self.config.skip_sites}
            filtered = []
            for cf in self._config_files:
                with open(cf) as f:
                    task = json.load(f)
                if not isinstance(task, dict):
                    continue
                sites = [s.lower() for s in task.get("sites", [])]
                if not any(s in skip for s in sites):
                    filtered.append(cf)
            self._config_files = filtered

        if not self._config_files:
            raise ValueError(
                "No config files found. Provide 'config_files' or a valid 'config_dir'."
            )

        # Browser env created lazily on first reset()
        self.browser_env = None

        # Dedicated single-thread executor for Playwright sync API.
        # Using executor.submit() instead of asyncio.to_thread() avoids
        # Python 3.12's contextvars propagation which leaks the asyncio
        # running loop into the thread, causing Playwright to raise
        # "Playwright Sync API inside the asyncio loop".
        self._executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)

        # Episode state
        self._steps_used: int = 0
        self.total_reward: float = 0.0
        self._task_intent: str = ""
        self._current_config_file: str = ""

    async def _run_sync(self, fn, *args, **kwargs):
        """Run a sync function in a dedicated thread WITHOUT asyncio context propagation.

        asyncio.to_thread() copies contextvars (including the running loop) to the
        worker thread, causing Playwright sync API to fail with 'Sync API inside
        asyncio loop'. executor.submit() + wrap_future bypasses this.
        """
        loop = asyncio.get_running_loop()
        future = self._executor.submit(fn, *args, **kwargs)
        return await asyncio.wrap_future(future, loop=loop)

    def _ensure_browser_env(self):
        """Lazily create ScriptBrowserEnv (lightweight — browser launches on reset)."""
        if self.browser_env is None:
            from vagen.envs.webarena.browser_env import ScriptBrowserEnv
            self.browser_env = ScriptBrowserEnv(
                headless=self.config.headless,
                observation_type=self.config.observation_type,
                current_viewport_only=self.config.current_viewport_only,
                viewport_size={
                    "width": self.config.viewport_width,
                    "height": self.config.viewport_height,
                },
            )

    # ------------------------------------------------------------------
    # GymImageEnv abstract methods
    # ------------------------------------------------------------------
    async def close(self) -> None:
        if self.browser_env is not None:
            await self._run_sync(self.browser_env.close)
            self.browser_env = None
        self._executor.shutdown(wait=False)

    async def system_prompt(self) -> Dict[str, Any]:
        return {"obs_str": _system_prompt()}

    async def reset(self, seed: int) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        self._ensure_browser_env()

        idx = seed % len(self._config_files)
        config_file = self._config_files[idx]

        obs, info = await self._run_sync(
            self.browser_env.reset, options={"config_file": config_file}
        )

        self._steps_used = 0
        self.total_reward = 0.0
        self._current_config_file = config_file

        # Load task intent from config
        with open(config_file) as f:
            task_config = json.load(f)
        self._task_intent = task_config.get("intent", "")

        obs_text = obs.get("text", str(obs)) if isinstance(obs, dict) else str(obs)
        obs_str = (
            f"Task: {self._task_intent}\n\n"
            f"Current page accessibility tree:\n{obs_text}\n"
        )
        return {"obs_str": obs_str}, info

    async def _evaluate_success(self, info: Dict[str, Any]) -> float:
        """Run WebArena evaluator to check task success."""
        from vagen.envs.webarena.evaluation_harness.evaluators import evaluator_router

        config_file = self._current_config_file
        answer = info.get("answer", "")
        browser_env = self.browser_env

        def _eval():
            evaluator = evaluator_router(config_file)
            last_action = {"answer": answer}
            trajectory = [last_action]

            page = browser_env.page
            client = page.context.new_cdp_session(page)
            try:
                score = evaluator(
                    trajectory=trajectory,
                    config_file=config_file,
                    page=page,
                    client=client,
                )
            finally:
                client.detach()
            return score

        return await self._run_sync(_eval)

    async def step(
        self, action_str: str
    ) -> Tuple[Dict[str, Any], float, bool, Dict[str, Any]]:
        self._steps_used += 1
        reward = 0.0
        done = False
        info: Dict[str, Any] = {}

        parsed = _parse_action(action_str)
        info.update(parsed)

        if parsed["format_correct"]:
            reward += self.config.format_reward

            action_text = parsed["action"]

            # Handle stop action — agent wants to submit answer and end
            if action_text.startswith("stop"):
                done = True
                answer = action_text[len("stop"):].strip().strip("[]")
                info["answer"] = answer
                obs = {"text": f"Agent submitted answer: {answer}"}
            else:
                from vagen.envs.webarena.browser_env import create_id_based_action

                try:
                    action = create_id_based_action(action_text)
                    obs, _, terminated, _, step_info = await self._run_sync(
                        self.browser_env.step, action
                    )
                    info.update(step_info)
                    done = terminated
                except Exception as e:
                    info["error"] = str(e)
                    obs = {"text": f"Action failed: {e}"}
        else:
            info["error"] = "format_invalid"
            obs = {"text": "Invalid action format. Use <action>...</action> tags."}

        # Max-step termination
        if not done and self._steps_used >= self.config.max_steps:
            done = True
            info.setdefault("error", "max_steps_reached")

        # Success detection via WebArena evaluator
        success = False
        if done and self._current_config_file:
            try:
                score = await self._evaluate_success(info)
                success = score > 0
                info["eval_score"] = score
            except Exception as e:
                info["eval_error"] = str(e)
        if success:
            reward += self.config.success_reward

        info["metrics"] = {
            "turn_metrics": {
                "action_is_valid": parsed["format_correct"],
                "steps_used": self._steps_used,
            },
            "traj_metrics": {
                "success": success,
            },
        }
        info["success"] = success
        self.total_reward += reward

        obs_text = obs.get("text", str(obs)) if isinstance(obs, dict) else str(obs)
        obs_str = f"Current page accessibility tree:\n{obs_text}\n"

        return {"obs_str": obs_str}, reward, done, info


# ------------------------------
# Local async test
# ------------------------------
if __name__ == "__main__":
    import fire

    async def main_async(
        config_file: str = "config_files/0.json",
        headless: bool = False,
        max_steps: int = 30,
    ):
        env = WebArenaEnv({
            "config_files": [config_file],
            "headless": headless,
            "max_steps": max_steps,
        })

        print("System Prompt:")
        sys_prompt = await env.system_prompt()
        print(sys_prompt["obs_str"])
        print("=" * 50)

        obs, info = await env.reset(seed=0)
        print("Initial Observation:")
        print(obs["obs_str"][:2000])  # truncate for readability

        for step in range(max_steps):
            print(f"\nStep {step + 1}:")
            try:
                action_input = input("Enter action (e.g. 'click [42]') or 'quit': ")
            except EOFError:
                break

            if action_input.lower() == "quit":
                break

            # Wrap in tags if not already
            if "<action>" not in action_input:
                action_input = f"<action>{action_input}</action>"

            obs, reward, done, info = await env.step(action_input)
            print(f"Reward: {reward}, Done: {done}")
            print(f"Observation (truncated):\n{obs['obs_str'][:2000]}")

            if done:
                print(f"Episode ended. Success: {info.get('success', False)}")
                print(f"Eval score: {info.get('eval_score', 'N/A')}")
                if "eval_error" in info:
                    print(f"Eval error: {info['eval_error']}")
                break

        print(f"\nTotal reward: {env.total_reward}")
        await env.close()

    def main(**kwargs):
        asyncio.run(main_async(**kwargs))

    fire.Fire(main)
