import asyncio
import json
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import httpx

from vagen.envs.gym_image_env import GymImageEnv


# -----------------------------
# Utils
# -----------------------------
def normalize_answer(s: str) -> str:
    s = s.lower().strip()
    s = re.sub(r"[^\w\s]", "", s)
    s = re.sub(r"\s+", " ", s)
    return s


def exact_match(pred: str, gold: str) -> bool:
    return normalize_answer(pred) == normalize_answer(gold)


def parse_action(text: str) -> Dict[str, Any]:
    """
    Parse Search-R1 style actions.

    Supported:
      <search>query</search>
      <answer>answer</answer>
      JSON: {"action": "search", "query": "..."}
    """
    out = {
        "format_correct": False,
        "action": None,
        "query": None,
        "answer": None,
    }

    text = text.strip()

    # JSON
    if text.startswith("{"):
        try:
            obj = json.loads(text)
            act = obj.get("action", "").lower()
            if act == "search":
                out.update(format_correct=True, action="search", query=obj.get("query"))
                return out
            if act == "final":
                out.update(format_correct=True, action="final", answer=obj.get("answer"))
                return out
        except Exception:
            pass

    # XML-style
    m = re.search(r"<search>(.*?)</search>", text, re.S)
    if m:
        out.update(format_correct=True, action="search", query=m.group(1).strip())
        return out

    m = re.search(r"<answer>(.*?)</answer>", text, re.S)
    if m:
        out.update(format_correct=True, action="final", answer=m.group(1).strip())
        return out

    return out


# -----------------------------
# Config
# -----------------------------
@dataclass
class SearchR1EnvConfig:
    dataset: List[Dict[str, Any]]          # [{"question","answer"}]
    retrieval_server_url: str = "http://127.0.0.1:8000"
    top_k: int = 5

    max_steps: int = 8
    max_searches: int = 4

    format_reward: float = 0.01
    search_penalty: float = -0.01
    correct_reward: float = 1.0
    wrong_reward: float = 0.0


# -----------------------------
# Environment
# -----------------------------
class SearchR1Env(GymImageEnv):
    """
    Search-R1 Environment for VAGEN (Dense Retrieval Version).

    Agent Actions:
      - SEARCH(query)
      - FINAL(answer)
    """

    def __init__(self, env_config: Dict[str, Any]):
        super().__init__(env_config)
        self.cfg = SearchR1EnvConfig(**env_config)

        self.client = httpx.AsyncClient(timeout=30.0)

        self.episode = None
        self.question = ""
        self.gold_answer = ""

        self.steps = 0
        self.searches = 0
        self.evidence: List[Dict[str, Any]] = []
        self.total_reward = 0.0

    # -----------------------------
    # VAGEN required methods
    # -----------------------------
    async def close(self):
        await self.client.aclose()

    async def system_prompt(self) -> Dict[str, Any]:
        return {
            "obs_str": (
                "You are a Search-R1 agent.\n"
                "You may iteratively SEARCH for evidence, then give a FINAL answer.\n\n"
                "Actions:\n"
                "<search>query</search>\n"
                "<answer>answer</answer>\n"
            )
        }

    async def reset(self, seed: int) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        idx = seed % len(self.cfg.dataset)
        self.episode = self.cfg.dataset[idx]

        self.question = self.episode["question"]
        self.gold_answer = self.episode["answer"]

        self.steps = 0
        self.searches = 0
        self.evidence = []
        self.total_reward = 0.0

        return {"obs_str": self._render_obs()}, {}

    async def step(self, action_str: str):
        self.steps += 1
        reward = 0.0
        done = False
        info: Dict[str, Any] = {}

        parsed = parse_action(action_str)
        info.update(parsed)

        if parsed["format_correct"]:
            reward += self.cfg.format_reward

        if parsed["action"] == "search":
            if self.searches >= self.cfg.max_searches:
                info["error"] = "search_budget_exceeded"
            else:
                self.searches += 1
                reward += self.cfg.search_penalty
                await self._do_search(parsed["query"])

        elif parsed["action"] == "final":
            pred = parsed["answer"]
            correct = exact_match(pred, self.gold_answer)
            reward += self.cfg.correct_reward if correct else self.cfg.wrong_reward
            done = True
            info["correct"] = correct
            info["success"] = correct

        if self.steps >= self.cfg.max_steps:
            done = True

        self.total_reward += reward
        obs = {"obs_str": self._render_obs()}

        info["metrics"] = {
            "steps": self.steps,
            "searches": self.searches,
        }

        return obs, reward, done, info

    # -----------------------------
    # Internal
    # -----------------------------
    async def _do_search(self, query: str):
        payload = {
            "query": query,
            "top_k": self.cfg.top_k,
        }
        resp = await self.client.post(
            f"{self.cfg.retrieval_server_url}/retrieve",
            json=payload,
        )
        resp.raise_for_status()
        data = resp.json()
        results = data.get("results", [])

        for r in results:
            doc = r.get("document", {})
            self.evidence.append({
                "id": doc.get("id"),
                "text": doc.get("contents", "")[:300],
                "score": r.get("score", 0.0),
            })

    def _render_obs(self) -> str:
        ev = "\n\n".join(
            f"[{i+1}] (score={e['score']:.3f}) {e['text']}"
            for i, e in enumerate(self.evidence)
        ) or "(no evidence yet)"

        return (
            f"Question:\n{self.question}\n\n"
            f"Evidence:\n{ev}\n\n"
            f"Budgets: steps={self.steps}/{self.cfg.max_steps}, "
            f"searches={self.searches}/{self.cfg.max_searches}\n"
        )

# ------------------------------
# Local async test (optional)
# ------------------------------
if __name__ == "__main__":
    import fire
    import asyncio
    import logging

    logging.basicConfig(
        level=logging.INFO,
        format='[%(levelname)s] %(message)s'
    )

    async def main_async(
        retrieval_server_url: str = "http://127.0.0.1:8000",
        max_steps: int = 6,
        max_searches: int = 3,
        top_k: int = 3,
    ):
        # Minimal toy dataset
        dataset = [
            {
                "question": "Who wrote The Old Man and the Sea?",
                "answer": "Ernest Hemingway",
            }
        ]

        cfg = {
            "dataset": dataset,
            "retrieval_server_url": retrieval_server_url,
            "max_steps": max_steps,
            "max_searches": max_searches,
            "top_k": top_k,
        }

        env = SearchR1Env(cfg)

        # System Prompt
        print("=" * 60)
        print("System Prompt:\n")
        sys_prompt = await env.system_prompt()
        print(sys_prompt["obs_str"])
        print("=" * 60)

        # Reset
        obs, _ = await env.reset(seed=0)

        print("\nInitial Observation:\n")
        print(obs["obs_str"])

        step = 0

        while True:
            step += 1
            print(f"\n--- Step {step} ---")

            try:
                action_input = input("Enter action (<search>...</search> or <final>...</final>) or 'quit': ")
            except EOFError:
                action_input = "quit"

            if action_input.lower() == "quit":
                break

            obs, reward, done, info = await env.step(action_input)

            print(f"\nReward: {reward}")
            print(f"Done: {done}")
            print(f"Info: {info}")

            print("\nObservation:")
            print(obs["obs_str"])

            if done:
                if info.get("success", False):
                    print("\n✅ Correct Answer!")
                else:
                    print("\n❌ Wrong or terminated.")
                break

        print("\nTotal Reward:", env.total_reward)

        await env.close()

    def main(**kwargs):
        asyncio.run(main_async(**kwargs))

    fire.Fire(main)
