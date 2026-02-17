import asyncio
import re
import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from vagen.envs.gym_image_env import GymImageEnv

# ------------------------------
# Minimal Local Retriever (TF-IDF)
# ------------------------------
class TfidfLocalRetriever:
    """
    Very small local retriever: fit TF-IDF on corpus texts once per episode (or cache globally).
    Returns top-k docs with scores.
    """
    def __init__(self):
        self._vectorizer = None
        self._X = None
        self._docs = None

    def fit(self, docs: List[Dict[str, Any]]) -> None:
        # Lazy import to avoid hard dependency at import time
        from sklearn.feature_extraction.text import TfidfVectorizer
        self._docs = docs
        texts = [d["text"] for d in docs]
        self._vectorizer = TfidfVectorizer(stop_words="english", max_features=50000)
        self._X = self._vectorizer.fit_transform(texts)

    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        from sklearn.metrics.pairwise import cosine_similarity
        if self._vectorizer is None or self._X is None or self._docs is None:
            raise RuntimeError("Retriever not fitted. Call fit(docs) first.")
        qv = self._vectorizer.transform([query])
        scores = cosine_similarity(qv, self._X).ravel()
        # top-k indices
        idxs = scores.argsort()[::-1][:top_k]
        out = []
        for i in idxs:
            d = self._docs[int(i)]
            out.append({
                "id": d.get("id", str(i)),
                "text": d["text"],
                "score": float(scores[int(i)]),
                "title": d.get("title", None),
            })
        return out


# ------------------------------
# Prompt helpers (SearchR1 style)
# ------------------------------
def _system_prompt() -> str:
    return (
        "You are a Search-R1 style agent. You can iteratively SEARCH a local corpus and then produce a FINAL answer.\n\n"
        "## Action format (choose ONE each step)\n"
        "1) Search:\n"
        "   <think>...</think><search>your query</search>\n"
        "2) Final answer:\n"
        "   <think>...</think><final>your answer</final>\n\n"
        "## Rules\n"
        "- Use SEARCH when you lack evidence.\n"
        "- Use FINAL only when you are confident.\n"
        "- Keep queries short and specific.\n"
        "- Do NOT hallucinate: rely on retrieved snippets.\n"
    )

def _format_observation(question: str, evidence: List[Dict[str, Any]], budgets: Dict[str, int]) -> str:
    ev_lines = []
    for i, e in enumerate(evidence, 1):
        title = f" ({e['title']})" if e.get("title") else ""
        ev_lines.append(f"[{i}] id={e.get('id')}{title} score={e.get('score', 0):.3f}\n{e.get('text','')}")
    ev_block = "\n\n".join(ev_lines) if ev_lines else "(none yet)"
    return (
        f"Question:\n{question}\n\n"
        f"Budgets: remaining_searches={budgets['remaining_searches']} remaining_steps={budgets['remaining_steps']}\n\n"
        f"Retrieved evidence (top snippets so far):\n{ev_block}\n\n"
        "Respond with either <search>...</search> or <final>...</final>.\n"
    )

def _normalize_answer(s: str) -> str:
    s = s.lower().strip()
    s = re.sub(r"\s+", " ", s)
    # remove punctuation (simple)
    s = re.sub(r"[^\w\s]", "", s)
    return s

def _exact_match(pred: str, gold: str) -> bool:
    return _normalize_answer(pred) == _normalize_answer(gold)

def _parse_action(action_str: str) -> Dict[str, Any]:
    """
    Robust action parser.
    Accepts:
      - <search>...</search> or <final>...</final>
      - JSON: {"action":"search","query":"..."} or {"action":"final","answer":"..."}
    """
    out = {
        "format_correct": False,
        "action_type": None,   # "search" | "final"
        "query": None,
        "answer": None,
        "raw": action_str,
    }

    s = action_str.strip()

    # Try JSON first
    if s.startswith("{") and s.endswith("}"):
        try:
            obj = json.loads(s)
            act = (obj.get("action") or obj.get("type") or "").lower()
            if act == "search":
                q = obj.get("query") or obj.get("q")
                if isinstance(q, str) and q.strip():
                    out.update(format_correct=True, action_type="search", query=q.strip())
                    return out
            if act in ["final", "answer"]:
                a = obj.get("answer") or obj.get("final")
                if isinstance(a, str) and a.strip():
                    out.update(format_correct=True, action_type="final", answer=a.strip())
                    return out
        except Exception:
            pass

    # Tag-based parsing
    m_search = re.search(r"<search>(.*?)</search>", s, re.DOTALL | re.IGNORECASE)
    if m_search:
        q = m_search.group(1).strip()
        if q:
            out.update(format_correct=True, action_type="search", query=q)
            return out

    m_final = re.search(r"<final>(.*?)</final>", s, re.DOTALL | re.IGNORECASE)
    if m_final:
        a = m_final.group(1).strip()
        if a:
            out.update(format_correct=True, action_type="final", answer=a)
            return out

    return out


# ------------------------------
# VAGEN SearchR1 Env
# ------------------------------
@dataclass
class SearchR1EnvConfig:
    # Dataset: list of episodes. Each episode: {"question": str, "answer": str, "corpus": [{"id","text",...}, ...]}
    dataset: Optional[List[Dict[str, Any]]] = None

    # Episode budgets
    max_steps: int = 8
    max_searches: int = 4
    top_k: int = 5
    max_evidence_items: int = 15  # cap memory in prompt

    # Rewards
    format_reward: float = 0.01
    search_penalty: float = -0.01
    correct_reward: float = 1.0
    wrong_reward: float = 0.0

    # If True: on invalid format, end episode early (strict); else just penalty and continue
    strict_format: bool = False

    # Prompt format
    include_system_example: bool = False  # placeholder, if you want to add few-shot later


class SearchR1Env(GymImageEnv):
    """
    A minimal Search-R1 style environment for VAGEN.
    - Observation: question + retrieved snippets + budgets.
    - Actions: SEARCH(query) or FINAL(answer) via tags/JSON.
    - Local retrieval: TF-IDF baseline (pluggable).
    """
    def __init__(self, env_config: Dict[str, Any]):
        super().__init__(env_config)
        self.config = SearchR1EnvConfig(**env_config)
        if not self.config.dataset:
            raise ValueError("SearchR1Env requires env_config['dataset'] as a list of episodes.")

        self._retriever = TfidfLocalRetriever()

        # Episode state
        self._episode: Optional[Dict[str, Any]] = None
        self._question: str = ""
        self._gold: str = ""
        self._evidence: List[Dict[str, Any]] = []
        self._steps_used: int = 0
        self._searches_used: int = 0
        self.total_reward: float = 0.0

    async def close(self) -> None:
        return

    async def system_prompt(self) -> Dict[str, Any]:
        return {"obs_str": _system_prompt()}

    async def reset(self, seed: int) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        # deterministic pick by seed
        idx = seed % len(self.config.dataset)
        self._episode = self.config.dataset[idx]
        self._question = str(self._episode["question"])
        self._gold = str(self._episode["answer"])
        self._evidence = []
        self._steps_used = 0
        self._searches_used = 0
        self.total_reward = 0.0

        corpus = self._episode.get("corpus", [])
        if not isinstance(corpus, list) or not corpus:
            raise ValueError("Each episode must contain a non-empty 'corpus' list of {text,...} docs.")
        # Fit retriever (can be cached globally if your corpus is fixed)
        await asyncio.to_thread(self._retriever.fit, corpus)

        obs = {"obs_str": self._render_obs()}
        return obs, {}

    async def step(self, action_str: str) -> Tuple[Dict[str, Any], float, bool, Dict[str, Any]]:
        self._steps_used += 1
        reward = 0.0
        done = False
        info: Dict[str, Any] = {}

        parsed = _parse_action(action_str)
        info.update(parsed)

        # Budget check
        remaining_steps = max(0, self.config.max_steps - self._steps_used)
        remaining_searches = max(0, self.config.max_searches - self._searches_used)

        # Format shaping
        if parsed["format_correct"]:
            reward += self.config.format_reward
        else:
            info["error"] = "format_invalid"
            if self.config.strict_format:
                done = True

        # Execute action
        if not done and parsed["action_type"] == "search":
            if remaining_searches <= 0:
                info["error"] = "no_search_budget"
            else:
                self._searches_used += 1
                reward += self.config.search_penalty
                q = parsed["query"]
                results = await asyncio.to_thread(self._retriever.search, q, self.config.top_k)

                # Append evidence with cap + de-dup by id
                seen = {e.get("id") for e in self._evidence}
                for r in results:
                    if r.get("id") not in seen:
                        self._evidence.append(r)
                        seen.add(r.get("id"))
                self._evidence = self._evidence[: self.config.max_evidence_items]

        elif not done and parsed["action_type"] == "final":
            pred = parsed["answer"]
            correct = _exact_match(pred, self._gold)
            info["correct"] = correct
            reward += self.config.correct_reward if correct else self.config.wrong_reward
            done = True

        # Terminate if step budget exhausted
        if not done and self._steps_used >= self.config.max_steps:
            info["error"] = info.get("error", "max_steps_reached")
            done = True

        # Metrics
        info["metrics"] = {
            "turn_metrics": {
                "format_correct": parsed["format_correct"],
                "action_type": parsed["action_type"],
                "searches_used": self._searches_used,
                "steps_used": self._steps_used,
            },
            "traj_metrics": {
                "success": bool(info.get("correct", False)),
            },
        }
        info["success"] = bool(info.get("correct", False))

        self.total_reward += reward
        obs = {"obs_str": self._render_obs()}
        return obs, reward, done, info

    def _render_obs(self) -> str:
        budgets = {
            "remaining_steps": max(0, self.config.max_steps - self._steps_used),
            "remaining_searches": max(0, self.config.max_searches - self._searches_used),
        }
        return _format_observation(self._question, self._evidence, budgets)


# ------------------------------
# Local async test (optional)
# ------------------------------
if __name__ == "__main__":
    import fire

    # A tiny toy dataset example
    TOY = [
        {
            "question": "Who wrote The Old Man and the Sea?",
            "answer": "Ernest Hemingway",
            "corpus": [
                {"id": "a", "text": "The Old Man and the Sea is a short novel written by the American author Ernest Hemingway."},
                {"id": "b", "text": "Hemingway also wrote A Farewell to Arms and For Whom the Bell Tolls."},
                {"id": "c", "text": "Moby-Dick is a novel by Herman Melville."},
            ],
        }
    ]

    async def main_async():
        env = SearchR1Env({"dataset": TOY, "max_steps": 6, "max_searches": 3, "top_k": 2})
        print((await env.system_prompt())["obs_str"])
        obs, _ = await env.reset(seed=0)
        print(obs["obs_str"])

        while True:
            a = input("\nAction: ")
            obs, r, done, info = await env.step(a)
            print(f"reward={r} done={done} info={info.get('error', '')} correct={info.get('correct', None)}")
            print(obs["obs_str"])
            if done:
                print("TOTAL:", env.total_reward, "SUCCESS:", info.get("success", False))
                break

    def main():
        asyncio.run(main_async())

    fire.Fire(main)
