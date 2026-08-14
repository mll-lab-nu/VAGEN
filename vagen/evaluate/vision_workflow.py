from __future__ import annotations
from typing import Any, Dict, List, Optional, Literal
from PIL import Image
import os
import json
import asyncio
import uuid
import logging

from vagen.evaluate.adapters.base_adapter import ModelAdapter
from vagen.evaluate.utils.mm_utils import _now_tag, extract_images
from vagen.evaluate.utils.json_utils import sanitize_for_json
from vagen.core.env_adapter import GymEnvAdapter
from vagen.core.runner import run_episode
from vagen.evaluate.chat_client import ChatClient
from vagen.harness import HARNESSES, build_harness
from vagen.harness.budget import DEFAULT_MAX_ENV_RESPONSE, default_summary_budget

logger = logging.getLogger(__name__)

# Optional: import provider error base class
try:
    import openai
    OpenAIError = openai.OpenAIError  # type: ignore
except Exception:  # pragma: no cover
    OpenAIError = Exception


class GenericVisionInferenceWorkflow:
    """
    Drive a Gym-like vision environment with a ModelAdapter.
    """

    def __init__(
        self,
        adapter: ModelAdapter,
        dump_dir: Optional[str] = None,
        dump_enabled: bool = True,  # kept for API compatibility; ignored in logic below
        success_keys: Optional[List[str]] = None,
        success_threshold: float = 0.99,
        chat_config: Optional[Dict[str, Any]] = None,
        harness: str = "concat",
        response_length_per_turn: Optional[int] = None,
        max_env_response_per_turn: Optional[int] = None,
        compact_budget: Optional[int] = None,
        compact_summary_budget: Optional[int] = None,
        tokenizer: Any = None,
        tokens_per_image: Optional[int] = None,
    ):
        self.adapter = adapter
        self.dump_dir = dump_dir
        # ★ The context policy is the same object training uses, chosen by name. It was a
        # bool here (`concat_multi_turn`), which could express concat and approximately
        # no_concat and could not express compaction at all -- training deleted that bool
        # rather than deprecating it, precisely so a stale config would be rejected instead
        # of quietly outvoting `harness`.
        if harness not in HARNESSES:
            raise ValueError(f"unknown harness {harness!r}; choose from {sorted(HARNESSES)}")
        self.harness_name = harness
        self.response_length_per_turn = response_length_per_turn
        self.max_env_response_per_turn = max_env_response_per_turn
        self.compact_budget = compact_budget
        self.compact_summary_budget = compact_summary_budget
        self.tokenizer = tokenizer
        # See chat_client.DEFAULT_TOKENS_PER_IMAGE: this feeds the compaction trigger, not
        # just an overflow guard, so a value far off the environment's real frame cost
        # makes `compact` close conversations before they have bought a turn.
        self.tokens_per_image = tokens_per_image
        # IMPORTANT: dump_enabled is ignored; we always dump for executed episodes
        self.dump_enabled = True
        self.success_keys = success_keys or ["success", "is_success", "solved"]
        self.success_threshold = success_threshold
        self.chat_config = dict(chat_config or {})
        if self.dump_dir:
            os.makedirs(self.dump_dir, exist_ok=True)

    async def _dump(
        self,
        rid: str,
        messages: List[Dict[str, Any]],
        assistant_texts: List[str],
        user_imgs_per_turn: List[List[Image.Image]],
        metrics: Optional[Dict[str, Any]] = None,
        dump_root: Optional[str] = None,
    ) -> None:
        """Persist messages/images/transcript and optional metrics."""
        base_dir = dump_root or self.dump_dir
        if not base_dir:
            return
        folder = os.path.join(base_dir, rid)
        os.makedirs(folder, exist_ok=True)
        # Sanitize metrics for JSON
        if metrics is not None:
            metrics = sanitize_for_json(metrics)

        def shadow(m: Dict[str, Any]) -> Dict[str, Any]:
            r = m.get("role", "")
            c = m.get("content")
            if isinstance(c, list):
                parts = []
                for p in c:
                    if p.get("type") == "text":
                        parts.append({"type": "text", "text": p.get("text", "")})
                    elif p.get("type") == "image_url":
                        parts.append({"type": "image_url", "image_url": {"url": "<data_url>"}})
                out = {"role": r, "content": parts}
            else:
                out = {"role": r, "content": c}
            return out

        # messages.json
        await asyncio.to_thread(
            lambda: open(os.path.join(folder, "messages.json"), "w", encoding="utf-8").write(
                json.dumps([shadow(m) for m in messages], ensure_ascii=False, indent=2)
            )
        )
        # assistant_texts.json
        await asyncio.to_thread(
            lambda: open(os.path.join(folder, "assistant_texts.json"), "w", encoding="utf-8").write(
                json.dumps(assistant_texts, ensure_ascii=False, indent=2)
            )
        )

        # Save user images
        img_dir = os.path.join(folder, "images")
        os.makedirs(img_dir, exist_ok=True)
        for t, imgs in enumerate(user_imgs_per_turn, start=1):
            for i, img in enumerate(imgs, start=1):
                path = os.path.join(img_dir, f"turn_{t:02d}_{i:02d}.png")
                await asyncio.to_thread(img.save, path, "PNG")

        # transcript.txt
        def to_line(m: Dict[str, Any]) -> str:
            role = m.get("role", "").upper()
            c = m.get("content")
            if isinstance(c, list):
                text = " ".join(p.get("text", "") for p in c if p.get("type") in ("text", "input_text", "output_text"))
            else:
                text = c or ""
            return f"{role}: {text.strip()}"

        transcript = "\n\n".join(to_line(m) for m in messages)
        await asyncio.to_thread(
            lambda: open(os.path.join(folder, "transcript.txt"), "w", encoding="utf-8").write(transcript)
        )

        # metrics.json
        if metrics is not None:
            await asyncio.to_thread(
                lambda: open(os.path.join(folder, "metrics.json"), "w", encoding="utf-8").write(
                    json.dumps(metrics, ensure_ascii=False, indent=2)
                )
            )


    def _build_harness(self, max_turns: int):
        """The harness, and the two per-call ceilings the client enforces.

        Deliberately no static budget check and no prompt region. Those exist in training
        because a conversation has to fit a training row; evaluation builds no rows, which
        is the case ``BaseHarness`` documents as ``response_len=None``. Setting
        ``response_length_per_turn`` in the eval config turns the accounting back on, and
        then a conversation is bounded exactly as it is in training.

        ★ The opening ceiling is None rather than a number. Deriving it from a
        prompt region that evaluation does not have gave zero, and a zero ceiling rejects
        the system prompt itself -- every episode died at its first call, having made no
        model call at all, and reported cleanly as a zero-turn episode.
        """
        per_turn = self.response_length_per_turn
        response_len = per_turn * max_turns if per_turn else None

        kw = {"response_len": response_len}
        if per_turn:
            kw["floor"] = per_turn
        if self.harness_name == "compact":
            budget = self.compact_budget
            summary = self.compact_summary_budget
            if budget and not summary:
                summary = default_summary_budget(budget, per_turn or budget)
            kw.update(budget=budget, summary_budget=summary)
        harness = build_harness(self.harness_name, **kw)

        # An observation is still bounded: that ceiling is a property of the environment,
        # not of whether there is a row to fit.
        continuation = self.max_env_response_per_turn
        if continuation is None:
            continuation = DEFAULT_MAX_ENV_RESPONSE
        return harness, None, continuation

    async def arun_episode(
        self,
        env_cls,
        env_config,
        seed,
        *,
        rollout_id: Optional[str] = None,
        dump_override: Optional[str] = None,
        max_turns: int = 1,
        episode_metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Run a single rollout and return episode results. Never raises.
        """
        env = env_cls(env_config)
        dump_root: Optional[str]
        if isinstance(dump_override, str) and dump_override:
            dump_root = dump_override
        else:
            dump_root = self.dump_dir
        if dump_root:
            os.makedirs(dump_root, exist_ok=True)
        rid = rollout_id or f"{_now_tag()}-{uuid.uuid4().hex[:8]}"
        env_config_dump = sanitize_for_json(env_config)

        messages: List[Dict[str, Any]] = []
        assistant_texts: List[str] = []
        user_imgs_per_turn: List[List[Image.Image]] = []

        rewards: List[float] = []
        infos: List[Dict[str, Any]] = []
        cumulative_reward: float = 0.0
        terminated: bool = False
        finish_reason: str = "max_turns"
        error_info: Optional[Dict[str, Any]] = None
        metadata = dict(episode_metadata or {})
        turn_limit = int(max_turns)
        assert turn_limit > 0, f"Invalid max_turns={turn_limit} in workflow"

        try:
            # ★ The shared episode loop, the same one training runs. `core/runner.py` was
            # written to drive "a verl rollout and a closed chat API"; this is the second
            # caller it was waiting for. Everything the old hand-rolled loop here did by
            # hand -- deciding what history a call carries, when a conversation is full,
            # how much of the budget a turn may spend -- is the harness's job, and doing it
            # twice is how evaluation ended up unable to express compaction.
            harness, opening, continuation = self._build_harness(turn_limit)
            client_kw = {} if self.tokens_per_image is None else {
                "tokens_per_image": self.tokens_per_image}
            client = ChatClient(self.adapter, self.chat_config, tokenizer=self.tokenizer,
                                response_limit=self.response_length_per_turn, **client_kw)
            client.opening_limit, client.continuation_limit = opening, continuation

            adapted = GymEnvAdapter(env, env_config.get("name", "env"), env_config)
            outcome = await run_episode(adapted, harness, client, seed=seed,
                                        max_turns=turn_limit)

            rewards = list(outcome.rewards)
            cumulative_reward = float(outcome.total_reward)
            terminated = bool(outcome.terminated)
            infos.append(outcome.info or {})
            finish_reason = "done" if terminated else "max_turns"

            # The transcript is what the client actually sent, across every conversation
            # the harness opened. Under no_concat and compact that is more than one, and
            # reading only the last would report a fraction of the episode.
            for conv in client.conversations():
                messages.extend(client.messages(conv.conversation_id))
            assistant_texts = [_text_of(m) for m in messages if m.get("role") == "assistant"]

            # Success heuristic
            success = False
            if infos:
                last_info = infos[-1]
                for k in self.success_keys:
                    if k in last_info:
                        success = bool(last_info[k])
                        break
            if not success and terminated and rewards:
                success = rewards[-1] > self.success_threshold

            # Merge error info into infos for metrics
            final_infos = list(infos)
            if error_info is not None:
                final_infos = [*final_infos, error_info]

            # Always dump executed episodes (ignore dump_override)
            metrics = {
                "rollout_id": rid,
                "seed": seed,
                "terminated": terminated,
                "finish_reason": finish_reason,
                "success": success,
                "cumulative_reward": float(cumulative_reward),
                "rewards": rewards,
                "num_turns": len(assistant_texts),
                "infos": final_infos,
                "env_config": env_config_dump,
            }
            metrics.setdefault("max_turns", turn_limit)
            if error_info is not None:
                metrics["error_details"] = error_info
            if metadata:
                metrics.update(metadata)
            await self._dump(
                rid,
                messages,
                assistant_texts,
                user_imgs_per_turn,
                metrics=sanitize_for_json(metrics),
                dump_root=dump_root,
            )

            result = {
                "rollout_id": rid,
                "final_text": assistant_texts[-1] if assistant_texts else "",
                "num_turns": len(assistant_texts),
                "messages": messages,
                "terminated": terminated,
                "finish_reason": finish_reason,
                "success": success,
                "cumulative_reward": float(cumulative_reward),
                "rewards": rewards,
                "infos": final_infos,
                "seed": seed,
            }
            if error_info is not None:
                result["error_details"] = error_info
            result.setdefault("max_turns", turn_limit)
            if metadata:
                result.update(metadata)
            return result
        except Exception as e:
            # Last resort: never propagate exceptions out
            # Even if an exception occurs before dumping, we try to dump a minimal metrics with error info.
            try:
                logger.info("Rollout %s failed with exception: %s", rid, repr(e))
                if dump_root:
                    minimal_metrics = {
                        "rollout_id": rid,
                        "seed": seed,
                        "terminated": False,
                        "finish_reason": "error",
                        "success": False,
                        "cumulative_reward": 0.0,
                        "rewards": [],
                        "num_turns": 0,
                        "infos": (infos or []) + [{"error": repr(e)}],
                        "env_config": env_config_dump,
                        "error_details": {
                            "error": repr(e),
                            "error_type": type(e).__name__,
                            "message": str(e),
                        },
                    }
                    minimal_metrics.setdefault("max_turns", turn_limit)
                    if metadata:
                        minimal_metrics.update(metadata)
                    await self._dump(
                        rid,
                        messages,
                        assistant_texts,
                        user_imgs_per_turn,
                        metrics=minimal_metrics,
                        dump_root=dump_root,
                    )
            except Exception:
                pass

            result = {
                "rollout_id": f"ERR-{uuid.uuid4().hex[:8]}",
                "final_text": "",
                "num_turns": 0,
                "messages": [],
                "terminated": False,
                "finish_reason": "error",
                "success": False,
                "cumulative_reward": 0.0,
                "rewards": [],
                "infos": (infos or []) + [{"error": repr(e)}],
                "seed": seed,
                "error": repr(e),
            }
            result["error_details"] = {
                "error": repr(e),
                "error_type": type(e).__name__,
                "message": str(e),
            }
            result.setdefault("max_turns", turn_limit)
            if metadata:
                result.update(metadata)
            return result
        finally:
            try:
                await env.close()
            except Exception:
                pass


def _text_of(message: dict) -> str:
    """The plain text of an API message, whatever content shape the adapter used."""
    content = message.get("content", "")
    if isinstance(content, str):
        return content
    return "".join(p.get("text", "") for p in content if isinstance(p, dict))
