from __future__ import annotations
from typing import Any, Dict, List, Optional
from PIL import Image
import os
import json
import asyncio
import uuid
import logging

from vagen.evaluation.backends import ModelAdapter
from vagen.evaluation.backends._common.rendering import _now_tag
from vagen.evaluation.serialization import sanitize_for_json
from vagen.envs import GymEnvAdapter
from vagen.rollout import run_episode
from vagen.envs import build_env, state_reward_names
from vagen.evaluation.client import ChatClient
from vagen.harness import build_harness, resolve_harness
from vagen.harness.compact import CompactHarness
from vagen.harness._common.budget import DEFAULT_MAX_ENV_RESPONSE, default_summary_budget

logger = logging.getLogger(__name__)



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
        max_response_length: Optional[int] = None,
        max_env_response_per_turn: Optional[int] = None,
        compact_budget: Optional[int] = None,
        compact_summary_budget: Optional[int] = None,
        tokenizer: Any = None,
        processor: Any = None,
        tokens_per_image: Optional[int] = None,
    ):
        self.adapter = adapter
        self.dump_dir = dump_dir
        # ★ The context policy is the same object training uses, chosen by name. It was a
        # bool here (`concat_multi_turn`), which could express concat and approximately
        # no_concat and could not express compaction at all -- training deleted that bool
        # rather than deprecating it, precisely so a stale config would be rejected instead
        # of quietly outvoting `harness`.
        # Resolved now rather than at the first episode, so a bad name fails once at
        # construction instead of once per rollout -- and any BaseHarness subclass is
        # accepted, whether registered by name or given as an import path.
        resolve_harness(harness)
        self.harness_name = harness
        self.response_length_per_turn = response_length_per_turn
        self.max_response_length = max_response_length
        self.max_env_response_per_turn = max_env_response_per_turn
        self.compact_budget = compact_budget
        self.compact_summary_budget = compact_summary_budget
        self.tokenizer = tokenizer
        self.processor = processor
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

        ★ ``response_len`` is NOT derived from ``response_length_per_turn * max_turns``.
        That looks like the episode's budget and is not: under concat the observations land
        in the same region, so ``g*T`` has to hold T generations *and* T observations, and
        ``_left()`` runs out early. Measured on the shipped frozenlake eval -- g=512, T=5,
        one frame an observation -- it ended the episode after 3 turns of 5 and still
        reported ``max_turns``. Training passes an independent ``data.max_response_length``
        for exactly this reason. Here the region is opt-in: unset, there is no accounting,
        which is the closed-API case ``BaseHarness`` documents as ``response_len=None``.

        The floor matches training's ``min(per_turn, response_len // 4)`` rather than
        ``per_turn``. A floor equal to a full 1/T of the region deletes turns on its own.
        """
        per_turn = self.response_length_per_turn
        response_len = self.max_response_length

        kw = {"response_len": response_len}
        if per_turn and response_len:
            kw["floor"] = min(per_turn, max(1, response_len // 4))
        elif per_turn:
            kw["floor"] = per_turn
        if issubclass(resolve_harness(self.harness_name), CompactHarness):
            budget = self.compact_budget
            if not budget and not response_len:
                # Neither trigger can fire, so the conversation would grow forever and this
                # would silently be concat -- the exact failure `harness:` was added to fix.
                raise ValueError(
                    "harness: compact needs compact_budget (or max_response_length) in the "
                    "eval config. With neither, no trigger can fire and it runs as concat."
                )
            summary = self.compact_summary_budget
            if not summary:
                summary = default_summary_budget(budget or response_len, per_turn or budget
                                                 or response_len)
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
        # Same factory training uses, so a `state_reward` block in envs[].config means the
        # same thing here. Evaluation has no trainer to read it from, which is most of why
        # the setting belongs to the environment.
        env = build_env(env_cls, env_config, max_turns=max_turns)
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

        # Built before the try so that a failure mid-episode still has the turns that
        # finished. They were being thrown away: the client was local to the try and the
        # transcript was only harvested on the success path, so a provider error on call 3
        # of 6 reported num_turns=0 and an empty transcript -- and `finish_reason: error`
        # then makes _purge_error_rollouts delete the dump on the next resumed run.
        harness, opening, continuation = self._build_harness(turn_limit)
        client = ChatClient(self.adapter, self.chat_config, tokenizer=self.tokenizer,
                                processor=self.processor,
                            response_limit=self.response_length_per_turn,
                            **({} if self.tokens_per_image is None
                               else {"tokens_per_image": self.tokens_per_image}))
        client.opening_limit, client.continuation_limit = opening, continuation
        adapted = _Recording(GymEnvAdapter(
            env, env_config.get("name", "env"), env_config,
            score_names=state_reward_names(env_config),
        ))
        outcome = None

        try:
            # ★ The shared episode loop, the same one training runs. `core/runner.py` was
            # written to drive "a verl rollout and a closed chat API"; this is the second
            # caller it was waiting for. Everything the old hand-rolled loop here did by
            # hand -- deciding what history a call carries, when a conversation is full,
            # how much of the budget a turn may spend -- is the harness's job, and doing it
            # twice is how evaluation ended up unable to express compaction.
            outcome = await run_episode(adapted, harness, client, seed=seed,
                                        max_turns=turn_limit)
        except Exception as e:  # noqa: BLE001 - recorded, never propagated
            error_info = {"error": repr(e), "error_type": type(e).__name__,
                          "message": str(e)}
            logger.info("Rollout %s failed with %s: %s", rid, type(e).__name__, e)
            finish_reason = "error"

        rewards = list(adapted.rewards)
        cumulative_reward = float(sum(rewards))
        infos = list(adapted.infos)
        user_imgs_per_turn = list(adapted.images)
        # The transcript is what the client actually sent, across every conversation the
        # harness opened. Under no_concat and compact that is more than one, and reading
        # only the last would report a fraction of the episode.
        for conv in client.conversations():
            messages.extend(client.messages(conv.conversation_id))
        assistant_texts = [_text_of(m) for m in messages if m.get("role") == "assistant"]

        if outcome is not None:
            terminated = bool(outcome.terminated)
            # ★ outcome.turns is environment steps. Recomputing it from the transcript
            # counts a compaction summary as a turn, so avg_turns grew with how often the
            # policy triggered compaction rather than with episode length.
            n_turns = outcome.turns
            if adapted.env_error:
                # The adapter reports a crashed env as done=True, which would otherwise be
                # indistinguishable from a solved episode in the summary.
                finish_reason, terminated = "env_error", False
            elif terminated:
                finish_reason = "done"
            elif n_turns == 0 and not adapted.rewards:
                # ★ Zero turns is not an ending, it is a non-answer: the endpoint returned
                # nothing (a refusal or a content filter) and run_episode stopped before
                # the first env step. Calling it `no_room` filed it as a completed episode,
                # so it counted as a zero in success_rate, stayed out of error_rollouts,
                # survived the purge, and was marked done by resume -- which means rerunning
                # could never repair it.
                finish_reason = "empty_generation"
            elif n_turns >= turn_limit:
                # run_episode marks running out of turns as `truncated` too, so the flag
                # alone cannot tell "used its whole budget" from "stopped early for lack
                # of room". The turn count can, and only the second is worth a name.
                finish_reason = "max_turns"
            elif outcome.truncated:
                finish_reason = "no_room"
            else:
                finish_reason = "max_turns"
        else:
            n_turns = len(rewards)

        try:
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
                "num_turns": n_turns,
                "infos": final_infos,
                "env_config": env_config_dump,
                # ★ Recorded so resume can tell whose rollout this is. Without it the key
                # is (env, seed, tag_id) alone, so evaluating a second model into the same
                # dump directory skips every episode and reprints the first model's
                # success_rate under the second model's name, exit 0, no warning.
                "model": getattr(self.adapter, "model", None) or getattr(
                    getattr(self.adapter, "inner", None), "model", None),
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
                "num_turns": n_turns,
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
                # ★ Recorded so resume can tell whose rollout this is. Without it the key
                # is (env, seed, tag_id) alone, so evaluating a second model into the same
                # dump directory skips every episode and reprints the first model's
                # success_rate under the second model's name, exit 0, no warning.
                "model": getattr(self.adapter, "model", None) or getattr(
                    getattr(self.adapter, "inner", None), "model", None),
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
        # No env.close() here: `run_episode` closes it in its own `finally`, so doing it
        # again closed every environment twice. Harmless for the shipped envs -- gym closes
        # are idempotent and the remote client guards on its session id -- but it would
        # bite the first env whose close is not.


def _text_of(message: dict) -> str:
    """The plain text of an API message, whatever content shape the adapter used."""
    content = message.get("content", "")
    if isinstance(content, str):
        return content
    return "".join(p.get("text", "") for p in content if isinstance(p, dict))


class _Recording:
    """Wraps the env adapter to keep what ``run_episode`` merges away.

    ``EpisodeResult`` carries one merged ``info`` dict and a reward list, which is all
    training needs. Evaluation's summary aligns per-turn fields on
    ``len(infos) == len(rewards) + 1`` (`summary_utils._build_per_turn_with_turn0`), so a
    single merged dict makes every turn past the first report ``{}``. And a *vision*
    harness that does not keep the frames cannot dump them: `images/` was being created
    empty on every rollout.
    """

    def __init__(self, inner):
        self.inner = inner
        self.infos: list = []
        self.rewards: list = []
        self.images: list = []
        self.env_error = False

    async def reset(self, seed=None):
        message, info = await self.inner.reset(seed)
        self.infos.append(dict(info or {}))
        self.images.append(_images_of(message))
        return message, info

    async def system_prompt(self):
        return await self.inner.system_prompt()

    async def step(self, action, **kw):
        message, reward, terminated, truncated, info = await self.inner.step(action, **kw)
        self.rewards.append(reward if isinstance(reward, (int, float)) else sum(reward))
        self.infos.append(dict(info or {}))
        self.images.append(_images_of(message))
        if (info or {}).get("env_error"):
            self.env_error = True
        return message, reward, terminated, truncated, info

    async def close(self):
        await self.inner.close()


def _images_of(message: dict) -> list:
    return list((message or {}).get("images") or [])
