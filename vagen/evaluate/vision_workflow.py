# All comments are in English.
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
    ):
        self.adapter = adapter
        self.dump_dir = dump_dir
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
                text = " ".join(p.get("text", "") for p in c if p.get("type") == "text")
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
            # Reset and obtain system/user initial messages
            obs, info = await env.reset(seed=seed)
            infos.append(info)

            # Normal execution path (this episode WILL be dumped)
            sys_obs = await env.system_prompt()
            sys_text = sys_obs.get("obs_str", "")
            sys_imgs = extract_images(sys_obs)
            messages.append(self.adapter.format_system(sys_text, sys_imgs))

            user_text = obs.get("obs_str", "")
            user_imgs = extract_images(obs)
            messages.append(self.adapter.format_user_turn(user_text, user_imgs))
            user_imgs_per_turn.append(user_imgs)

            for t in range(turn_limit):
                # Safeguard completion
                try:
                    reply = await self.adapter.acompletion(messages, **self.chat_config)
                except OpenAIError as e:
                    error_info = {
                        "provider_error": repr(e),
                        "error_type": type(e).__name__,
                        "message": str(e),
                    }
                    logger.info("Rollout %s provider error: %s", rid, repr(e))
                    finish_reason = "model_error"
                    terminated = False
                    break
                except Exception as e:
                    error_info = {
                        "unexpected_error": repr(e),
                        "error_type": type(e).__name__,
                        "message": str(e),
                    }
                    logger.info("Rollout %s unexpected model error: %s", rid, repr(e))
                    finish_reason = "model_error"
                    terminated = False
                    break

                assistant_texts.append(reply)
                messages.append({"role": "assistant", "content": [{"type": "text", "text": reply}]})

                try:
                    next_obs, r, done, step_info = await env.step(reply)
                except Exception as e:
                    error_info = {
                        "env_step_error": repr(e),
                        "error_type": type(e).__name__,
                        "message": str(e),
                    }
                    logger.info("Rollout %s env step error: %s", rid, repr(e))
                    finish_reason = "env_error"
                    terminated = False
                    break

                rewards.append(float(r))
                cumulative_reward += float(r)
                infos.append(step_info or {})

                user_text = next_obs.get("obs_str", "")
                user_imgs = extract_images(next_obs)
                messages.append(self.adapter.format_user_turn(user_text, user_imgs))
                user_imgs_per_turn.append(user_imgs)

                if done:
                    terminated = True
                    finish_reason = "done"
                    break
                if t + 1 >= turn_limit:
                    finish_reason = "max_turns"
                    break

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
