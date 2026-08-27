from __future__ import annotations
import asyncio
import json
import logging
import os
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

from vagen.evaluation.workflow import GenericVisionInferenceWorkflow
from vagen.evaluation.backends import REGISTRY
from vagen.evaluation.backends._common.retry import ThrottledAdapter, ThrottleRetryPolicy

# Summary writing by scanning dump_dir (not from in-memory results)
try:
    from vagen.evaluation.summary import write_rollouts_summary_from_dump
except Exception:
    write_rollouts_summary_from_dump = None  # type: ignore

logger = logging.getLogger("vagen.evaluation.runner")


def _safe_read_json(p: Path) -> Optional[Dict[str, Any]]:
    """Best-effort JSON reader; returns None on any error."""
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None


# One definition, in summary_utils, because both this module and the summary writer key
# deletion, resume and reporting off it -- and they used to disagree.
from vagen.evaluation.summary import NORMAL_FINISH_REASONS  # noqa: E402


def _load_sizers(name):
    """``(processor, tokenizer)`` for a model id or path; either may be None.

    A processor is what makes sizing exact, because it is the only thing that can price a
    frame -- and the frame is the part an estimate gets badly wrong. A tokenizer alone
    still fixes the text. Neither is required: failure is a warning, and the client falls
    back to characters-and-a-constant.
    """
    if not name:
        return None, None
    processor = tokenizer = None
    try:
        from transformers import AutoProcessor
        processor = AutoProcessor.from_pretrained(name)
    except Exception as exc:  # noqa: BLE001 - text-only models have no processor
        logger.info("no processor for %r (%s); frames will be estimated", name, exc)
    try:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(name)
    except Exception as exc:  # noqa: BLE001
        logger.warning("could not load a tokenizer for %r (%s); sizes will be estimated "
                       "from characters", name, exc)
    return processor, tokenizer


async def run_eval_parallel(
    jobs: List[Dict[str, Any]],
    *,
    backend: str,
    backend_cfg: Dict[str, Any],
    model: str,
    default_max_turns: int,
    dump_dir: Optional[str] = "./rollouts",
    max_concurrent_jobs: int = 4,
    live_summary: bool = False,
) -> List[Dict[str, Any]]:
    """
    Single-backend parallel runner (fault-tolerant + live summary).
    - Build ONE client and ONE adapter for the given backend/model.
    - Run episodes in parallel with an episode-level gate and a shared request-level gate.
    - No task exception will abort the batch; a structured failure record is returned instead.
    - Live summary: refresh summary.json by scanning dump_dir on each episode completion.
    """

    # Build client/adapter once
    client = REGISTRY.build_client(backend, backend_cfg)
    adapter_kwargs = dict(
        client=client,
        model=model,
    )
    base_adapter_factory = lambda **kw: REGISTRY.build_adapter(backend, **{**adapter_kwargs, **kw})

    # Concurrency gates
    episode_gate = asyncio.Semaphore(max(1, max_concurrent_jobs))
    req_gate = asyncio.BoundedSemaphore(max(1, int(backend_cfg.get("max_concurrency", 2))))

    policy = ThrottleRetryPolicy(
        max_concurrency=9999,  # ignored; we use shared gate
        max_retries=int(backend_cfg.get("max_retries", 6)),
        min_backoff=float(backend_cfg.get("min_backoff", 0.5)),
        max_backoff=float(backend_cfg.get("max_backoff", 8.0)),
        shared_gate=req_gate,
    )

    async def _runner(data: Dict[str, Any], per_job_adapter_kwargs: Dict[str, Any]) -> Dict[str, Any]:
        """Run one episode safely, never raising exceptions out of this function.

        ★ The whole body is inside the guard, including the setup. It was not: the tag_id
        check, the max_turns assert, the adapter build and the workflow construction all
        sat above the try, and `await fut` re-raises -- so ONE malformed job aborted the
        batch and discarded every episode that had already finished. A bad `harness:` value
        reached here the same way.
        """
        try:
            return await _run_one(data, per_job_adapter_kwargs)
        except Exception as exc:  # noqa: BLE001
            logger.exception("Job setup failed for env=%s seed=%s",
                             data.get("env_name"), data.get("seed"))
            return {"rollout_id": f"ERR-{uuid.uuid4().hex[:8]}", "seed": data.get("seed"),
                    "tag_id": data.get("tag_id"), "split": data.get("split"),
                    "env_name": data.get("env_name"), "success": False,
                    "num_turns": 0, "cumulative_reward": 0.0, "rewards": [],
                    "terminated": False, "finish_reason": "setup_error",
                    # ★ Both keys. `main()` reports errors off "error"; the in-episode
                    # failure path sets it and this one did not, so a setup failure was
                    # printed as an ordinary status line and never reached the summary.
                    "error": repr(exc),
                    "error_details": {"error": repr(exc), "error_type": type(exc).__name__}}

    async def _run_one(data: Dict[str, Any], per_job_adapter_kwargs: Dict[str, Any]) -> Dict[str, Any]:
        adapter = ThrottledAdapter(base_adapter_factory(**per_job_adapter_kwargs), policy)
        env_config: Dict[str, Any] = data["env_config"]

        tag_id = data.get("tag_id")
        # Keep tag_id as original type (int or str)
        if tag_id is None:
            raise ValueError(f"Env '{data.get('env_name')}' is missing tag_id.")
        if not isinstance(tag_id, (int, str)):
            tag_id = str(tag_id)

        data["tag_id"] = tag_id

        turn_limit_int = int(data.get("max_turns", default_max_turns))
        # Not an assert: under `python -O` that vanishes, and max_turns=0 then runs
        # range(0) and reports a full set of 0-turn "normal" episodes.
        if turn_limit_int <= 0:
            raise ValueError(f"max_turns={turn_limit_int} for env '{data.get('env_name')}'")


        episode_metadata = {
            "tag_id": tag_id,
            "split": data.get("split"),
            "env_name": data.get("env_name"),
        }
        episode_metadata = {k: v for k, v in episode_metadata.items() if v is not None}
        episode_metadata["max_turns"] = turn_limit_int

        tag_dump_dir: Optional[str] = None
        if dump_dir:
            tag_dump_dir = os.path.join(dump_dir, f"tag_{tag_id}")
            os.makedirs(tag_dump_dir, exist_ok=True)

        wf = GenericVisionInferenceWorkflow(
            adapter=adapter,
            dump_dir=dump_dir,
            dump_enabled=True,  # ignored in workflow; always dump executed episodes
            chat_config=data.get("chat_config") or {},
            harness=data.get("harness", "concat"),
            response_length_per_turn=data.get("response_length_per_turn"),
            max_response_length=data.get("max_response_length"),
            max_env_response_per_turn=data.get("max_env_response_per_turn"),
            compact_budget=data.get("compact_budget"),
            compact_summary_budget=data.get("compact_summary_budget"),
            tokens_per_image=data.get("tokens_per_image"),
            **dict(zip(("processor", "tokenizer"),
                       _load_sizers(data.get("tokenizer") or model))),
        )
        async with episode_gate:
            logger.info(
                "Job start env=%s tag=%s seed=%s config=%s",
                data.get("env_name"),
                tag_id,
                data.get("seed"),
                data.get("env_config"),
            )
            try:
                result = await wf.arun_episode(
                    env_cls=data["env_cls"],
                    env_config=env_config,
                    seed=data["seed"],
                    rollout_id=None,
                    dump_override=tag_dump_dir,
                    max_turns=turn_limit_int,
                    episode_metadata=episode_metadata or None,
                )
                if episode_metadata and isinstance(result, dict):
                    for k, v in episode_metadata.items():
                        result.setdefault(k, v)
                logger.info(
                    "Job finish env=%s tag=%s seed=%s config=%s rid=%s reason=%s",
                    data.get("env_name"),
                    tag_id,
                    data.get("seed"),
                    data.get("env_config"),
                    result.get("rollout_id"),
                    result.get("finish_reason"),
                )
                return result
            except Exception as e:
                # Try to keep running; this record will not be scanned by summary unless it dumped successfully
                failure = {
                    "rollout_id": f"ERR-{uuid.uuid4().hex[:8]}",
                    "error": repr(e),
                    "seed": data.get("seed"),
                }
                if episode_metadata:
                    failure.update(episode_metadata)
                if tag_dump_dir:
                    failure.setdefault("dump_dir", tag_dump_dir)
                logger.exception(
                    "Job error env=%s tag=%s seed=%s: %s",
                    data.get("env_name"),
                    tag_id,
                    data.get("seed"),
                    e,
                )
                return failure

    # Launch with as_completed for live summary
    tasks: List[asyncio.Task] = []
    for j in jobs:
        data = j["data"]
        per_job_adapter_kwargs = j.get("adapter_kwargs", {})
        tasks.append(asyncio.create_task(_runner(data, per_job_adapter_kwargs)))

    results: List[Dict[str, Any]] = []
    for fut in asyncio.as_completed(tasks):
        item = await fut
        results.append(item)

        # Live summary refresh by scanning dump_dir
        if live_summary and write_rollouts_summary_from_dump and dump_dir:
            tag_val = item.get("tag_id")
            if tag_val is not None:
                try:
                    tag_dir = os.path.join(dump_dir, f"tag_{tag_val}")
                    # model=... for the same reason main() passes it: a dump directory can hold
                    # more than one checkpoint, and an unfiltered summary averages them.
                    write_rollouts_summary_from_dump(dump_dir=tag_dir,
                                                     filename="summary.json",
                                                     model=model)
                except Exception:
                    # Best effort: never fail the run due to summary writing
                    pass

    return results
