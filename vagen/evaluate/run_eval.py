# All comments are in English.
from __future__ import annotations

import argparse
import asyncio
import copy
import json
import logging
import math
import os
import shutil
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

from omegaconf import DictConfig, OmegaConf, open_dict

from vagen.evaluate.register_builtins import *  # populate registry
from vagen.envs.registry import get_env_cls
from vagen.evaluate import runner as _runner_mod
from vagen.evaluate.runner import (
    run_eval_chunk_subprocess,
    run_eval_parallel,
    split_jobs_round_robin,
)
from vagen.evaluate.utils.seeding_utils import generate_seeds_for_spec
from vagen.evaluate.utils.summary_utils import write_rollouts_summary_from_dump
DEFAULT_CONFIG_PATH = os.path.join(os.path.dirname(__file__), "conf", "evaluate.yaml")


logger = logging.getLogger("view_suite.run_eval")
if not logger.handlers:
    logging.basicConfig(level=logging.INFO)


@dataclass
class EnvSpec:
    """Configuration for one logical environment family."""

    name: str
    n_envs: int
    split: str
    tag_id: Union[int, str] = 0
    config: Dict[str, Any] = field(default_factory=dict)
    chat_config: Dict[str, Any] = field(default_factory=dict)
    seed: List[int] = field(default_factory=lambda: [0])
    seed_list: Optional[List[int]] = None
    max_turns: Optional[int] = None
    concat_multi_turn: bool = True


def _looks_like_path_key(key: str) -> bool:
    low = key.lower()
    return low.endswith("_path") or low.endswith("_dir") or ("path" in low) or ("dir" in low)


def _resolve_paths_in_config(obj: Any, base_dir: str) -> Any:
    """Recursively resolve fields that look like paths relative to the config file."""
    if isinstance(obj, dict):
        out: Dict[str, Any] = {}
        for k, v in obj.items():
            out[k] = _resolve_paths_in_config(v, base_dir)
            val = out[k]
            if isinstance(val, str) and _looks_like_path_key(k):
                if val and not os.path.isabs(val):
                    if not val.startswith(("http://", "https://", "ws://", "wss://")):
                        expanded = os.path.expandvars(val)
                        out[k] = os.path.abspath(os.path.join(base_dir, expanded))
        return out
    if isinstance(obj, list):
        return [_resolve_paths_in_config(x, base_dir) for x in obj]
    return obj


def _parse_env_specs(cfg: Dict[str, Any]) -> List[EnvSpec]:
    envs_cfg = cfg.get("envs")
    if not envs_cfg:
        raise ValueError("No envs specified. Provide env definitions under 'envs:'.")

    raw_default_chat_cfg = cfg.get("default_chat_config")
    if raw_default_chat_cfg is None:
        default_chat_cfg: Dict[str, Any] = {}
    elif isinstance(raw_default_chat_cfg, dict):
        default_chat_cfg = raw_default_chat_cfg
    else:
        raise TypeError(
            f"default_chat_config must be a mapping, got {type(raw_default_chat_cfg).__name__}"
        )

    specs: List[EnvSpec] = []
    for item in envs_cfg:
        if not isinstance(item, dict):
            raise TypeError("Each env spec must be a mapping")
        if "tag_id" not in item or item.get("tag_id") is None:
            raise ValueError(f"Env spec '{item.get('name')}' is missing 'tag_id'. Provide a tag_id (int or str).")

        tag_id_val = item.get("tag_id")
        # Keep tag_id as-is (int or str), but convert to str if it's something else
        if not isinstance(tag_id_val, (int, str)):
            tag_id_val = str(tag_id_val)

        # Per-env chat_config takes priority; fall back to top-level default_chat_config
        if "chat_config" in item:
            raw_chat_cfg = item.get("chat_config")
            if raw_chat_cfg is None:
                chat_cfg = {}
            elif isinstance(raw_chat_cfg, dict):
                chat_cfg = raw_chat_cfg
            else:
                raise TypeError(
                    f"env '{item.get('name')}' chat_config must be a mapping, "
                    f"got {type(raw_chat_cfg).__name__}"
                )
        else:
            chat_cfg = copy.deepcopy(default_chat_cfg)

        spec = EnvSpec(
            name=str(item["name"]),
            n_envs=int(item["n_envs"]),
            split=str(item.get("split", "default")),
            tag_id=tag_id_val,
            config=item.get("config") or {},
            chat_config=chat_cfg,
            seed=item.get("seed") if "seed" in item else [0],
            seed_list=item.get("seed_list"),
            max_turns=item.get("max_turns"),
            concat_multi_turn=item.get("concat_multi_turn", True),
        )
        specs.append(spec)
    return specs


def _resolve_dump_dir(cfg: Dict[str, Any], base_dir: str) -> str:
    exp_cfg = cfg.get("experiment") or {}
    dump_dir = exp_cfg.get("dump_dir", "./rollouts")
    if not isinstance(dump_dir, str):
        raise TypeError("experiment.dump_dir must be a string path")
    dump_dir = os.path.expandvars(dump_dir)
    if not os.path.isabs(dump_dir):
        dump_dir = os.path.abspath(os.path.join(base_dir, dump_dir))
    return dump_dir


def _read_json(path: str) -> Optional[Dict[str, Any]]:
    """Best-effort JSON load; returns None on any error."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def _iter_rollout_dirs(dump_dir: str):
    """Yield ``os.DirEntry`` for each ``dump_dir/tag_*/<rollout>`` directory."""
    for tag_entry in os.scandir(dump_dir):
        if not tag_entry.is_dir() or not tag_entry.name.startswith("tag_"):
            continue
        for rollout_entry in os.scandir(tag_entry.path):
            if rollout_entry.is_dir():
                yield rollout_entry


def _rollout_finish_reason(metrics: Dict[str, Any]) -> Optional[str]:
    """Derive finish_reason: explicit field, else terminated+success → 'done'."""
    fr = metrics.get("finish_reason")
    if fr:
        return fr
    if metrics.get("terminated") and metrics.get("success"):
        return "done"
    return None


def _rmtree_logged(path: str, ok_msg: str, fail_msg: str) -> None:
    try:
        shutil.rmtree(path, ignore_errors=False)
        logger.info("%s: %s", ok_msg, path)
    except Exception:
        logger.warning("%s: %s", fail_msg, path)


def _purge_error_rollouts(dump_dir: Optional[str], resume_mode: str) -> None:
    """Remove previous error rollouts so reruns start clean. No-op when resume is off."""
    if resume_mode == "off" or not dump_dir or not os.path.isdir(dump_dir):
        return
    # Read through the runner module so main()'s override is reflected here.
    success_reasons = set(_runner_mod.NORMAL_FINISH_REASONS)
    for rollout in _iter_rollout_dirs(dump_dir):
        metrics_path = os.path.join(rollout.path, "metrics.json")
        if not os.path.isfile(metrics_path):
            _rmtree_logged(rollout.path,
                "Removed rollout without metrics",
                "Failed to remove rollout without metrics")
            continue
        metrics = _read_json(metrics_path)
        if metrics is None:
            continue
        if _rollout_finish_reason(metrics) in success_reasons:
            continue
        _rmtree_logged(rollout.path,
            "Removed previous error rollout folder",
            "Failed to remove error rollout folder")


def _refresh_tag_summaries(dump_dir: Optional[str]) -> None:
    if not dump_dir or not os.path.isdir(dump_dir):
        return
    for tag_entry in os.scandir(dump_dir):
        if not tag_entry.is_dir() or not tag_entry.name.startswith("tag_"):
            continue
        try:
            outp = write_rollouts_summary_from_dump(dump_dir=tag_entry.path, filename="summary.json")
            logger.info("Resume: refreshed summary %s", outp)
        except Exception as exc:
            logger.warning("Resume: failed to refresh summary for %s: %s", tag_entry.path, exc)


def _collect_completed_runs(dump_dir: Optional[str]) -> Dict[Tuple[str, int, Union[int, str]], str]:
    """Find completed (success) runs keyed by (env_name, seed, tag_id)."""
    completed: Dict[Tuple[str, int, Union[int, str]], str] = {}
    if not dump_dir or not os.path.isdir(dump_dir):
        return completed

    for rollout in _iter_rollout_dirs(dump_dir):
        metrics = _read_json(os.path.join(rollout.path, "metrics.json"))
        if metrics is None:
            continue
        if _rollout_finish_reason(metrics) not in _runner_mod.NORMAL_FINISH_REASONS:
            continue
        meta = _read_json(os.path.join(rollout.path, "meta.json")) or {}
        # Prefer meta.json over metrics.json, but fall back only when meta's
        # value is missing/None — NOT when it's a falsy-but-valid 0.
        def _pick(key: str) -> Any:
            v = meta.get(key)
            return v if v is not None else metrics.get(key)
        env_name, seed, tag_id = _pick("env_name"), _pick("seed"), _pick("tag_id")
        if env_name is None or seed is None or tag_id is None:
            continue
        try:
            if not isinstance(tag_id, (int, str)):
                tag_id = str(tag_id)
            completed[(str(env_name), int(seed), tag_id)] = "done"
        except (TypeError, ValueError):
            continue
    return completed


def _job_resume_key(data: Dict[str, Any]) -> Optional[Tuple[str, int, Union[int, str]]]:
    env_name = data.get("env_name")
    seed = data.get("seed")
    tag_id = data.get("tag_id")
    if env_name is None or seed is None or tag_id is None:
        return None
    try:
        # Keep tag_id as original type (int or str)
        if not isinstance(tag_id, (int, str)):
            tag_id = str(tag_id)
        return (str(env_name), int(seed), tag_id)
    except (TypeError, ValueError):
        return None




def _expand_jobs(
    env_specs: List[EnvSpec],
    base_seed: int,
    base_dir: str,
    default_max_turns: Optional[int] = None,
) -> List[Dict[str, Any]]:
    jobs: List[Dict[str, Any]] = []
    for spec_idx, spec in enumerate(env_specs):
        env_cls = get_env_cls(spec.name)
        resolved_config = copy.deepcopy(spec.config)
        seeds = generate_seeds_for_spec(spec, base_seed, spec_idx)
        job_max_turns = int(spec.max_turns if spec.max_turns is not None else default_max_turns or 10)
        env_chat_cfg = spec.chat_config or {}

        for i in range(spec.n_envs):
            seed = seeds[i]
            job_config = copy.deepcopy(resolved_config)
            chat_cfg = copy.deepcopy(env_chat_cfg)
            job_data = {
                "env_cls": env_cls,
                "env_config": job_config,
                "seed": int(seed),
                "tag_id": spec.tag_id,  # Keep original type (int or str)
                "split": spec.split,
                "env_name": spec.name,
                "max_turns": job_max_turns,
                "chat_config": chat_cfg,
                "concat_multi_turn": spec.concat_multi_turn,
            }
            jobs.append({"data": job_data})
    return jobs


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate ViewSuite agents across multiple env specs.")
    parser.add_argument("--config", type=str, default=None, help="Path to evaluation YAML config.")
    parser.add_argument(
        "overrides",
        nargs="*",
        help="Optional OmegaConf dotlist overrides, e.g. run.backend=sglang run.max_concurrent_jobs=8",
    )
    return parser.parse_args()


def _resolve_defaults(cfg_path: str, cfg: DictConfig, _visited: Optional[set] = None) -> DictConfig:
    """
    If the config contains a ``defaults:`` list, load each referenced YAML
    file and deep-merge them in order, then merge the current config on top.

    Paths in ``defaults`` are resolved relative to the directory of
    *cfg_path*.  The ``.yaml`` extension is appended automatically if omitted.

    Example usage inside a YAML config::

        defaults:
          - base_viewsuite        # loads base_viewsuite.yaml next to this file
          - ../shared/backends    # relative path also works

        # only the fields you want to override
        run:
          backend: "claude"
        experiment:
          dump_dir: ${fileroot}/rollouts/claude
    """
    defaults = OmegaConf.select(cfg, "defaults", default=None)
    if not defaults:
        return cfg

    if _visited is None:
        _visited = set()
    abs_cfg_path = os.path.abspath(cfg_path)
    if abs_cfg_path in _visited:
        raise ValueError(f"Cyclic defaults reference detected at: {abs_cfg_path}")
    _visited.add(abs_cfg_path)

    base_dir = os.path.dirname(cfg_path)
    merged = OmegaConf.create()

    for entry in defaults:
        if not isinstance(entry, str):
            raise TypeError(
                f"Each entry in 'defaults' must be a string, got {type(entry).__name__}: "
                f"{entry!r} (in {cfg_path})"
            )
        ref = entry
        if not ref.endswith((".yaml", ".yml")):
            ref += ".yaml"
        ref_path = os.path.normpath(os.path.join(base_dir, ref))
        if not os.path.isfile(ref_path):
            raise FileNotFoundError(f"Default config not found: {ref_path} (referenced from {cfg_path})")
        base_cfg = OmegaConf.load(ref_path)
        # recursively resolve nested defaults
        base_cfg = _resolve_defaults(ref_path, base_cfg, _visited)
        merged = OmegaConf.merge(merged, base_cfg)

    # remove the 'defaults' key itself before merging
    with open_dict(cfg):
        if "defaults" in cfg:
            del cfg["defaults"]

    merged = OmegaConf.merge(merged, cfg)
    return merged


def _load_config(cfg_path: str, overrides: List[str]) -> DictConfig:
    cfg: DictConfig = OmegaConf.load(cfg_path)  # type: ignore
    cfg = _resolve_defaults(cfg_path, cfg)
    if overrides:
        override_cfg = OmegaConf.from_dotlist(overrides)
        cfg = OmegaConf.merge(cfg, override_cfg)
    return cfg


def _run_jobs_multiprocess(
    *,
    jobs: List[Dict[str, Any]],
    backend: str,
    backend_cfg: Dict[str, Any],
    model: str,
    default_max_turns: Optional[int],
    dump_dir: Optional[str],
    max_concurrent_jobs: int,
    resume_mode: str,
    live_summary: bool,
    num_workers: int,
    normal_finish_reasons: Optional[List[str]],
) -> List[Dict[str, Any]]:
    """Fan ``jobs`` out to ``num_workers`` subprocess workers sharing one backend.

    Global budgets (``max_concurrent_jobs``, ``backend_cfg.max_concurrency``)
    are ceil-divided across workers so total backend pressure stays at the
    configured limit. Each worker crashes independently — failed chunks
    surface as one structured error record per job, not as a batch abort.
    """
    from concurrent.futures import ProcessPoolExecutor, as_completed
    import multiprocessing as mp

    chunks = split_jobs_round_robin(jobs, num_workers)
    if not chunks:
        return []
    n_actual = len(chunks)

    # Ceil-split the global budgets; ``max(1, …)`` keeps tiny caps usable.
    per_worker_max_jobs = max(1, math.ceil(max_concurrent_jobs / n_actual))
    global_backend_conc = int(backend_cfg.get("max_concurrency", 2))
    per_worker_backend_conc = max(1, math.ceil(global_backend_conc / n_actual))
    per_worker_backend_cfg = {**backend_cfg, "max_concurrency": per_worker_backend_conc}

    payload_base = dict(
        backend=backend, backend_cfg=per_worker_backend_cfg, model=model,
        default_max_turns=default_max_turns, dump_dir=dump_dir,
        max_concurrent_jobs=per_worker_max_jobs, resume_mode=resume_mode,
        live_summary=live_summary, normal_finish_reasons=normal_finish_reasons,
    )
    worker_payloads = [{"jobs": chunk, **payload_base} for chunk in chunks]

    logger.info(
        "Multi-process rollout: %d worker(s), %d jobs (~%d/worker); "
        "max_concurrent_jobs %d→%d/worker, backend max_concurrency %d→%d/worker",
        n_actual, len(jobs), len(jobs) // n_actual,
        max_concurrent_jobs, per_worker_max_jobs,
        global_backend_conc, per_worker_backend_conc,
    )

    # spawn: clean child state, portable, no fork-after-CUDA breakage.
    ctx = mp.get_context("spawn")
    results: List[Dict[str, Any]] = []
    with ProcessPoolExecutor(max_workers=n_actual, mp_context=ctx) as pool:
        fut_to_idx = {pool.submit(run_eval_chunk_subprocess, p): i
                      for i, p in enumerate(worker_payloads)}
        for fut in as_completed(fut_to_idx):
            idx = fut_to_idx[fut]
            try:
                results.extend(fut.result())
            except (KeyboardInterrupt, SystemExit):
                raise
            except BaseException as exc:  # noqa: BLE001 — incl. BrokenProcessPool
                logger.exception("Worker %d crashed: %s", idx, exc)
                # One error record per job in the crashed chunk so summaries
                # still account for them.
                for job in worker_payloads[idx]["jobs"]:
                    data = job.get("data", {})
                    rec = {
                        "rollout_id": f"WORKER-ERR-{idx}-{uuid.uuid4().hex[:8]}",
                        "error": f"worker {idx} crashed: {exc!r}",
                    }
                    for k in ("env_name", "split", "tag_id", "seed"):
                        if data.get(k) is not None:
                            rec[k] = data[k]
                    results.append(rec)
    return results


def main() -> None:
    args = _parse_args()
    cfg_path = args.config or DEFAULT_CONFIG_PATH
    cfg_path = os.path.abspath(cfg_path)
    if not os.path.exists(cfg_path):
        raise FileNotFoundError(f"Config file not found: {cfg_path}")

    cfg_node = _load_config(cfg_path, args.overrides)
    print("=== Effective Config ===")
    print(OmegaConf.to_yaml(cfg_node, resolve=True))

    cfg: Dict[str, Any] = OmegaConf.to_container(cfg_node, resolve=True)  # type: ignore
    base_dir = os.path.dirname(cfg_path)
    cfg = _resolve_paths_in_config(cfg, base_dir)

    run_cfg = cfg.get("run") or {}
    backend = str(run_cfg.get("backend", "openai")).lower()
    resume_mode = str(run_cfg.get("resume", "skip_completed"))
    live_summary = bool(run_cfg.get("live_summary", False))
    max_concurrent = int(run_cfg.get("max_concurrent_jobs", 4))
    base_seed = int(run_cfg.get("base_seed", run_cfg.get("start_seed", 0)))
    # num_workers > 1 → opt-in multi-process mode: round-robin partition the
    # jobs across ``ProcessPoolExecutor`` workers sharing the same backend.
    # Use this when single-process CPU (PIL, JSON, regex) is the bottleneck.
    num_workers = int(run_cfg.get("num_workers", 1))
    # Override what counts as a "successfully completed" run. The reasoning-
    # augmentation pipeline narrows this to {"done"} so max_turns exits are
    # retried instead of skipped on resume. Rebound on runner so both the
    # parent (via ``_runner_mod``) and child workers (via payload) see it.
    nfr_override = run_cfg.get("normal_finish_reasons")
    if nfr_override is not None:
        _runner_mod.NORMAL_FINISH_REASONS = set(nfr_override)

    backend_cfg: Dict[str, Any] = cfg.get("backends", {})[backend]
    model = backend_cfg.get("model") or backend_cfg.get("deployment")
    if not model:
        raise ValueError(f"[{backend}] requires 'model' (or 'deployment' for Azure) in backends.{backend}.*")

    env_specs = _parse_env_specs(cfg)
    default_max_turns = (cfg.get("experiment") or {}).get("default_max_turns")
    jobs = _expand_jobs(env_specs, base_seed, base_dir, default_max_turns)
    print(f"Prepared {len(jobs)} jobs from {len(env_specs)} environment specs.")

    dump_dir = _resolve_dump_dir(cfg, base_dir)
    if resume_mode != "off":
        logger.info("Resume mode=%s; pruning error rollouts under %s", resume_mode, dump_dir)
        _purge_error_rollouts(dump_dir, resume_mode)
        _refresh_tag_summaries(dump_dir)

    completed_index: Dict[Tuple[str, int, int], str] = {}
    if resume_mode == "skip_completed":
        completed_index = _collect_completed_runs(dump_dir)
        logger.info("Resume: detected %d completed rollouts to skip", len(completed_index))

    if completed_index:
        pending_jobs = []
        skipped = 0
        for job in jobs:
            key = _job_resume_key(job["data"])
            if key and completed_index.get(key) == "done":
                skipped += 1
                data = job["data"]
                logger.info(
                    "Skipping completed rollout env=%s tag=%s seed=%s",
                    data.get("env_name"),
                    data.get("tag_id"),
                    data.get("seed"),
                )
                continue
            pending_jobs.append(job)
        if skipped:
            logger.info("Resume: skipped %d/%d jobs", skipped, len(jobs))
        jobs = pending_jobs
    logger.info("Total pending jobs: %d", len(jobs))

    if num_workers <= 1 or len(jobs) <= 1:
        # Default in-process path — identical to the pre-multi-worker behaviour.
        results = asyncio.run(
            run_eval_parallel(
                jobs,
                backend=backend,
                backend_cfg=backend_cfg,
                model=model,
                default_max_turns=default_max_turns,
                dump_dir=dump_dir,
                max_concurrent_jobs=max_concurrent,
                resume_mode=resume_mode,
                live_summary=live_summary,
            )
        )
    else:
        results = _run_jobs_multiprocess(
            jobs=jobs,
            backend=backend,
            backend_cfg=backend_cfg,
            model=model,
            default_max_turns=default_max_turns,
            dump_dir=dump_dir,
            max_concurrent_jobs=max_concurrent,
            resume_mode=resume_mode,
            live_summary=live_summary,
            num_workers=num_workers,
            normal_finish_reasons=(
                list(nfr_override) if nfr_override is not None else None
            ),
        )

    error_records_by_tag: Dict[Union[int, str], List[Dict[str, Any]]] = {}
    tag_ids_seen: set[Union[int, str]] = set()
    for r in results:
        rid = r.get("rollout_id")
        finish_reason = r.get("finish_reason") or r.get("skipped") or ""
        tag_id_val = r.get("tag_id")
        tag_info = ""
        if tag_id_val is not None:
            tag_info = f"(tag={tag_id_val})"
            tag_ids_seen.add(tag_id_val)
        error_msg = r.get("error")
        if error_msg:
            print(f"{rid} ERROR: {error_msg} {tag_info}")
            detail: Dict[str, Any] = {"rollout_id": rid, "error": error_msg}
            for key in ("tag_id", "env_name", "split", "seed"):
                if key in r and r.get(key) is not None:
                    detail[key] = r.get(key)
            if tag_id_val is not None:
                error_records_by_tag.setdefault(tag_id_val, []).append(detail)
        else:
            print(rid, finish_reason, tag_info)

    from vagen.evaluate.utils.summary_utils import write_rollouts_summary_from_dump

    # Sort tag_ids with str(x) as key to handle both int and str
    for tag_id in sorted(tag_ids_seen, key=str):
        tag_dir = os.path.join(dump_dir, f"tag_{tag_id}") if dump_dir else None
        if not tag_dir:
            continue
        outp = write_rollouts_summary_from_dump(dump_dir=tag_dir, filename="summary.json")
        tag_errors = error_records_by_tag.get(tag_id)
        if tag_errors:
            try:
                with open(outp, "r", encoding="utf-8") as f:
                    summary_payload = json.load(f)
            except Exception:
                summary_payload = {"created_at": None}
            summary_payload["error_details"] = tag_errors
            with open(outp, "w", encoding="utf-8") as f:
                json.dump(summary_payload, f, ensure_ascii=False, indent=2)
            print(f"[Error details appended] {outp}")
        print(f"[Summary written] {outp}")


if __name__ == "__main__":
    main()
