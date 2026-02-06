# All comments are in English.
from __future__ import annotations

import argparse
import asyncio
import copy
import json
import logging
import os
import shutil
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

from omegaconf import DictConfig, OmegaConf

from vagen.evaluate.register_builtins import *  # populate registry
from vagen.envs.registry import get_env_cls
from vagen.evaluate.runner import run_eval_parallel, NORMAL_FINISH_REASONS
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

        spec = EnvSpec(
            name=str(item["name"]),
            n_envs=int(item["n_envs"]),
            split=str(item.get("split", "default")),
            tag_id=tag_id_val,
            config=item.get("config") or {},
            chat_config=item.get("chat_config") or {},
            seed=item.get("seed") if "seed" in item else [0],
            seed_list=item.get("seed_list"),
            max_turns=item.get("max_turns"),
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


def _purge_error_rollouts(dump_dir: Optional[str], resume_mode: str) -> None:
    """
    Remove previous error rollouts so reruns start clean.
    Only invoked when resume mode keeps completed runs.
    """
    if resume_mode == "off" or not dump_dir:
        return
    if not os.path.isdir(dump_dir):
        return

    success_reasons = set(NORMAL_FINISH_REASONS)
    for tag_entry in os.scandir(dump_dir):
        if not tag_entry.is_dir() or not tag_entry.name.startswith("tag_"):
            continue
        for rollout_entry in os.scandir(tag_entry.path):
            if not rollout_entry.is_dir():
                continue
            metrics_path = os.path.join(rollout_entry.path, "metrics.json")
            if not os.path.isfile(metrics_path):
                try:
                    shutil.rmtree(rollout_entry.path, ignore_errors=False)
                    logger.info("Removed rollout without metrics: %s", rollout_entry.path)
                except Exception:
                    logger.warning("Failed to remove rollout without metrics: %s", rollout_entry.path)
                continue
            try:
                with open(metrics_path, "r", encoding="utf-8") as f:
                    metrics = json.load(f)
            except Exception:
                continue

            finish_reason = metrics.get("finish_reason")
            if not finish_reason:
                terminated = bool(metrics.get("terminated"))
                success = bool(metrics.get("success"))
                if terminated and success:
                    finish_reason = "done"

            if finish_reason in success_reasons:
                continue

            try:
                shutil.rmtree(rollout_entry.path, ignore_errors=False)
                logger.info("Removed previous error rollout folder: %s", rollout_entry.path)
            except Exception:
                logger.warning("Failed to remove error rollout folder: %s", rollout_entry.path)


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
    """
    Scan existing rollouts to find completed (success) runs keyed by (env_name, seed, tag_id).
    """
    completed: Dict[Tuple[str, int, Union[int, str]], str] = {}
    if not dump_dir or not os.path.isdir(dump_dir):
        return completed

    for tag_entry in os.scandir(dump_dir):
        if not tag_entry.is_dir() or not tag_entry.name.startswith("tag_"):
            continue
        for rollout_entry in os.scandir(tag_entry.path):
            if not rollout_entry.is_dir():
                continue
            metrics_path = os.path.join(rollout_entry.path, "metrics.json")
            if not os.path.isfile(metrics_path):
                continue
            try:
                with open(metrics_path, "r", encoding="utf-8") as f:
                    metrics = json.load(f)
            except Exception:
                continue

            finish_reason = metrics.get("finish_reason")
            if not finish_reason:
                terminated = bool(metrics.get("terminated"))
                success = bool(metrics.get("success"))
                if terminated and success:
                    finish_reason = "done"

            if finish_reason not in NORMAL_FINISH_REASONS:
                continue

            meta_path = os.path.join(rollout_entry.path, "meta.json")
            meta_payload: Optional[Dict[str, Any]] = None
            if os.path.isfile(meta_path):
                try:
                    with open(meta_path, "r", encoding="utf-8") as f:
                        meta_payload = json.load(f)
                except Exception:
                    meta_payload = None

            env_name = (meta_payload or {}).get("env_name") or metrics.get("env_name")
            seed = (meta_payload or {}).get("seed") or metrics.get("seed")
            tag_id = (meta_payload or {}).get("tag_id") or metrics.get("tag_id")
            if env_name is None or seed is None or tag_id is None:
                continue
            try:
                # Keep tag_id as original type (int or str)
                if not isinstance(tag_id, (int, str)):
                    tag_id = str(tag_id)
                key = (str(env_name), int(seed), tag_id)
            except (TypeError, ValueError):
                continue
            completed[key] = "done"
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
) -> List[Dict[str, Any]]:
    jobs: List[Dict[str, Any]] = []
    for spec_idx, spec in enumerate(env_specs):
        env_cls = get_env_cls(spec.name)
        resolved_config = _resolve_paths_in_config(copy.deepcopy(spec.config), base_dir)
        seeds = generate_seeds_for_spec(spec, base_seed, spec_idx)
        job_max_turns = int(spec.max_turns)
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


def _load_config(cfg_path: str, overrides: List[str]) -> DictConfig:
    cfg: DictConfig = OmegaConf.load(cfg_path)  # type: ignore
    if overrides:
        override_cfg = OmegaConf.from_dotlist(overrides)
        cfg = OmegaConf.merge(cfg, override_cfg)
    return cfg


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

    run_cfg = cfg.get("run") or {}
    backend = str(run_cfg.get("backend", "openai")).lower()
    resume_mode = str(run_cfg.get("resume", "skip_completed"))
    live_summary = bool(run_cfg.get("live_summary", False))
    max_concurrent = int(run_cfg.get("max_concurrent_jobs", 4))
    base_seed = int(run_cfg.get("base_seed", run_cfg.get("start_seed", 0)))

    backend_cfg: Dict[str, Any] = OmegaConf.to_container(cfg_node.backends[backend], resolve=True)  # type: ignore
    model = backend_cfg.get("model") or backend_cfg.get("deployment")
    if not model:
        raise ValueError(f"[{backend}] requires 'model' (or 'deployment' for Azure) in backends.{backend}.*")

    env_specs = _parse_env_specs(cfg)
    default_max_turns = (cfg.get("experiment") or {}).get("default_max_turns")
    jobs = _expand_jobs(env_specs, base_seed, base_dir)
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
