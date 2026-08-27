from __future__ import annotations

import argparse
import asyncio
import copy
import json
import logging
import os
import re
import shutil
from dataclasses import dataclass, field, fields
from typing import Any, Dict, List, Optional, Tuple, Union

import datetime
import yaml
from omegaconf import DictConfig, OmegaConf, open_dict

from vagen.envs import get_env_cls
from vagen.evaluation.backends import REGISTRY  # noqa: F401  populate registry
from vagen.evaluation.runner import NORMAL_FINISH_REASONS, run_eval_parallel
from vagen.evaluation.seeding import generate_seeds_for_spec
from vagen.evaluation.summary import write_rollouts_summary_from_dump
#: There is no default config, and there never was one: this pointed at
#: `vagen/evaluation/conf/evaluate.yaml`, a directory that has never existed in the repo, so
#: `python -m vagen.evaluation` with no --config raised FileNotFoundError on a path
#: the user had no way to recognise as fictional. A config names the environments to run;
#: there is nothing sensible to guess.
_EXAMPLE_CONFIGS_DIR = "examples/evaluate"


logger = logging.getLogger("vagen.evaluation.cli")
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
    # ★ The context policy, by name, exactly as training spells it in trainer.harness:
    # concat | no_concat | compact. This replaces `concat_multi_turn`, a bool that could
    # express concat and approximately no_concat and could not express compaction at all.
    # Training deleted that bool rather than deprecating it so a stale override would be
    # rejected instead of quietly outvoting the harness; eval was the one place it lived on.
    harness: str = "concat"
    # Optional, and off by default: evaluation builds no training row, so there is no
    # region a conversation has to fit. Set response_length_per_turn to bound a turn the
    # same way training does -- it becomes the API call's max_tokens -- and the rest
    # follows from it.
    response_length_per_turn: Optional[int] = None
    #: The response region a conversation must fit, as data.max_response_length is
    #: in training. Optional: unset, there is no accounting, which is what a closed
    #: API wants. Set it and compaction and the room checks work as they do in
    #: training. Do NOT expect response_length_per_turn * max_turns to stand in for
    #: it -- observations land in the same region.
    max_response_length: Optional[int] = None
    max_env_response_per_turn: Optional[int] = None
    compact_budget: Optional[int] = None
    compact_summary_budget: Optional[int] = None
    #: Only used when sizes have to be estimated (no tokenizer). It feeds the compaction
    #: trigger as well as the overflow guard -- see chat_client.DEFAULT_TOKENS_PER_IMAGE.
    tokens_per_image: Optional[int] = None
    #: A HuggingFace tokenizer id or path. Given one, text sizes are exact instead of
    #: estimated at 4 characters a token, which is what the compaction trigger runs on.
    #: Images are still estimated -- a tokenizer cannot price a frame; that is
    #: tokens_per_image's job. Leave unset for a closed API, where there is nothing to load.
    tokenizer: Optional[str] = None


#: Every key an env entry may set. Anything else is a typo or a setting that moved, and
#: both are worth an error -- see _parse_env_specs.
_ENV_SPEC_KEYS = {f.name for f in fields(EnvSpec)}


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
        # ★ Refuse a key nobody reads. This function copies a fixed list out of each env
        # entry and used to drop everything else without a word, so a `harness: compact`
        # in an eval config was accepted, ignored, and ran concat -- and a misspelled key
        # looked exactly like a working one. The failure it replaces is silent and the
        # error is cheap.
        unknown = set(item) - _ENV_SPEC_KEYS
        if unknown:
            raise ValueError(
                f"env entry {item.get('name')!r} sets {sorted(unknown)}, which nothing "
                f"reads. Known keys: {sorted(_ENV_SPEC_KEYS)}."
            )
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
            harness=str(item.get("harness", "concat")),
            response_length_per_turn=item.get("response_length_per_turn"),
            max_response_length=item.get("max_response_length"),
            max_env_response_per_turn=item.get("max_env_response_per_turn"),
            compact_budget=item.get("compact_budget"),
            compact_summary_budget=item.get("compact_summary_budget"),
            tokens_per_image=item.get("tokens_per_image"),
            tokenizer=item.get("tokenizer"),
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


#: What `resume:` may say. "off" keeps everything, including previous error rollouts.
_RESUME_MODES = {"off", "skip_completed", "force_rerun"}


def _resume_mode(value) -> str:
    """Normalise `resume:`, and refuse a value nothing understands.

    ★ YAML parses a bare ``off`` as the boolean False, so ``resume: off`` used to
    stringify to ``"False"``, match none of the ``== "off"`` guards, and quietly do the
    opposite of what it said -- error rollouts were purged on a run that asked for nothing
    to be touched. Both spellings are accepted now, and anything else is an error rather
    than a silent fallthrough to whichever branch happens to be last.
    """
    if value is None:
        # `resume:` with nothing after it, or `run.resume=` on the CLI. Previously this
        # fell through to force_rerun behaviour rather than stopping the run.
        return "skip_completed"
    if isinstance(value, bool):
        return "off" if value is False else "skip_completed"
    text = str(value).strip().lower()
    if text in _RESUME_MODES:
        return text
    raise ValueError(
        f"run.resume={value!r} is not a resume mode; choose from {sorted(_RESUME_MODES)}. "
        f"Note YAML reads a bare `off` as the boolean False -- quote it as \"off\" if you "
        f"want the literal string."
    )


def _purge_error_rollouts(dump_dir: Optional[str], resume_mode: str, *, tags: set) -> None:
    """
    Remove previous error rollouts so reruns start clean.
    Only invoked when resume mode keeps completed runs.

    ``tags`` is required rather than defaulting to None: unscoped means "every tag in the
    directory", and navigation puts three tags in one dump dir, so a caller that forgets it
    discards results nobody asked to rerun.
    """
    if resume_mode == "off" or not dump_dir:
        return
    if not os.path.isdir(dump_dir):
        return

    success_reasons = set(NORMAL_FINISH_REASONS)
    for tag_entry in os.scandir(dump_dir):
        if not tag_entry.is_dir() or not tag_entry.name.startswith("tag_"):
            continue
        if tags is not None and tag_entry.name not in tags:
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


def _refresh_tag_summaries(dump_dir: Optional[str], *, model: Optional[str] = None,
                           tags: set) -> None:
    """Recompute each tag's summary.json for ``model``, after error rollouts were purged.

    ``tags`` is required rather than defaulting to None: unscoped means "every tag in the
    directory", which for this function is the destructive reading -- see below.
    """
    if not dump_dir or not os.path.isdir(dump_dir):
        return
    for tag_entry in os.scandir(dump_dir):
        if not tag_entry.is_dir() or not tag_entry.name.startswith("tag_"):
            continue
        if tag_entry.name not in tags:
            continue
        # ★ This runs at startup, BEFORE any of this run's rollouts exist, and the summary
        # is filtered to `model`. So evaluating checkpoint B into the directory checkpoint A
        # used rewrites A's summary.json as `n_episodes: 0, success_rate: 0.0` -- and the
        # per-tag rewrite that would have fixed it only happens if B's run reaches the end.
        # A bad api_key or a Ctrl-C then leaves A's results replaced by a zero. Keep the old
        # file unless the new one actually has episodes behind it.
        summary_path = os.path.join(tag_entry.path, "summary.json")
        previous: Optional[bytes] = None
        if os.path.isfile(summary_path):
            try:
                with open(summary_path, "rb") as f:
                    previous = f.read()
            except OSError:
                previous = None
        try:
            # model=... so a dump directory holding two checkpoints does not get a summary
            # averaged across both. Without it, a resume that skips every job (because the
            # run is already complete) republishes the blended number and the per-tag
            # rewrite below never runs, since `results` is empty.
            outp = write_rollouts_summary_from_dump(dump_dir=tag_entry.path,
                                                    filename="summary.json", model=model)
            if previous is not None and _summary_is_empty(outp):
                with open(summary_path, "wb") as f:
                    f.write(previous)
                logger.info("Resume: %s has no rollouts for this model yet; kept the "
                            "existing summary rather than zeroing it", tag_entry.path)
            else:
                logger.info("Resume: refreshed summary %s", outp)
        except Exception as exc:
            logger.warning("Resume: failed to refresh summary for %s: %s", tag_entry.path, exc)


def _summary_is_empty(path: Optional[str]) -> bool:
    """True when a summary was written with nothing behind it. Unreadable counts as empty
    only if it is also missing -- a summary we cannot parse is not one we should overwrite."""
    if not path or not os.path.isfile(path):
        return True
    try:
        with open(path, "r", encoding="utf-8") as f:
            return int(json.load(f).get("n_episodes") or 0) <= 0
    except Exception:
        return False


def _collect_completed_runs(
    dump_dir: Optional[str],
) -> Dict[Tuple[str, int, Union[int, str], str], str]:
    """
    Scan existing rollouts to find completed (success) runs keyed by
    (env_name, seed, tag_id, model) -- the same shape `_job_resume_key` builds.
    """
    completed: Dict[Tuple[str, int, Union[int, str], str], str] = {}
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
                key = (str(env_name), int(seed), tag_id, str(metrics.get("model") or ""))
            except (TypeError, ValueError):
                continue
            completed[key] = "done"
    return completed


def _job_resume_key(data: Dict[str, Any]) -> Optional[Tuple]:
    """★ The model is part of the key. Without it, evaluating a second checkpoint into a
    dump directory the first one used skipped every episode and reprinted the first
    model's success_rate under the second model's name -- exit 0, nothing in the output
    saying so. A rollout produced by a different model answers a different question."""
    env_name = data.get("env_name")
    seed = data.get("seed")
    tag_id = data.get("tag_id")
    if env_name is None or seed is None or tag_id is None:
        return None
    try:
        # Keep tag_id as original type (int or str)
        if not isinstance(tag_id, (int, str)):
            tag_id = str(tag_id)
        return (str(env_name), int(seed), tag_id, str(data.get("resume_model") or ""))
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
                "harness": spec.harness,
                "response_length_per_turn": spec.response_length_per_turn,
                "max_response_length": spec.max_response_length,
                "max_env_response_per_turn": spec.max_env_response_per_turn,
                "compact_budget": spec.compact_budget,
                "compact_summary_budget": spec.compact_summary_budget,
                "tokens_per_image": spec.tokens_per_image,
                "tokenizer": spec.tokenizer,
            }
            jobs.append({"data": job_data})
    return jobs


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate VAGEN agents across multiple env specs.")
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
          - base_sokoban          # loads base_sokoban.yaml next to this file
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
        # ★ Applied one at a time with OmegaConf.update, not merged from a dotlist.
        # `from_dotlist` turns `envs.0.n_envs=1` into a DICT keyed "0", and merging that
        # onto a list raises `Cannot merge DictConfig with ListConfig` -- so no override
        # could reach an element of `envs`, which is where every per-environment setting
        # lives. `update` understands the index.
        for item in overrides:
            if "=" not in item:
                raise ValueError(f"override {item!r} is not key=value")
            key, _, raw = item.partition("=")
            # Parsed as YAML so lists, ints and booleans arrive as themselves rather than
            # as strings: `envs.0.seed=[1,60,1]` has to be a list.
            try:
                value = yaml.safe_load(raw)
            except yaml.YAMLError as exc:
                raise ValueError(f"override {item!r}: {raw!r} is not valid YAML") from exc
            # ★ A YAML timestamp is not a config value. `2024-12-01` loads as a date, which
            # OmegaConf refuses as an unsupported type -- and it is a real API version
            # string (backends.azure.azure_api_version). Dates have no other use here.
            if isinstance(value, (datetime.date, datetime.datetime)):
                value = raw
            key = key.strip()
            # ★ A key that does not already exist is almost always a typo, and OmegaConf
            # would create it: `run.backendd=vllm` and `experiment.dumpdir=/tmp/x` were
            # both accepted in silence, the run went ahead with the real setting, and it
            # exited 0. Only `envs[]` entries were validated. Hydra's own convention for
            # deliberately adding a key is a `+` prefix, so use that here too.
            if key.startswith("+"):
                key = key[1:]
            elif _may_be_absent(key):
                pass
            elif OmegaConf.select(cfg, key, default=_ABSENT) is _ABSENT:
                raise ValueError(
                    f"override {item!r} names {key!r}, which is not in the config. "
                    f"{_did_you_mean(cfg, key)}To add a key that is genuinely new, prefix "
                    f"it: +{item}"
                )
            OmegaConf.update(cfg, key, value, merge=True)
    return cfg


#: Distinct from None, which is a legitimate value for several keys (backends.*.base_url).
_ABSENT = object()

#: `envs.<i>.<field>` for a field EnvSpec defines but the yaml leaves at its default. Most
#: env entries do not spell out `harness`, and `envs.0.harness=no_concat` is the documented
#: way to evaluate a checkpoint under another context policy -- rejecting it for being
#: absent would refuse the override this validation is least entitled to refuse. A wrong
#: field name here is still caught, by _parse_env_specs, with a better message.
_ENV_OVERRIDE = re.compile(r"^envs\.\d+\.(?P<field>[A-Za-z_][A-Za-z0-9_]*)(?P<rest>\..*)?$")


def _may_be_absent(key: str) -> bool:
    m = _ENV_OVERRIDE.match(key)
    if not m:
        return False
    field = m.group("field")
    # Anything below `config` or `chat_config` is passed through to the environment or the
    # client, so this module has no list to check it against.
    if m.group("rest"):
        return field in ("config", "chat_config")
    return field in _ENV_SPEC_KEYS


def _did_you_mean(cfg: DictConfig, key: str) -> str:
    """The siblings of the deepest path segment that does resolve, if that narrows it."""
    import difflib

    parent, _, leaf = key.rpartition(".")
    m = _ENV_OVERRIDE.match(key)
    if m and not m.group("rest"):
        # The yaml only spells out the env fields it changes, so its own keys are a worse
        # suggestion list than the dataclass's: `envs.0.harnes` should point at `harness`
        # whether or not this config happens to set it.
        candidates = sorted(_ENV_SPEC_KEYS)
    else:
        node = OmegaConf.select(cfg, parent, default=None) if parent else cfg
        if not isinstance(node, DictConfig):
            return ""
        candidates = [str(k) for k in node.keys()]
    close = difflib.get_close_matches(leaf, candidates, n=3, cutoff=0.6)
    if not close:
        return ""
    where = f"{parent}." if parent else ""
    return "Did you mean " + " or ".join(f"{where}{c}" for c in close) + "? "


def main() -> None:
    args = _parse_args()
    if not args.config:
        raise SystemExit(
            "--config is required: an eval config names the environments to run, and there "
            f"is nothing to guess. The shipped ones are under {_EXAMPLE_CONFIGS_DIR}/<env>/, "
            "e.g.\n"
            "  python -m vagen.evaluation --config examples/evaluate/sokoban/config.yaml"
        )
    cfg_path = os.path.abspath(args.config)
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
    resume_mode = _resume_mode(run_cfg.get("resume", "skip_completed"))
    live_summary = bool(run_cfg.get("live_summary", False))
    max_concurrent = int(run_cfg.get("max_concurrent_jobs", 4))
    base_seed = int(run_cfg.get("base_seed", run_cfg.get("start_seed", 0)))

    backends_cfg: Dict[str, Any] = cfg.get("backends", {}) or {}
    if backend not in backends_cfg:
        # A bare KeyError here names the string and nothing else, and the two most likely
        # causes -- a typo, and a backend the README lists but eval_default.yaml does not
        # define -- both look identical from the traceback.
        raise ValueError(
            f"run.backend={backend!r} has no entry under `backends:`. Configured: "
            f"{sorted(backends_cfg)}. Add a `backends.{backend}:` block, or pick one of those."
        )
    backend_cfg: Dict[str, Any] = backends_cfg[backend]
    model = backend_cfg.get("model") or backend_cfg.get("deployment")
    if not model:
        raise ValueError(f"[{backend}] requires 'model' (or 'deployment' for Azure) in backends.{backend}.*")

    env_specs = _parse_env_specs(cfg)
    default_max_turns = (cfg.get("experiment") or {}).get("default_max_turns")
    jobs = _expand_jobs(env_specs, base_seed, base_dir, default_max_turns)
    print(f"Prepared {len(jobs)} jobs from {len(env_specs)} environment specs.")

    dump_dir = _resolve_dump_dir(cfg, base_dir)
    if resume_mode == "force_rerun" and dump_dir and os.path.isdir(dump_dir):
        # ★ Clear the old rollouts, or they are summarised alongside the new ones. Nothing
        # keys a rollout directory to a seed -- they are {timestamp}-{uuid8} -- and
        # write_rollouts_summary_from_dump scans every directory holding a metrics.json.
        # Measured: a 2-episode force_rerun reported n_episodes=4 with seeds [1,1,2,2] and
        # a success_rate averaged over both runs, indistinguishable from a clean one.
        # ★ Only the tags this run will write. Clearing every `tag_*` destroys the
        # completed rollouts of tags that are merely configured elsewhere -- navigation
        # puts three in one dump dir, so re-running a trimmed config wiped the other two.
        wanted = {f"tag_{j['data'].get('tag_id')}" for j in jobs}
        for tag in sorted(wanted):
            path = os.path.join(dump_dir, tag)
            if os.path.isdir(path):
                logger.info("force_rerun: clearing previous rollouts under %s", path)
                try:
                    shutil.rmtree(path)
                except OSError as exc:      # a concurrent run holding the same dir
                    logger.warning("could not clear %s: %s", path, exc)
    if resume_mode != "off":
        logger.info("Resume mode=%s; pruning error rollouts under %s", resume_mode, dump_dir)
        # Scoped to this run's tags, like the force_rerun clearing above: navigation puts
        # three tags in one dump dir, and purging a tag this run does not touch discards
        # results nobody asked to rerun.
        _purge_error_rollouts(dump_dir, resume_mode,
                              tags={f"tag_{j['data'].get('tag_id')}" for j in jobs})
        _refresh_tag_summaries(dump_dir, model=model,
                               tags={f"tag_{j['data'].get('tag_id')}" for j in jobs})

    completed_index: Dict[Tuple[str, int, Union[int, str], str], str] = {}
    # ★ Resume compares the model too. See metrics.json's "model": a rollout produced by a
    # different checkpoint answers a different question, and reusing it is silent.
    if resume_mode == "skip_completed":
        completed_index = _collect_completed_runs(dump_dir)
        logger.info("Resume: detected %d completed rollouts to skip", len(completed_index))

    if completed_index:
        pending_jobs = []
        skipped = 0
        for job in jobs:
            job["data"]["resume_model"] = str(model or "")
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
        # `error_details` is what arun_episode's normal return path sets; `error` is what
        # the runner's setup guard sets. Reading only the latter meant an episode that
        # failed inside the loop was printed as an ordinary status line and never reached
        # summary["error_details"].
        error_msg = r.get("error") or (r.get("error_details") or {}).get("error")
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

    # ★ A config error hits every job identically, and per-job error handling then turns
    # a loud crash into a clean-looking run: exit 0, n_episodes 0, and if the dump dir has
    # prior rollouts the summary reprints THOSE numbers under this run's name. Say it, and
    # exit non-zero.
    # ★ Any total failure, not just one before the episode. A bad api_key, a wrong model
    # id or an unreachable base_url fails INSIDE each episode -- finish_reason "error" --
    # and used to exit 0 reporting success_rate 0.0, which is indistinguishable from a
    # model that simply solved nothing.
    DEAD = {"setup_error", "error", "env_error", "empty_generation"}
    dead = [r for r in results if r.get("finish_reason") in DEAD]
    if dead and len(dead) == len(results):
        first = dead[0].get("error") or dead[0].get("error_details") or dead[0].get("finish_reason")
        raise SystemExit(
            f"all {len(results)} episodes failed ({dead[0].get('finish_reason')}), so "
            f"nothing was evaluated -- this is not a score of zero. First failure: {first}"
        )

    from vagen.evaluation.summary import write_rollouts_summary_from_dump

    # Sort tag_ids with str(x) as key to handle both int and str
    for tag_id in sorted(tag_ids_seen, key=str):
        tag_dir = os.path.join(dump_dir, f"tag_{tag_id}") if dump_dir else None
        if not tag_dir:
            continue
        outp = write_rollouts_summary_from_dump(dump_dir=tag_dir, filename="summary.json",
                                                model=model)
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
