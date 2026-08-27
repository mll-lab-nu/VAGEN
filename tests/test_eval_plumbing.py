"""The eval path end to end, through `run_eval_parallel`.

Mutation testing showed this stretch was entirely unguarded: dropping `"harness"` from
`_expand_jobs` and hardcoding `concat` in the runner both left the suite green. The tests
that named those bugs stopped at `_parse_env_specs`, three hops earlier, or asserted on
`inspect.getsource`.

So these drive the real runner with a stub backend and assert on what came out.
"""

from __future__ import annotations

import asyncio
import json
import os

import pytest

from vagen.evaluation.backends import REGISTRY
from vagen.evaluation.runner import run_eval_parallel

BACKEND = "stub_for_tests"


class _StubAdapter:
    """Records every call's messages, per adapter instance."""

    seen: list = []

    def __init__(self, **kw):
        self.model = kw.get("model")

    def format_system(self, text, images):
        return {"role": "system", "content": [{"type": "text", "text": text}]}

    def format_user_turn(self, text, images):
        return {"role": "user", "content": [{"type": "text", "text": text}]}

    def format_assistant_turn(self, text):
        return {"role": "assistant", "content": [{"type": "text", "text": text}]}

    async def acompletion(self, messages, **cfg):
        _StubAdapter.seen.append([m["role"] for m in messages])
        return "<think>go</think><answer>Right</answer>"


class _Env:
    def __init__(self, env_config):
        self.i = 0

    async def reset(self, seed=None):
        return {"obs_str": "observation 0"}, {}

    async def system_prompt(self):
        return {"obs_str": "you are a solver"}

    async def step(self, action, **kw):
        self.i += 1
        return {"obs_str": f"observation {self.i}"}, 1.0, False, {}

    async def close(self):
        pass


@pytest.fixture(autouse=True)
def _stub_backend(monkeypatch):
    REGISTRY.register_adapter(BACKEND, _StubAdapter)
    REGISTRY.register_client(BACKEND, lambda cfg: object())
    _StubAdapter.seen = []
    # The runner sizes text with a real tokenizer, so a model name it cannot resolve locally
    # becomes ~3 HuggingFace HEAD requests per job. Swallowed, so the tests pass either way
    # -- but on an air-gapped runner that is a DNS timeout per job, and one run here took
    # 100s against 6s offline. Nothing in this file asserts on sizes.
    monkeypatch.setattr("vagen.evaluation.runner._load_sizers", lambda name: (None, None))
    yield
    REGISTRY._adapters.pop(BACKEND, None)
    REGISTRY._clients.pop(BACKEND, None)


def _job(seed=1, tag="t", **extra):
    data = {"env_cls": _Env, "env_config": {"name": "Stub"}, "seed": seed, "tag_id": tag,
            "split": "test", "env_name": "Stub", "max_turns": 3, "chat_config": {}}
    data.update(extra)
    return {"data": data}


def _run(jobs, dump_dir, **kw):
    return asyncio.run(run_eval_parallel(
        jobs, backend=BACKEND, backend_cfg={}, model=kw.pop("model", "m-A"),
        default_max_turns=3, dump_dir=str(dump_dir), max_concurrent_jobs=2, **kw))


# --------------------------------------------------- the three-hop harness plumbing
@pytest.mark.parametrize("harness,expected,extra", [
    ("concat", ["system", "user", "assistant", "user", "assistant", "user"], {}),
    ("no_concat", ["system", "user"], {}),
    # compact is the one with extra validation of its own, so it is the one most able to
    # resolve to something other than what the config asked for.
    ("compact", ["system", "user", "assistant", "user", "assistant", "user"],
     {"compact_budget": 4096, "response_length_per_turn": 256}),
])
def test_the_harness_reaches_the_model_call(tmp_path, harness, expected, extra):
    """★ config -> _expand_jobs -> runner -> workflow. Every earlier test stopped at the
    first hop, so dropping the key at hop two or hardcoding concat at hop three both left
    the suite green -- and `harness: compact` in a yaml would again run as concat."""
    _run([_job(harness=harness, **extra)], tmp_path)
    assert _StubAdapter.seen, "no model call was made"
    assert _StubAdapter.seen[-1] == expected, (
        f"{harness} sent {_StubAdapter.seen[-1]}, so the key did not reach the call")


def test_a_bad_job_does_not_take_the_finished_ones_with_it(tmp_path):
    """Behavioural, replacing an inspect.getsource assertion that passed even with a bare
    `raise` reinstated in the handler."""
    results = _run([_job(seed=1), _job(seed=2, tag=None), _job(seed=3)], tmp_path)
    reasons = [r.get("finish_reason") for r in results]
    assert len(results) == 3, "the batch was truncated by the bad job"
    assert reasons.count("setup_error") == 1, reasons
    assert sum(r.get("num_turns", 0) for r in results) == 6, "finished turns were lost"


# ------------------------------------------------------------- resume and rerun
def _read_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _models_on_disk(dump_dir):
    out = []
    for root, _dirs, files in os.walk(dump_dir):
        if "metrics.json" in files:
            out.append(_read_json(os.path.join(root, "metrics.json")).get("model"))
    return sorted(x for x in out if x)


def test_each_rollout_records_which_model_produced_it(tmp_path):
    """The premise the two tests below rest on. Resume and the summary both recover the
    model from metrics.json; if the workflow stops writing it they degrade quietly rather
    than fail, so pin it here where the failure names the cause."""
    _run([_job(seed=1)], tmp_path, model="m-A")
    _run([_job(seed=1)], tmp_path, model="m-B")
    assert _models_on_disk(tmp_path) == ["m-A", "m-B"]


def test_resume_matches_a_rollout_to_its_own_model_and_no_other(tmp_path):
    """★ Both halves of the key, against real files on disk.

    The earlier test for this called `_job_resume_key` alone -- the pure half -- so
    removing the model from `_collect_completed_runs`, which reads it back out of
    metrics.json, left the suite green. That is the half that decides whether checkpoint
    B's run is skipped and A's success_rate reprinted under B's name.

    Note `run_eval_parallel` cannot cover this: resume lives in `main()`.
    """
    from vagen.evaluation.cli import _collect_completed_runs, _job_resume_key

    _run([_job(seed=1)], tmp_path, model="m-A")
    completed = _collect_completed_runs(str(tmp_path))
    assert completed, "the finished rollout was not indexed at all"

    same = _job_resume_key({"env_name": "Stub", "seed": 1, "tag_id": "t",
                            "resume_model": "m-A"})
    other = _job_resume_key({"env_name": "Stub", "seed": 1, "tag_id": "t",
                             "resume_model": "m-B"})
    assert completed.get(same) == "done", (
        "a rerun of the SAME model is not recognised, so resume never skips anything")
    assert other not in completed, (
        "a different checkpoint matches A's rollout: its episodes would be skipped and "
        "A's numbers reprinted under B's name")


def _error_rollout(tag_dir, name="0-deadbeef"):
    """A rollout the purge is entitled to delete: an error, so it will be rerun."""
    d = tag_dir / name
    d.mkdir(parents=True)
    (d / "metrics.json").write_text(json.dumps(
        {"finish_reason": "error", "env_name": "Stub", "seed": 9, "tag_id": "x",
         "model": "m-A"}), encoding="utf-8")
    return d


def test_the_purge_clears_only_the_tags_this_run_writes(tmp_path):
    """★ Untested entirely: restoring "clear every tag_*" left the suite green. navigation
    puts three tags in one dump dir, so the unscoped version destroyed results nobody
    asked to rerun.

    Both rollouts must be *error* rollouts: the purge only ever removes those, so seeding
    finished ones makes the assertion true whatever the scoping does.
    """
    from vagen.evaluation.cli import _purge_error_rollouts

    keep = _error_rollout(tmp_path / "tag_keep")
    rerun = _error_rollout(tmp_path / "tag_rerun")

    _purge_error_rollouts(str(tmp_path), "skip_completed", tags={"tag_rerun"})
    assert not rerun.is_dir(), "the tag this run does write was not cleared"
    assert keep.is_dir(), "a tag this run does not touch was purged"


def test_the_startup_refresh_does_not_zero_another_model_s_summary(tmp_path):
    """★ The refresh runs before this run's rollouts exist and filters on `model`, so for a
    second checkpoint it found nothing and wrote `n_episodes: 0, success_rate: 0.0` over the
    first checkpoint's results. The per-tag rewrite that would repair it only runs if the
    run reaches the end -- a bad api_key or a Ctrl-C made the loss permanent."""
    from vagen.evaluation.cli import _refresh_tag_summaries
    from vagen.evaluation.summary import write_rollouts_summary_from_dump

    _run([_job(seed=1)], tmp_path, model="m-A")
    tag = tmp_path / "tag_t"
    write_rollouts_summary_from_dump(dump_dir=str(tag), filename="summary.json", model="m-A")
    before = _read_json(tag / "summary.json")
    assert before["n_episodes"] == 1, "fixture did not produce a summary to protect"

    _refresh_tag_summaries(str(tmp_path), model="m-B", tags={"tag_t"})

    after = _read_json(tag / "summary.json")
    assert after["n_episodes"] == before["n_episodes"], (
        "starting an eval of a second checkpoint destroyed the first one's summary")
    assert after["success_rate"] == before["success_rate"]


def test_the_summary_counts_one_run_not_two(tmp_path):
    """A rerun writes new {timestamp}-{uuid8} directories beside the old ones, and the
    summary scans the directory -- so without the model filter a 1-episode config reported
    two episodes at a blended rate."""
    from vagen.evaluation.summary import write_rollouts_summary_from_dump

    _run([_job(seed=1)], tmp_path, model="m-A")
    _run([_job(seed=1)], tmp_path, model="m-B")
    tag = tmp_path / "tag_t"
    filtered = _read_json(write_rollouts_summary_from_dump(
        dump_dir=str(tag), filename="s.json", model="m-B"))
    assert filtered["n_episodes"] == 1, (
        f"summary counted {filtered['n_episodes']} episodes for a one-episode run")


# --------------------------------------------------------- config overrides
@pytest.mark.parametrize("override,ok", [
    ("run.backend=vllm", True),
    ("backends.openai.model=foo", True),
    ("envs.0.seed=[1,60,1]", True),
    # EnvSpec defines it, the yaml leaves it at the default, and it is the documented way
    # to evaluate under another context policy -- the case this check must not refuse.
    ("envs.0.harness=no_concat", True),
    ("envs.0.config.dim_room=[8,8]", True),          # passed through to the environment
    ("envs.0.chat_config.temperature=0.7", True),    # passed through to the client
    ("+something.genuinely.new=1", True),            # hydra's convention, honoured here
    ("run.backendd=vllm", False),
    ("experiment.dumpdir=/tmp/x", False),
    ("envs.0.harnes=no_concat", False),
])
def test_a_mistyped_override_is_refused_rather_than_invented(tmp_path, override, ok):
    """★ OmegaConf.update creates whatever key it is given. `run.backendd=vllm` and
    `experiment.dumpdir=/tmp/x` were both accepted in silence -- the run went ahead on the
    real setting and exited 0, so the only evidence was results in the wrong place, or a
    backend you did not choose."""
    from vagen.evaluation.cli import _load_config

    cfg = "examples/evaluate/sokoban/config.yaml"
    if ok:
        _load_config(cfg, [override])
        return
    with pytest.raises(ValueError, match="not in the config"):
        _load_config(cfg, [override])


def test_the_refusal_names_the_key_it_meant(tmp_path):
    from vagen.evaluation.cli import _load_config

    with pytest.raises(ValueError, match=r"Did you mean run\.backend\?"):
        _load_config("examples/evaluate/sokoban/config.yaml", ["run.backendd=vllm"])
    # From EnvSpec's fields, not from the keys this yaml happens to spell out.
    with pytest.raises(ValueError, match=r"Did you mean envs\.0\.harness\?"):
        _load_config("examples/evaluate/sokoban/config.yaml", ["envs.0.harnes=x"])
