"""Evaluation runs the same episode loop, and the harness comes from the config.

Before this, ``evaluate/`` carried a second implementation of the turn loop that predated
the harness abstraction. It hardcoded concat, approximated no_concat with a boolean, and
could not express compaction at all -- and because ``_parse_env_specs`` copied a fixed key
list and dropped the rest in silence, writing ``harness: compact`` in an eval config was
accepted, ignored, and ran concat. Nothing in the suite touched the eval path, so none of
that was visible.

These tests assert the thing that distinguishes the three policies: what history each
model call carries. They use a stub adapter, so what is under test is the wiring and the
harnesses, not any endpoint.
"""

from __future__ import annotations

import asyncio

import pytest

from vagen.evaluation.workflow import GenericVisionInferenceWorkflow


#: A reply of a realistic size. The stub used to answer in 39 characters (~10 tokens)
#: against response_length_per_turn=512, which is ~30x too small for any budget-driven
#: behaviour to show up -- the test named "an episode runs the turns it was configured
#: for" passed against the code that lost 40% of them.
_REPLY_PAD = "reasoning. " * 180          # ~2000 chars, ~500 tokens


class _Adapter:
    """Records the messages of every call and answers with a fixed action."""

    def __init__(self, reply=f"<think>{_REPLY_PAD}</think><answer>Right</answer>"):
        self.reply = reply
        self.calls: list[list[dict]] = []

    def format_system(self, text, images):
        return {"role": "system", "content": [{"type": "text", "text": text}]}

    def format_user_turn(self, text, images):
        return {"role": "user", "content": [{"type": "text", "text": text}]}

    def format_assistant_turn(self, text):
        return {"role": "assistant", "content": [{"type": "text", "text": text}]}

    async def acompletion(self, messages, **chat_config):
        self.calls.append([dict(m) for m in messages])
        self.chat_config_seen = dict(chat_config)
        return self.reply


#: Long enough that a compaction budget in the low hundreds is actually reached. The
#: client estimates 4 characters to the token, so this is ~100 tokens an observation.
_PAD = "." * 380


class _Env:
    """A text-only environment that never terminates on its own."""

    def __init__(self, env_config):
        self.config = env_config
        self.i = 0

    #: Set by _run(vision=True). One frame an observation, which is what makes
    #: tokens_per_image part of the arithmetic instead of dead config.
    vision = False

    def _obs(self, i):
        obs = {"obs_str": f"observation {i} {_PAD}"}
        if self.vision:
            # A real PIL image, not a placeholder object: the dump path calls img.save,
            # and GymEnvAdapter runs the frames through _normalize_images.
            from PIL import Image
            obs["obs_str"] = "<image>\n" + obs["obs_str"]
            obs["multi_modal_input"] = {"<image>": [Image.new("RGB", (16, 16))]}
        return obs

    async def reset(self, seed=None):
        self.i = 0
        return self._obs(0), {}

    async def system_prompt(self):
        return {"obs_str": "you are a solver"}

    async def step(self, action, **kw):
        self.i += 1
        return self._obs(self.i), 1.0, False, {"turn": self.i}

    async def close(self):
        self.closed = True


def _run(harness, turns=4, vision=False, **kw):
    adapter = _Adapter()
    env_cls = _Env
    if vision:
        env_cls = type("_VisionEnv", (_Env,), {"vision": True})
    wf = GenericVisionInferenceWorkflow(adapter=adapter, dump_dir=None,
                                        harness=harness, **kw)
    result = asyncio.run(wf.arun_episode(env_cls, {"name": "Stub"}, seed=0, max_turns=turns))
    return adapter, result


def _roles(call):
    return [m["role"] for m in call]


# --------------------------------------------------------------- the three policies
def test_concat_carries_the_whole_conversation_forward():
    adapter, _ = _run("concat")
    lengths = [len(c) for c in adapter.calls]
    assert lengths == sorted(lengths) and lengths[-1] > lengths[0], lengths
    # system once, at the front, and every earlier turn still present
    assert _roles(adapter.calls[-1])[0] == "system"
    assert _roles(adapter.calls[-1]).count("assistant") == len(adapter.calls) - 1


def test_no_concat_sends_only_the_system_prompt_and_the_latest_observation():
    adapter, _ = _run("no_concat")
    for call in adapter.calls:
        assert _roles(call) == ["system", "user"], _roles(call)
    # ...and it is the *latest* observation, not the first one repeated
    seen = [c[-1]["content"][0]["text"] for c in adapter.calls]
    assert len(set(seen)) == len(seen), seen


def test_compact_summarises_and_reopens_rather_than_growing_forever():
    """The policy the old eval loop could not express at all. A conversation is closed by
    asking the model to summarise, and the next one opens on that summary."""
    adapter, _ = _run("compact", turns=6, response_length_per_turn=1024,
                      max_response_length=16384, compact_budget=2500,
                      compact_summary_budget=500)
    lengths = [len(c) for c in adapter.calls]
    # it must come back down at least once -- that is the reopen
    assert any(b < a for a, b in zip(lengths, lengths[1:])), lengths


@pytest.mark.parametrize("harness", ["concat", "no_concat", "compact"])
def test_every_harness_produces_a_scored_episode(harness):
    _, result = _run(harness, turns=3, response_length_per_turn=1024,
                     max_response_length=8192, compact_budget=3000,
                     compact_summary_budget=600)
    assert result["num_turns"] == 3, f"{harness} ran {result['num_turns']} turns of 3"
    assert result["cumulative_reward"] > 0


# --------------------------------------------------------------- config, not code
def test_an_unknown_harness_is_refused_by_name():
    with pytest.raises(ValueError, match="unknown harness"):
        GenericVisionInferenceWorkflow(adapter=_Adapter(), harness="concat_multi_turn")


def test_the_harness_key_reaches_the_workflow_from_the_yaml(tmp_path):
    """★ The bug this whole change exists for: the key used to be dropped between the
    config and the workflow, silently, so every eval ran concat whatever it said."""
    from vagen.evaluation.cli import _parse_env_specs

    specs = _parse_env_specs(
        {"envs": [{"name": "Sokoban", "n_envs": 1, "tag_id": 0,
                   "harness": "compact", "compact_budget": 900}]}
    )
    assert specs[0].harness == "compact"
    assert specs[0].compact_budget == 900


def test_a_key_nothing_reads_is_an_error_rather_than_a_shrug():
    from vagen.evaluation.cli import _parse_env_specs

    with pytest.raises(ValueError, match="which nothing reads"):
        _parse_env_specs({"envs": [{"name": "Sokoban", "n_envs": 1,
                                    "harnes": "compact"}]})


def test_response_length_per_turn_becomes_the_api_call_max_tokens():
    """It was in an eval config already, and dropped. Now it bounds a turn the same way
    it does in training rather than leaving the endpoint to its own default."""
    adapter, _ = _run("concat", turns=2, response_length_per_turn=77)
    assert adapter.chat_config_seen.get("max_tokens") == 77


# ------------------------------------------------- what the reviewers found, pinned
#
# Each of these is a defect the harness rewrite introduced and the suite did not catch.


def test_an_episode_runs_the_turns_it_was_configured_for():
    """★ The worst of them. Deriving the response region as response_length_per_turn *
    max_turns looks like the episode's budget and is not -- under concat the observations
    land in the same region, so the region runs out early. Measured on the shipped
    frozenlake eval (g=512, T=5, one frame an observation): 3 turns of 5, reported as
    `max_turns`."""
    # ★ max_response_length is given, so the accounting is ON -- without it response_len
    # is None and this exercises the no-accounting branch instead of the arithmetic. And
    # the env renders a frame, so an observation costs what one really costs.
    adapter, result = _run("concat", turns=5, vision=True,
                           response_length_per_turn=512, max_response_length=8192)
    assert result["num_turns"] == 5, f"lost turns: {result['num_turns']} of 5"
    assert len(adapter.calls) == 5


def test_num_turns_counts_environment_steps_not_transcript_messages():
    """Under compact the transcript carries one extra assistant message per compaction,
    so recomputing the count from it inflates avg_turns by the compaction rate -- a
    quantity set by how verbosely the policy writes."""
    adapter, result = _run("compact", turns=6, response_length_per_turn=1024,
                           max_response_length=16384, compact_budget=2500,
                           compact_summary_budget=500)
    assert result["num_turns"] == 6
    assert len(adapter.calls) > 6, "this config was supposed to compact"


def test_the_finish_reason_of_a_full_episode_is_one_the_summary_treats_as_normal():
    """Anything outside NORMAL_FINISH_REASONS is filed as an error rollout and deleted by
    _purge_error_rollouts on the next resumed run."""
    from vagen.evaluation.runner import NORMAL_FINISH_REASONS

    _, result = _run("concat", turns=3)
    assert result["finish_reason"] in NORMAL_FINISH_REASONS


def test_per_turn_infos_stay_aligned_with_rewards():
    """summary_utils aligns per-turn fields on len(infos) == len(rewards) + 1. run_episode
    merges every step's info into one dict, so passing that through left every turn past
    the first reporting {}."""
    _, result = _run("concat", turns=4)
    assert len(result["infos"]) == len(result["rewards"]) + 1


def test_a_failure_partway_through_keeps_the_turns_that_finished():
    """The client and transcript used to be built inside the try, so a provider error on
    call 3 of 6 reported num_turns=0 and an empty transcript -- and `error` is not a normal
    finish reason, so the dump was then deleted on the next resumed run."""
    class _Failing(_Adapter):
        async def acompletion(self, messages, **cfg):
            if len(self.calls) >= 2:
                raise RuntimeError("provider exploded")
            return await super().acompletion(messages, **cfg)

    adapter = _Failing()
    wf = GenericVisionInferenceWorkflow(adapter=adapter, dump_dir=None, harness="concat")
    result = asyncio.run(wf.arun_episode(_Env, {"name": "Stub"}, seed=0, max_turns=6))
    assert result["finish_reason"] == "error"
    assert result["num_turns"] == 2, f"lost the finished turns: {result['num_turns']}"
    assert len(result["rewards"]) == 2
    assert any(m["role"] == "assistant" for m in result["messages"])


def test_an_empty_reply_is_a_refusal_not_something_to_retry():
    """The base client retries an empty generation three times because an engine returning
    nothing is an interruption. A chat API returning "" is a refusal, and asking again just
    pays for it four times."""
    class _Empty(_Adapter):
        async def acompletion(self, messages, **cfg):
            await super().acompletion(messages, **cfg)
            return ""

    adapter = _Empty()
    wf = GenericVisionInferenceWorkflow(adapter=adapter, dump_dir=None, harness="concat")
    asyncio.run(wf.arun_episode(_Env, {"name": "Stub"}, seed=0, max_turns=1))
    assert len(adapter.calls) == 1, f"{len(adapter.calls)} calls for one refusal"


def test_compact_without_a_trigger_is_refused_rather_than_run_as_concat():
    """With no compact_budget and no max_response_length neither trigger can fire, so the
    conversation grows forever -- silently concat, under a name that says otherwise."""
    with pytest.raises(ValueError, match="compact needs compact_budget"):
        wf = GenericVisionInferenceWorkflow(adapter=_Adapter(), dump_dir=None,
                                            harness="compact")
        asyncio.run(wf.arun_episode(_Env, {"name": "Stub"}, seed=0, max_turns=3))


# ------------------------------------------ resume and aggregation, pinned
#
# The eval path had no tests at all, and these are the failures that produce a wrong
# NUMBER rather than an error: a rerun that double-counts, a resume that reprints another
# model's score, and one bad job that discards the whole batch.


def test_resume_will_not_reuse_a_rollout_from_a_different_model():
    """★ The key is (env, seed, tag, model). Without the model, evaluating checkpoint B
    into the directory checkpoint A used skipped all of A's episodes and reprinted A's
    success_rate under B's name -- exit 0, nothing in the output saying so."""
    from vagen.evaluation.cli import _job_resume_key

    base = {"env_name": "Sokoban", "seed": 1, "tag_id": "t"}
    a = _job_resume_key({**base, "resume_model": "/ckpt/A"})
    b = _job_resume_key({**base, "resume_model": "/ckpt/B"})
    assert a != b, "two models share a resume key"
    assert a == _job_resume_key({**base, "resume_model": "/ckpt/A"})


def test_one_normal_finish_reason_set_governs_deletion_and_reporting():
    """There were two, and they disagreed: `no_room` was kept on disk and treated as
    completed by resume while being reported as an error rollout."""
    from vagen.evaluation.runner import NORMAL_FINISH_REASONS as a
    from vagen.evaluation.summary import NORMAL_FINISH_REASONS as b

    assert a == b
    _, result = _run("concat", turns=3)
    assert result["finish_reason"] in a


def test_a_job_that_cannot_even_be_set_up_does_not_take_the_batch_with_it():
    """The tag_id check, the max_turns validation, the adapter build and the workflow
    construction all sat above the try, and `await fut` re-raises -- so one malformed job
    aborted the run and discarded every episode that had already finished."""
    import inspect

    from vagen.evaluation import runner

    src = inspect.getsource(runner)
    assert "async def _run_one" in src, "setup is not inside the guarded path"
    body = src[src.index("async def _runner"):src.index("async def _run_one")]
    assert "try:" in body and "except Exception" in body


# --------------------------------------------- any BaseHarness, not just the built-in three
#
# Evaluation is usually where a new context policy is tried first, and an eval config is a
# yaml -- there is nowhere to put a decorator that would have run by then. So a name from
# the registry and an import path both work, and both are checked against BaseHarness.


class _EveryOtherTurn(_Adapter):
    pass


def _custom_harness_cls():
    from vagen.harness import ConcatHarness

    class ShoutHarness(ConcatHarness):
        """A real BaseHarness subclass that is not one of the three."""

    return ShoutHarness


def test_a_registered_custom_harness_is_selectable_by_name():
    from vagen.harness import HARNESSES, register_harness

    cls = _custom_harness_cls()
    register_harness("shout")(cls)
    try:
        adapter, result = _run("shout", turns=3)
        assert result["num_turns"] == 3
        assert len(adapter.calls) == 3
    finally:
        HARNESSES.pop("shout", None)


def test_a_custom_harness_works_as_an_import_path_with_nothing_registered():
    """``module:Class``, for the case where there is no package to hang a decorator on."""
    adapter, result = _run("vagen.harness.no_concat:NoConcatHarness", turns=3)
    assert [m["role"] for m in adapter.calls[-1]] == ["system", "user"]
    assert result["num_turns"] == 3


def test_registering_something_that_is_not_a_harness_is_refused():
    from vagen.harness import register_harness

    with pytest.raises(TypeError, match="does not subclass BaseHarness"):
        register_harness("nope")(dict)


def test_an_import_path_naming_the_wrong_class_fails_at_construction():
    """Not at the first next_call, where it would be an AttributeError deep in the loop."""
    with pytest.raises(TypeError, match="does not subclass BaseHarness"):
        GenericVisionInferenceWorkflow(adapter=_Adapter(), harness="collections:OrderedDict")


def test_shadowing_a_registered_harness_is_refused():
    """A silent rebinding means a run reports the policy it was configured with and
    executes another one."""
    from vagen.harness import register_harness

    with pytest.raises(ValueError, match="already registered"):
        register_harness("concat")(_custom_harness_cls())


# --------------------------------------------------- the vision half, which had no tests
#
# The workflow is called GenericVisionInferenceWorkflow and every test above ran text-only,
# so two regressions in the same commit went unnoticed: frames stopped being dumped
# entirely, and tokens_per_image -- documented as the term that drives the compaction
# trigger -- was never exercised.


def test_vision_rollouts_keep_their_frames(tmp_path):
    """★ `images/` was being created empty on every rollout. `run_episode` returns rewards
    and a merged info; the frames only survive because the workflow records them per step."""
    adapter = _Adapter()
    wf = GenericVisionInferenceWorkflow(adapter=adapter, dump_dir=str(tmp_path),
                                        harness="concat")
    env_cls = type("_VisionEnv", (_Env,), {"vision": True})
    asyncio.run(wf.arun_episode(env_cls, {"name": "Stub"}, seed=0, max_turns=3))

    rollouts = [p for p in tmp_path.iterdir() if p.is_dir()]
    assert rollouts, "nothing was dumped"
    images = rollouts[0] / "images"
    assert images.is_dir() and any(images.iterdir()), "images/ was created and left empty"


def test_tokens_per_image_changes_what_the_harness_thinks_a_turn_costs():
    """★ It is not just an overflow guard -- it feeds the compaction trigger. Set far above
    the environment's real frame cost it made every compact episode die in
    CompactionMakesNoProgress, on numbers that were wrong."""
    # max_env_response_per_turn has to be above the image price, or the observation
    # ceiling trims the frame away first and the price never reaches the accounting --
    # which is itself the right behaviour, and was what made the first version of this
    # test see no difference at all.
    common = dict(turns=3, vision=True, response_length_per_turn=512,
                  max_response_length=8192, max_env_response_per_turn=16384)
    cheap, _ = _run("concat", tokens_per_image=32, **common)
    dear, _ = _run("concat", tokens_per_image=4096, **common)
    # Same episode, same replies; only the image price differs, and the expensive one
    # exhausts the region first.
    assert len(cheap.calls) > len(dear.calls), (
        f"tokens_per_image had no effect: {len(cheap.calls)} vs {len(dear.calls)} calls")


def test_a_crashed_environment_is_not_reported_as_a_solved_episode():
    """The adapter turns any step exception into done=True, which without the env_error
    check is indistinguishable from a normal terminal state -- so eval error rates
    under-reported environment failures to zero."""
    class _Broken(_Env):
        async def step(self, action, **kw):
            raise RuntimeError("simulator died")

    adapter = _Adapter()
    wf = GenericVisionInferenceWorkflow(adapter=adapter, dump_dir=None, harness="concat")
    result = asyncio.run(wf.arun_episode(_Broken, {"name": "Stub"}, seed=0, max_turns=3))
    assert result["finish_reason"] == "env_error", result["finish_reason"]
    assert result["success"] is False


def test_an_unusable_observation_ceiling_is_refused_not_silently_emptied():
    """★ The shipped sokoban eval played blind from turn 2 and reported a success rate.

    `max_env_response_per_turn: 256` was copied from the training yaml, where a frame is
    ~96 real tokens -- but evaluation without a processor prices one at 800, so every
    continuation measured 814, and `_shrink` scaled the text to zero and dropped the frame.
    The model saw the empty string. One warning, exit 0, a number written to summary.json.

    A ceiling that cannot hold one observation is a config error, and has to say so.
    """
    from vagen.rollout.client import ContextTooLarge

    adapter = _Adapter()
    wf = GenericVisionInferenceWorkflow(
        adapter=adapter, dump_dir=None, harness="concat",
        max_env_response_per_turn=256, tokens_per_image=800)
    env_cls = type("_VisionEnv", (_Env,), {"vision": True})
    result = asyncio.run(wf.arun_episode(env_cls, {"name": "Stub"}, seed=0, max_turns=3))
    assert result["finish_reason"] == "error", (
        f"expected a refusal, got {result['finish_reason']!r} -- the observation was cut "
        f"to nothing and the episode reported normally")
    assert "cannot be brought under" in str(result.get("error_details", {}))


def test_a_processor_prices_a_frame_instead_of_guessing_at_it():
    """The estimate exists only for a closed API. Given the served model, evaluation
    measures what training measures -- 40x apart on a 96x96 frame, which is the difference
    between a ceiling that fits and one that erases the observation."""
    from PIL import Image
    from transformers import AutoProcessor

    from model_path import local_snapshot
    from vagen.evaluation.client import ChatClient

    msg = {"role": "user", "content": "<image>\nDecide your next action(s).",
           "images": [Image.new("RGB", (96, 96))]}
    proc = AutoProcessor.from_pretrained(local_snapshot("Qwen/Qwen2.5-VL-3B-Instruct"))

    measured = ChatClient(_Adapter(), processor=proc).measure([msg])
    estimated = ChatClient(_Adapter()).measure([msg])
    assert measured < 256, f"a 96x96 frame measured {measured}; the shipped ceiling is 256"
    assert estimated > measured * 5, "the estimate is supposed to be the pessimistic one"
