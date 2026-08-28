"""The configurations that cannot produce an episode, caught before the rollout.

Every case here is decidable from the numbers alone. Left to run, each fails in a way
that costs the generation first and then either crashes with a shape or a length nobody
can trace back to the config, or does not fail at all -- compaction that buys no turns
finishes the episode, writes well-formed rows, and reports nothing.
"""

from __future__ import annotations

import pytest

from vagen.harness._common.budget import Budgets, BudgetError, check, default_summary_budget
from vagen.harness.compact import CompactHarness, CompactionMakesNoProgress


def _b(**kw):
    base = dict(prompt_len=1000, response_len=8000, per_turn=1000, max_turns=5)
    return Budgets(**{**base, **kw})


# ------------------------------------------------------------------ every mode
@pytest.mark.parametrize("mode", ["concat", "no_concat", "compact"])
def test_a_generation_larger_than_the_region_it_lands_in_is_refused(mode):
    with pytest.raises(BudgetError, match="does not fit the response region"):
        check(mode, _b(per_turn=9000, response_len=8000, compact_budget=4000, summary_budget=1000))


@pytest.mark.parametrize("mode", ["concat", "no_concat", "compact"])
def test_a_workable_configuration_passes_in_every_mode(mode):
    check(mode, _b(compact_budget=4000, summary_budget=1000))


# ---------------------------------------------------------------------- concat
def test_concat_warns_about_an_episode_whose_turns_cannot_all_fit():
    """A warning, not a refusal, and the change is deliberate.

    The arithmetic is a worst case -- every turn generating its full allowance -- and it
    is survivable two ways now: the generation is bounded by the room actually left, and
    what overflows anyway is truncated rather than refused. Refusing would rule out any
    long episode on the strength of a case a real rollout does not reach.
    """
    # ★ Matched through the `=` and the total. The previous regex stopped one token short
    # of the sum, so it went on passing after the sum stopped following from the terms it
    # printed -- a test asserting a false equation is worse than no test.
    with pytest.warns(UserWarning, match=r"max_turns=10 x response_length_per_turn=1000 = 10000 tokens"):
        check("concat", _b(max_turns=10, per_turn=1000, response_len=8000, env_response=200))


def test_concat_points_at_the_mode_that_solves_it_rather_than_only_the_number():
    with pytest.warns(UserWarning, match="trainer.harness=compact"):
        check("concat", _b(max_turns=10))


def test_the_same_episode_is_fine_under_no_concat():
    # The turns land in ten separate rows there, so nothing has to hold all of them.
    check("no_concat", _b(max_turns=10, per_turn=1000, response_len=8000))


# --------------------------------------------------------------------- compact
def test_a_summary_as_long_as_the_region_it_lives_in_is_refused():
    """A summary is a generation, and the client clamps it to response_length_per_turn --
    so a reservation larger than that is reserving room for something that cannot be
    written into it."""
    with pytest.raises(BudgetError, match="larger than anything that can be written"):
        check("compact", _b(compact_budget=400, summary_budget=8000, per_turn=512,
                            response_len=8000))



def test_an_unhelpful_optional_budget_warns_rather_than_refuses():
    """`2k <= m` used to be fatal. It is advisory now: the region trigger is what has to
    hold, and this only says the *optional* second trigger is set somewhere unhelpful."""
    wide = dict(per_turn=4000, response_len=40000, prompt_len=10000, env_response=100)
    check("compact", _b(compact_budget=4000, summary_budget=2000, **wide))
    with pytest.warns(UserWarning, match="more than half"):
        check("compact", _b(compact_budget=4000, summary_budget=2001, **wide))



def test_a_conversation_with_no_room_to_work_in_is_refused():
    """The one static thing compaction still cannot recover from: the summary, its
    request and one generation do not fit the region, so every conversation would close
    before its first turn."""
    with pytest.raises(BudgetError, match="no room to buy a turn"):
        check("compact", _b(prompt_len=1000, response_len=1000, compact_budget=6800,
                            summary_budget=600, per_turn=600, env_response=100,
                            summary_request_len=20))



def test_the_summary_cannot_outrun_a_single_generation():
    with pytest.raises(BudgetError, match="larger than anything that can be written"):
        check("compact", _b(per_turn=500, compact_budget=4000, summary_budget=1500,
                            response_len=8000))



def test_compact_without_a_summary_budget_says_so():
    """compact_budget is optional now -- the response region is the real bound -- but the
    summary budget is not: it is what gets reserved."""
    with pytest.raises(BudgetError, match="compact_summary_budget"):
        check("compact", _b(compact_budget=4000, summary_budget=None))
    check("compact", _b(compact_budget=None, summary_budget=100))



def test_the_derived_defaults_satisfy_the_rules_they_are_checked_against():
    """Defaults its own checker rejects would fail every unconfigured run."""
    from dataclasses import replace

    from vagen.harness._common.budget import default_env_response

    for m in (8, 100, 400, 4000, 40000):
        for per_turn in (64, 1024, 8000):
            b = _b(compact_budget=m, per_turn=per_turn,
                   response_len=max(4 * m + 4 * per_turn, 8000), prompt_len=max(1000, m))
            k = default_summary_budget(m, per_turn)
            b = replace(b, summary_budget=k)
            # Derived, so it must say so: the exact "can a conversation buy a turn"
            # check is meaningless against a number chosen to be generous.
            b = replace(b, env_response=default_env_response("compact", b))
            assert k >= 1
            check("compact", b)


def test_the_room_check_turns_over_exactly_where_it_says():
    """A bound the error tells you to use, that then fails, costs a second submission."""
    from dataclasses import replace

    # summary + request + generation + one observation + the floor below which the next
    # generation is not worth making. The observation IS charged here -- a turn pays for
    # one, and this is the one relation E belongs in (see budget.py's header).
    b = _b(compact_budget=None, summary_budget=500, per_turn=1000, env_response=300,
           prompt_len=4000, summary_request_len=13)
    for n_r in range(1800, 4000):
        floor = min(1000, max(1, n_r // 4))
        if 500 + 13 + 1000 + 300 + floor <= n_r:
            needed = n_r
            break
    check("compact", replace(b, response_len=needed))
    with pytest.raises(BudgetError, match="no room to buy a turn"):
        check("compact", replace(b, response_len=needed - 1))



def test_the_window_is_the_hard_context_when_that_is_the_smaller_one():
    """rollout.max_model_len below the sum of the regions is the real ceiling: the engine
    refuses past it, and no amount of room in the training tensors changes that."""
    from dataclasses import replace

    b = _b(compact_budget=None, summary_budget=500, per_turn=1000, env_response=300,
           prompt_len=4000, response_len=8000)
    assert b.window == b.row
    assert replace(b, context=4096).window == 4096
    with pytest.warns(UserWarning, match="rollout.max_model_len"):
        check("concat", replace(b, context=4096, max_turns=10, per_turn=1000))



# ------------------------------------------------------- the ceilings, enforced live
def test_an_oversized_observation_is_cut_to_the_ceiling():
    """Cut rather than refused: max_env_response_per_turn exists so an episode is bounded,
    and a bound that kills the rollout when an environment exceeds it only moves the
    failure. One oversized observation costs its own tail."""
    from vagen.rollout.client import InferenceClient

    class _C(InferenceClient):
        # one token per character, so sizes are legible
        def encode(self, messages): return [0] * sum(len(m["content"]) for m in messages)
        async def generate(self, prompt_ids, **kw): raise AssertionError("not reached")

    c = _C()
    c.opening_limit, c.continuation_limit = 1000, 400
    msgs = [{"role": "user", "content": "o" * 900}]
    assert c._fit_messages(msgs, opening=True) == msgs            # fits the prompt region
    cut = c._fit_messages(msgs, opening=False)
    assert len(cut[0]["content"]) <= 400
    assert cut[0]["content"] == "o" * len(cut[0]["content"])       # head kept, not tail


def _char_client():
    from vagen.rollout.client import InferenceClient

    class _C(InferenceClient):
        def encode(self, messages): return [0] * sum(len(m["content"]) for m in messages)
        async def generate(self, prompt_ids, **kw): raise AssertionError("not reached")
    return _C()


def test_an_oversized_opening_is_trimmed_but_the_system_prompt_is_never_cut():
    """★ The line is the system prompt, not the opening. Instructions cut identically on
    every episode are a config error laundered into silently degraded data; an oversized
    *observation* is just an oversized observation, and under no_concat every call is an
    opening -- so refusing them outright discarded every turn an episode had earned."""
    c = _char_client()
    c.opening_limit, c.continuation_limit = 1000, 400
    out = c._fit_messages([{"role": "system", "content": "s" * 100},
                           {"role": "user", "content": "o" * 1200}], opening=True)
    assert out[0]["content"] == "s" * 100, "the system prompt was cut"
    assert len(out[1]["content"]) < 1200 and out[1]["content"], "the observation was not trimmed"


def test_opening_trim_does_not_render_a_system_only_prompt():
    """Some templates, including Qwen3.5, reject a history with no user turn."""
    from vagen.rollout.client import InferenceClient

    class _Strict(InferenceClient):
        def encode(self, messages):
            if not any(message.get("role") == "user" for message in messages):
                raise RuntimeError("No user query found in messages")
            return [0] * sum(len(message["content"]) for message in messages)

        async def generate(self, prompt_ids, **kw):
            raise AssertionError("not reached")

    client = _Strict()
    client.opening_limit, client.continuation_limit = 100, 100
    trimmed = client._fit_messages(
        [{"role": "system", "content": "s" * 40},
         {"role": "user", "content": "o" * 100}],
        opening=True,
    )

    assert trimmed[0]["content"] == "s" * 40
    assert 0 < len(trimmed[1]["content"]) < 100


def test_an_opening_whose_system_prompt_alone_does_not_fit_still_refuses():
    """No cut repairs a prompt region too small to hold the instructions."""
    from vagen.rollout.client import ContextTooLarge

    c = _char_client()
    c.opening_limit, c.continuation_limit = 1000, 400
    with pytest.raises(ContextTooLarge, match="opening a conversation"):
        c._fit_messages([{"role": "system", "content": "s" * 1200}], opening=True)


def test_a_cut_drops_whole_images_once_the_text_is_gone():
    """A partial image is not an image: placeholders and frames have to stay 1:1, so a
    frame that will not fit is dropped entire, along with its placeholder."""
    from vagen.rollout.client import InferenceClient

    class _C(InferenceClient):
        def encode(self, messages):
            n = 0
            for m in messages:
                n += len(m["content"]) + 500 * len(m.get("images") or [])
            return [0] * n
        async def generate(self, prompt_ids, **kw): raise AssertionError("not reached")

    c = _C()
    c.continuation_limit = 600
    cut = c._fit_messages(
        [{"role": "user", "content": "<image><image>xx", "images": ["a", "b"]}],
        opening=False)
    assert len(cut[0]["images"]) == 1
    assert cut[0]["content"].count("<image>") == 1, "placeholders and frames diverged"


def test_the_two_ceilings_come_from_the_mode():
    from vagen.harness._common.budget import context_limits

    b = _b(prompt_len=9000, response_len=8000, per_turn=512, max_turns=5,
           env_response=1360, compact_budget=1300, summary_budget=325)
    assert context_limits("concat", b) == (9000, 1360)
    # ★ no_concat gets the observation ceiling too. It used to be handed `opening`, so
    # max_env_response_per_turn reached the client in two modes out of three and was inert
    # in the third -- while the docs call it "the ceiling one observation is cut to" with
    # no qualification. Every call there opens a conversation, so the observation lands in
    # the prompt region and the binding limit is whichever is smaller.
    assert context_limits("no_concat", b) == (9000, 1360)
    assert context_limits("no_concat", _b(prompt_len=500, response_len=8000, per_turn=512,
                                          max_turns=5, env_response=1360)) == (500, 500)
    # Compaction's openings are bounded by the prompt region like everyone else's. They
    # used to be bounded by compact_budget, from when that number was what a conversation
    # had to fit inside; it is an optional trigger now, and treating a trigger as a
    # ceiling killed the episode on its first call -- the opening is the system prompt
    # plus the first observation, with no summary in it yet to compact away.
    assert context_limits("compact", b) == (9000, 1360)


# ------------------------------------------------- max_env_response_per_turn (E)
#
# The environment-side twin of response_length_per_turn. Before it existed, E was
# "the largest value that still passes the checks", derived from the response region --
# which made every relation written in terms of E self-consistent and bounded nothing the
# environment actually did.


def test_the_default_env_response_is_a_flat_ceiling_not_a_share_of_the_region():
    """★ The derived value grew with the response region, so a bigger answer budget
    implied a bigger observation. At max_response_length=12288 it came to 10240 against a
    measured sokoban observation of 96 tokens, and the T*g + (T-1)*E warning then reported
    a 5-turn episode as needing 51k where a real one needs 11k -- firing on every rollout
    of every correctly sized config, which is how a warning stops being read."""
    from vagen.harness._common.budget import DEFAULT_MAX_ENV_RESPONSE, default_env_response

    big = default_env_response("concat", _b(response_len=12288, per_turn=2048))
    assert big == DEFAULT_MAX_ENV_RESPONSE == 2048


@pytest.mark.parametrize("mode", ["concat", "no_concat", "compact"])
def test_the_ceiling_does_not_depend_on_the_configuration_at_all(mode):
    """★ E is a truncation ceiling and nothing else, so there is nothing to derive it
    from. Both earlier derivations were wrong in the same way -- they made the size of an
    observation a function of how much room the model was given to answer it. From the
    worst-case sum it went negative and refused every observation; from the room left it
    grew with the response region, reaching 10240 against a measured 96."""
    from vagen.harness._common.budget import DEFAULT_MAX_ENV_RESPONSE, default_env_response

    for b in (_b(response_len=2000, per_turn=1000), _b(response_len=64000, per_turn=1000)):
        assert default_env_response(mode, b) == DEFAULT_MAX_ENV_RESPONSE


@pytest.mark.parametrize("mode", ["concat", "compact"])
def test_a_five_turn_episode_passes_without_warning(mode):
    """5 x 2048 generations inside an 11264 region. The observations are deliberately NOT
    part of this -- E feeds no static relation any more -- which is why the name no longer
    claims they are: the old one promised an observation budget the assertion never
    measured, and passed just as happily at env_response=100000."""
    import warnings

    b = _b(response_len=11264, per_turn=2048, max_turns=5, env_response=256, compact_budget=4000, summary_budget=1000)
    with warnings.catch_warnings():
        warnings.simplefilter("error")   # any budget warning becomes a failure
        check(mode, b)


def test_the_spec_accepts_the_new_name_and_the_old_one():
    from vagen.training.dataset import EnvSpec

    assert EnvSpec(name="Sokoban", n_envs=1, max_env_response_per_turn=256).max_env_response_per_turn == 256
    # the deprecated spelling is what the oversized-observation error told people to set
    assert EnvSpec(name="Sokoban", n_envs=1, env_response_length=256).max_env_response_per_turn == 256
    assert EnvSpec(name="Sokoban", n_envs=1).max_env_response_per_turn is None


def test_the_spec_refuses_two_different_values_for_one_quantity():
    from vagen.training.dataset import EnvSpec

    with pytest.raises(ValueError, match="They are one quantity"):
        EnvSpec(name="Sokoban", n_envs=1, max_env_response_per_turn=256, env_response_length=512)


# ------------------------------------------------------- thinking_token_budget
#
# The lever `response_length_per_turn` is not. `max_tokens` cuts the turn wherever it
# happens to be; this makes the engine close the reasoning block and let the model answer.
# Measured on Qwen3.5 at a 2048 max_tokens cap, 92% of turns were cut before `</think>`
# and so had no answer to score.


def test_the_budget_is_carried_as_a_sampling_key_and_nothing_else():
    """★ Model-agnostic on purpose. VAGEN passes a token count; what a reasoning block
    looks like lives in engine config (a registered reasoning_parser, or an explicit
    start/end pair), because that is per-family knowledge. Nothing here may learn that
    `<think>` is the delimiter -- the next family spells it differently."""
    import inspect

    from vagen.training.agent_loop import gym_loop

    src = inspect.getsource(gym_loop)
    assert "thinking_token_budget" in src
    assert "<think>" not in src and "</think>" not in src, \
        "a reasoning delimiter leaked into the agent loop; it belongs in engine config"


def test_the_spec_carries_the_budget_and_defaults_to_off():
    """Off unless asked for: set, it makes vLLM refuse the request unless the engine also
    has reasoning_config, so a default would break every run that does not want it."""
    from vagen.training.dataset import EnvSpec

    assert EnvSpec(name="Sokoban", n_envs=1).thinking_token_budget is None
    assert EnvSpec(name="Sokoban", n_envs=1, thinking_token_budget=512).thinking_token_budget == 512


def test_protocol_stop_strings_are_isolated_and_keep_the_delimiter():
    from vagen.training.dataset import EnvSpec

    first = EnvSpec(name="Sokoban", n_envs=1)
    second = EnvSpec(name="Sokoban", n_envs=1)
    first.stop_strings.append("<custom-stop>")
    assert second.stop_strings == ["</answer>"]

    sampling = {"temperature": 0.6}
    kwargs = {"stop_strings": first.stop_strings}
    if kwargs.get("stop_strings"):
        sampling = {
            **sampling,
            "stop": list(kwargs["stop_strings"]),
            "include_stop_str_in_output": True,
        }
    assert sampling["stop"] == ["</answer>", "<custom-stop>"]
    assert sampling["include_stop_str_in_output"] is True
    from vllm import SamplingParams
    params = SamplingParams(max_tokens=1024, **sampling)
    assert params.stop == ["</answer>", "<custom-stop>"]
    assert params.include_stop_str_in_output is True


def test_the_budget_reaches_the_client_sampling_params():
    """It has to survive as a plain extra key: verl builds its sampling dict from a fixed
    field list with no pass-through, but the engine call is
    `SamplingParams(max_tokens=..., **sampling_params)`, so an extra key arrives intact."""
    sampling = {"temperature": 1.0}
    kwargs = {"thinking_token_budget": 512}
    if kwargs.get("thinking_token_budget"):                     # the gym_loop line
        sampling = {**sampling, "thinking_token_budget": int(kwargs["thinking_token_budget"])}
    from vllm import SamplingParams
    p = SamplingParams(max_tokens=2048, **sampling)
    assert p.thinking_token_budget == 512


def test_an_unset_budget_adds_no_key_at_all():
    """A None must not become `thinking_token_budget=None` in the dict: vLLM checks for
    `is not None`, so passing it explicitly would demand reasoning_config from every run."""
    for value in (None, 0):
        sampling = {"temperature": 1.0}
        kwargs = {"thinking_token_budget": value}
        if kwargs.get("thinking_token_budget"):
            sampling = {**sampling, "thinking_token_budget": int(kwargs["thinking_token_budget"])}
        assert "thinking_token_budget" not in sampling


def test_compact_refuses_a_conversation_that_cannot_buy_a_turn_because_of_the_observation():
    """★ The hole that opened when E was briefly taken out of every relation. Differential
    search found 27 configurations refused before and accepted after; this is the smallest.
    Accepted, it compacts after one turn on every episode, raises CompactionMakesNoProgress,
    and returns an EMPTY BATCH deterministically -- with only a per-episode warning."""
    with pytest.raises(BudgetError, match="no room to buy a turn"):
        check("compact", _b(response_len=2000, per_turn=512, summary_budget=256,
                            summary_request_len=70, env_response=2048,
                            compact_budget=None, prompt_len=2000))
