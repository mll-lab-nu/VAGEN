"""The configurations that cannot produce an episode, caught before the rollout.

Every case here is decidable from the numbers alone. Left to run, each fails in a way
that costs the generation first and then either crashes with a shape or a length nobody
can trace back to the config, or does not fail at all -- compaction that buys no turns
finishes the episode, writes well-formed rows, and reports nothing.
"""

from __future__ import annotations

import pytest

from vagen.harness.budget import Budgets, BudgetError, check, default_summary_budget
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
def test_concat_refuses_an_episode_whose_turns_cannot_all_fit():
    """concat's response region holds the whole episode, so this is arithmetic, not luck."""
    with pytest.raises(BudgetError, match=r"10 x response_length_per_turn=1000 \+ 9 x env_response_length=200"):
        check("concat", _b(max_turns=10, per_turn=1000, response_len=8000, env_response=200))


def test_concat_points_at_the_mode_that_solves_it_rather_than_only_the_number():
    with pytest.raises(BudgetError, match="trainer.harness=compact"):
        check("concat", _b(max_turns=10))


def test_the_same_episode_is_fine_under_no_concat():
    # The turns land in ten separate rows there, so nothing has to hold all of them.
    check("no_concat", _b(max_turns=10, per_turn=1000, response_len=8000))


# --------------------------------------------------------------------- compact
def test_a_summary_as_long_as_the_budget_it_compresses_into_is_refused():
    """The live default before this existed: the summary generated against the turn
    budget, so at compact_budget=400 it was allowed to be 8000 tokens -- twenty times
    the thing it was being written into. It worked only because the model wrote short
    ones, and nothing would have said so if it stopped."""
    with pytest.raises(BudgetError, match="buys no turns"):
        check("compact", _b(compact_budget=400, summary_budget=8000, per_turn=8000, response_len=8000))


def test_the_boundary_is_half():
    # per_turn is raised past the summary budget and the window past the peak, so this
    # exercises the halving rule and not one of the others -- which would otherwise fire
    # first and let the test pass for the wrong reason.
    wide = dict(per_turn=4000, response_len=40000, prompt_len=10000)
    check("compact", _b(compact_budget=4000, summary_budget=2000, **wide))
    with pytest.raises(BudgetError, match="more than half"):
        check("compact", _b(compact_budget=4000, summary_budget=2001, **wide))


def test_the_peak_of_a_compacted_conversation_has_to_fit_the_row_it_becomes():
    """A conversation is largest at the moment it is summarised, not before."""
    with pytest.raises(BudgetError, match="largest at the moment it is summarised"):
        check("compact", _b(prompt_len=1000, response_len=4000, compact_budget=4800,
                            summary_budget=1000, per_turn=1000, summary_request_len=20))


def test_the_summary_cannot_outrun_a_single_generation():
    with pytest.raises(BudgetError, match="cannot be longer than one"):
        check("compact", _b(per_turn=500, compact_budget=4000, summary_budget=1500))


def test_compact_without_its_budgets_says_which_one_is_missing():
    with pytest.raises(BudgetError, match="compact_budget"):
        check("compact", _b(compact_budget=None, summary_budget=100))
    with pytest.raises(BudgetError, match="compact_summary_budget"):
        check("compact", _b(compact_budget=4000, summary_budget=None))


def test_the_derived_defaults_satisfy_the_rules_they_are_checked_against():
    """Defaults its own checker rejects would fail every unconfigured run."""
    from dataclasses import replace

    from vagen.harness.budget import default_env_response

    for m in (8, 100, 400, 4000, 40000):
        for per_turn in (64, 1024, 8000):
            b = _b(compact_budget=m, per_turn=per_turn,
                   response_len=max(8 * m, 8000, per_turn), prompt_len=max(1000, m))
            k = default_summary_budget(m, per_turn)
            b = replace(b, summary_budget=k)
            b = replace(b, env_response=default_env_response("compact", b))
            assert k >= 1
            check("compact", b)


def test_the_top_of_the_band_is_exactly_where_the_check_turns_over():
    """A bound the error tells you to use, that then fails, costs a second submission."""
    from dataclasses import replace

    from vagen.harness.budget import compact_budget_bounds

    b = _b(compact_budget=1, summary_budget=500, per_turn=1000, env_response=300,
           prompt_len=4000, response_len=8000, summary_request_len=13)
    _, highest = compact_budget_bounds(b)
    check("compact", replace(b, compact_budget=highest))
    with pytest.raises(BudgetError, match="largest at the moment it is summarised"):
        check("compact", replace(b, compact_budget=highest + 1))


def test_the_window_is_the_hard_context_when_that_is_the_smaller_one():
    """rollout.max_model_len below the sum of the regions is the real ceiling: the engine
    refuses past it, and no amount of room in the training tensors changes that."""
    from dataclasses import replace

    from vagen.harness.budget import compact_budget_bounds

    b = _b(compact_budget=1, summary_budget=500, per_turn=1000, env_response=300,
           prompt_len=4000, response_len=8000)
    roomy = compact_budget_bounds(b)[1]
    tight = compact_budget_bounds(replace(b, context=4096))[1]
    assert tight < roomy
    with pytest.raises(BudgetError, match="rollout.max_model_len"):
        check("compact", replace(b, context=4096, compact_budget=roomy))


# --------------------------------------------------------------- the live dynamics
SYS = {"role": "system", "content": "sys"}
OBS = {"role": "user", "content": "obs"}
REQ = 13


def simulate(*, m, k, system, env_response, generation, turns):
    """Drive the harness exactly as the runner does, over a conversation of known size.

    The arithmetic lives in the test only to *describe* a conversation -- what the system
    prompt, an observation and a response cost. Every decision about when to compact is
    the harness's.
    """
    h = CompactHarness(budget=m, summary_budget=k)
    h.begin(SYS, OBS)
    used = peak = here = 0
    seeded, per_conversation = False, []

    for _ in range(turns):
        while True:
            call = h.next_call()
            summary = "Summarise" in str(call.messages[0]["content"])
            if call.conversation_id is None:               # opens a conversation
                used, here = system + (k if seeded else 0) + env_response + generation, 0
            elif summary:                                  # the request, then the summary
                used += REQ + k
            else:                                          # an ordinary turn
                used += env_response + generation
            peak = max(peak, used)
            h.note_usage(used)

            class _R:
                text, conversation_id = "x", "c"
            if h.accept(_R()) is None:                     # the summary was consumed
                per_conversation.append(here)
                seeded = True
            else:
                here += 1
                break
    per_conversation.append(here)
    return peak, per_conversation


# Sokoban vision, measured: a 589-token system prompt, observations of 44-58 with a
# 96x96 frame, and responses of about 80 against a configured ceiling of 512.
SOKOBAN = dict(system=589, env_response=58, generation=80)


def test_compaction_buys_turns_at_a_budget_that_fits_the_system_prompt():
    peak, per_conversation = simulate(m=1300, k=325, turns=12, **SOKOBAN)
    assert len(per_conversation) > 1, "the budget was never reached, so nothing compacted"
    assert min(per_conversation[:-1]) >= 2, (
        f"a conversation held one turn: {per_conversation}. That is no_concat with a "
        f"summary attached, which is what compaction is supposed to avoid."
    )
    assert peak <= 1300 + 58 + 512 + REQ + 325, "the peak escaped the guaranteed bound"


def test_a_budget_under_the_system_prompt_cannot_work_and_says_so():
    """compact_budget=400 against a 589-token system prompt, which is what this ran with
    for three runs. Every conversation opens over the budget, so every one of them
    summarises after a single turn."""
    with pytest.raises(CompactionMakesNoProgress, match="2 conversations in a row"):
        simulate(m=400, k=100, turns=12, **SOKOBAN)


def test_the_trigger_measures_a_turn_rather_than_charging_the_ceiling():
    """The reason the trigger cannot use env_response_length + response_length_per_turn.

    Those are ceilings -- on Sokoban 58 + 512 against a real turn of 138 -- and a trigger
    charging them fires after the first turn of every conversation. It would look safe:
    the peak stays inside the budget, every row is well-formed, and the mode is silently
    no_concat at twice the price.
    """
    h = CompactHarness(budget=1300, summary_budget=325)
    h.begin(SYS, OBS)

    class _R:
        text, conversation_id = "x", "c"

    def turn(used):
        # The runner's order, and it matters: note_usage runs before accept, so an opening
        # call is still unattributed to a conversation when its size is reported. Swap the
        # two and the system prompt is charged to the first turn.
        h.next_call()
        h.note_usage(used)
        h.accept(_R())

    turn(589 + 58 + 80)                  # the opening call: system prompt, observation, response
    turn(589 + 58 + 80 + 138)            # one ordinary turn on top

    assert h.turn_cost == 138, f"the estimate charged {h.turn_cost}, not what a turn cost"
    assert h.turn_cost < 58 + 512, "the estimate is the ceiling, so it predicts nothing"


def test_the_opening_call_is_never_counted_as_a_turn():
    """It carries the system prompt, so charging it as a turn would overestimate every
    later one -- on Sokoban by 589 tokens, enough to trip the budget immediately."""
    h = CompactHarness(budget=100_000, summary_budget=100)
    h.begin(SYS, OBS)
    h.next_call()

    class _R:
        text, conversation_id = "x", "c"
    h.note_usage(727)                     # 589 + 58 + 80, all of it the opening call
    h.accept(_R())
    assert h.turn_cost == 0, "the opening call was mistaken for a turn"


def test_one_short_conversation_is_data_and_does_not_kill_the_run():
    """An unusually large observation should not end a training job -- only a budget that
    cannot buy a turn should, and that is what a second one in a row proves."""
    h = CompactHarness(budget=1000, summary_budget=250)
    h.begin(SYS, OBS)
    for used, expect_summary in ((950, False), (1100, True), (300, False), (500, False)):
        h.note_usage(used)
        call = h.next_call()

        class _R:
            text, conversation_id = "x", "c"
        assert ("Summarise" in str(call.messages[0]["content"])) is expect_summary
        h.accept(_R())


def test_the_summary_call_is_bounded_by_its_own_budget_not_the_turn_budget():
    """Without this the summary is capped by response_length_per_turn, which is the whole
    reason it could be longer than the budget it is compressing into."""
    h = CompactHarness(budget=100, summary_budget=25)
    h.begin(SYS, OBS)
    h.next_call()

    class _R:
        text, conversation_id = "act", "c1"
    h.accept(_R())
    h.note_usage(10)
    h.next_call()
    h.accept(_R())
    h.note_usage(999)

    summary_call = h.next_call()
    assert "Summarise" in summary_call.messages[0]["content"]
    assert summary_call.sampling_params == {"max_new_tokens": 25}


def test_the_runner_forwards_a_calls_own_limits():
    """A limit the harness sets and the runner drops is worse than no limit: it reads as
    bounded everywhere it is written down."""
    import asyncio

    from vagen.core.harness import BaseHarness, Call
    from vagen.core.runner import run_episode

    seen = []

    class _Client:
        tokenizer = None

        async def send(self, messages, conversation_id=None, **kw):
            seen.append(kw.get("sampling_params"))

            class _R:
                text, conversation_id, token_ids = "act", "c1", [1]
            return _R()

        def usage(self, cid): return 1
        def reward(self, cid, v): pass

    class _H(BaseHarness):
        def next_call(self):
            return Call([{"role": "user", "content": "x"}], None,
                        sampling_params={"max_new_tokens": 7})

    class _Env:
        async def reset(self, seed): return {"obs_str": "o"}, {}
        async def system_prompt(self): return {"role": "system", "content": "s"}
        async def step(self, action, **kw): return {"obs_str": "o"}, 0.0, True, False, {}
        async def close(self): pass

    asyncio.run(run_episode(_Env(), _H(), _Client(), max_turns=1,
                            sampling_params={"temperature": 0.5}))
    assert seen == [{"temperature": 0.5, "max_new_tokens": 7}], seen


# ------------------------------------------------------- the ceilings, enforced live
def test_an_oversized_observation_is_named_where_it_happens():
    """Left to cap_token_ids it surfaces at the end of the episode as a truncated row,
    which says nothing about the observation that caused it."""
    from vagen.core.client import ContextTooLarge, InferenceClient

    class _C(InferenceClient):
        def encode(self, messages): return [0] * 900
        async def generate(self, prompt_ids, **kw): raise AssertionError("not reached")

    c = _C()
    c.opening_limit, c.continuation_limit = 1000, 400
    c._check_context([0] * 900, opening=True)                     # fits the prompt region
    with pytest.raises(ContextTooLarge, match="an observation came to 900"):
        c._check_context([0] * 900, opening=False)


def test_the_two_ceilings_come_from_the_mode():
    from vagen.harness.budget import context_limits

    b = _b(prompt_len=9000, response_len=8000, per_turn=512, max_turns=5,
           env_response=1360, compact_budget=1300, summary_budget=325)
    assert context_limits("concat", b) == (9000, 1360)
    assert context_limits("no_concat", b) == (9000, 9000)
    # Compaction opens conversations too, and one that opens at the budget summarises
    # after a single turn -- so its openings are bounded by the budget, not the region.
    assert context_limits("compact", b) == (1300, 1300)
