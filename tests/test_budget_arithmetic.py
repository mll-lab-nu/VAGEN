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
    with pytest.raises(BudgetError, match=r"10 x response_length_per_turn=1000 = 10000"):
        check("concat", _b(max_turns=10, per_turn=1000, response_len=8000))


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
    # per_turn raised past the summary budget so this exercises the halving rule and not
    # the one-generation rule, which would otherwise fire first and pass for the wrong reason.
    check("compact", _b(compact_budget=4000, summary_budget=2000, per_turn=4000))
    with pytest.raises(BudgetError, match="buys no turns"):
        check("compact", _b(compact_budget=4000, summary_budget=2001, per_turn=4000))


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


def test_the_derived_summary_budget_satisfies_the_rule_it_is_checked_against():
    """A default that its own checker rejects would fail every unconfigured run."""
    for m in (8, 100, 400, 4000, 40000):
        for per_turn in (64, 1024, 8000):
            k = default_summary_budget(m, per_turn)
            assert k >= 1
            check("compact", _b(compact_budget=m, summary_budget=k, per_turn=per_turn,
                                response_len=max(8000, per_turn), prompt_len=max(1000, m)))


# ------------------------------------------------------- the runtime companion
def _drive(h, budget_after_seed, per_turn_cost, turns):
    """Run the harness against a conversation of a known size, as the runner does."""
    h.begin({"role": "system", "content": "sys"}, {"role": "user", "content": "obs"})
    used = 0
    for t in range(turns):
        while True:
            used = budget_after_seed if h._conversation_id is None else used + per_turn_cost
            h.next_call()
            h.note_usage(used)

            class _R:
                text, conversation_id = f"act{t}", "c1"
            if h.accept(_R()) is not None:
                break
        h.add_observation({"role": "user", "content": "obs"})


def test_one_short_conversation_is_data_and_does_not_kill_the_run():
    """An unusually large observation should not end a training job."""
    h = CompactHarness(budget=100, summary_budget=25)
    # Opens at 90, so the first turn trips it; afterwards it opens at 10 and holds several.
    h.begin({"role": "system", "content": "s"}, {"role": "user", "content": "o"})
    for used, expect_summary in ((90, False), (140, True), (10, False), (50, False)):
        h.note_usage(used)
        call = h.next_call()

        class _R:
            text, conversation_id = "x", "c1"
        assert ("Summarise" in str(call.messages[0]["content"])) is expect_summary
        h.accept(_R())


def test_two_in_a_row_is_the_configuration_and_is_refused():
    h = CompactHarness(budget=100, summary_budget=25)
    with pytest.raises(CompactionMakesNoProgress, match="2 conversations in a row"):
        _drive(h, budget_after_seed=150, per_turn_cost=50, turns=6)


def test_the_summary_call_is_bounded_by_its_own_budget_not_the_turn_budget():
    """Without this the summary is capped by response_length_per_turn, which is the
    whole reason it could be longer than the budget it is compressing into."""
    h = CompactHarness(budget=100, summary_budget=25)
    h.begin({"role": "system", "content": "s"}, {"role": "user", "content": "o"})
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
    """A limit the harness sets and the runner drops is worse than no limit: it reads
    as bounded everywhere it is written down."""
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
            return Call([{"role": "user", "content": "x"}], None, sampling_params={"max_new_tokens": 7})

    class _Env:
        async def reset(self, seed): return {"obs_str": "o"}, {}
        async def system_prompt(self): return {"role": "system", "content": "s"}
        async def step(self, action, **kw): return {"obs_str": "o"}, 0.0, True, False, {}
        async def close(self): pass

    asyncio.run(run_episode(_Env(), _H(), _Client(), max_turns=1,
                            sampling_params={"temperature": 0.5}))
    assert seen == [{"temperature": 0.5, "max_new_tokens": 7}], seen
