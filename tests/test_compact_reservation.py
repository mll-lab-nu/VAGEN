"""Compaction against the region it actually has to fit in.

The old rule compared an estimate of the last turn's cost against a number chosen next to
the region rather than derived from it. This one measures: how much of the response region
the conversation has spent, and how big the observation about to be sent is. The invariant
it maintains is that a conversation closing on a summary still fits its row:

    resp + |req| + summary  <=  max_response_length

``|req|`` is not optional. The summary *request* is a user message into the same
conversation, so it lands in the same response region; reserving only the summary
overflows by the request every single time.
"""

from __future__ import annotations

import random

import pytest

from vagen.harness import CompactHarness
from vagen.harness.compact import CompactionMakesNoProgress

SYS = {"role": "system", "content": "s"}
OBS = {"role": "user", "content": "o"}
REQ = 39            # what gym_loop measures for the request plus the summary's wrapper


class _R:
    text, conversation_id = "act", "c"


def simulate(*, n_r, g, k, req=REQ, reserve_req=None, floor=None, budget=None,
             system=589, obs=58, gen=106, turns=20, obs_at=None, gen_at=None):
    """Drive the real harness over a conversation of known sizes.

    Only the *sizes* are the test's; every decision is the harness's.
    Returns (peak response region, turns per conversation, compactions).
    """
    # `req` is what the request really costs; `reserve_req` is what the harness is told
    # to set aside for it. They differ only in the test that shows why they must not.
    h = CompactHarness(budget=budget, summary_budget=k, response_len=n_r,
                       summary_request_len=req if reserve_req is None else reserve_req,
                       floor=g if floor is None else floor)
    h.begin(SYS, OBS)
    resp, seeded, peak, here, per_conv, sums = 0, False, 0, 0, [], 0

    for t in range(turns):
        while True:
            o = 0 if h._conversation_id is None else (obs_at(t) if obs_at else obs)
            h.note_room(resp, o)
            call = h.next_call()
            summary = "Summarise" in str(call.messages[0]["content"])
            limit = (call.sampling_params or {}).get("max_new_tokens")

            if call.conversation_id is None:            # opens: goes to the prompt region
                resp = min(gen_at(t) if gen_at else gen, limit or 10**9)
            elif summary:
                resp += req + k
            else:
                resp += o + min(gen_at(t) if gen_at else gen, limit or 10**9)
            peak = max(peak, resp)
            h.note_usage(system + (k if seeded else 0) + o + resp)

            if h.accept(_R()) is None:                  # the summary was consumed
                per_conv.append(here)
                here, sums, seeded, resp = 0, sums + 1, True, 0
            else:
                here += 1
                break
        h.add_observation(OBS)
    per_conv.append(here)
    return peak, per_conv, sums


# Sokoban vision, measured from a real run.
SOKOBAN = dict(n_r=6144, g=512, k=325, system=589, obs=58, gen=106)


# ------------------------------------------------------------------- the invariant
def test_the_summary_always_fits_the_row_it_lands_in():
    peak, _, sums = simulate(turns=200, **SOKOBAN)
    assert sums > 0, "nothing compacted, so the invariant was never exercised"
    assert peak <= SOKOBAN["n_r"], f"the response region reached {peak} against {SOKOBAN['n_r']}"


def test_reserving_only_the_summary_overflows_by_the_request_every_time():
    """The failure the |req| half of the reserve exists to prevent, made visible.

    A harness told the request costs nothing closes each conversation `req` tokens over
    the region. It is deterministic, not occasional.
    """
    # floor=1 so max_new_tokens is clamped to whatever room is left and the conversation
    # fills exactly up to the reserve boundary. That is where the two answers separate:
    # with a floor at g the turn granularity is coarser than the request and the
    # difference usually falls between two turns.
    common = dict(n_r=6144, g=512, k=325, system=589, obs=58, gen=512, floor=1, turns=200)
    honest = simulate(**common)[0]
    blind = simulate(reserve_req=0, **common)[0]
    assert honest <= 6144, f"the honest reserve did not hold the invariant: {honest}"
    assert blind > 6144, (
        f"reserving nothing for the request stayed inside the region ({blind}), so this "
        f"test proves nothing about why |req| is in the reserve"
    )
    assert blind - 6144 <= REQ, "the overflow should be exactly the unreserved request"


@pytest.mark.parametrize("seed", range(25))
def test_the_invariant_holds_under_random_sizes(seed):
    """Spiky observations and generations, which is what an environment actually gives."""
    rng = random.Random(seed)
    n_r = rng.choice([1024, 2048, 6144, 16384])
    g = rng.randint(32, max(64, n_r // 4))
    k = rng.randint(1, g)
    biggest_obs = 600
    # Two turns, not one: a region that fits exactly one turn per conversation is the
    # degenerate case the runtime guard refuses, and it has nothing to say about the
    # invariant. The reserve plus two full turns is what "compaction buys turns" means.
    if k + REQ + 2 * (g + biggest_obs) > n_r:
        pytest.skip("no room for two turns; the guard refuses this before it runs")
    peak, _, _ = simulate(
        n_r=n_r, g=g, k=k, turns=60, system=rng.randint(10, 800),
        obs_at=lambda t, rng=rng: rng.choice([10, 60, 200, biggest_obs]),
        gen_at=lambda t, rng=rng: rng.randint(1, g))
    assert peak <= n_r, f"n_r={n_r} g={g} k={k}: response region reached {peak}"


# ------------------------------------------------------------------------ termination
def test_a_conversation_is_closed_rather_than_squeezed_below_the_floor():
    """A five-token generation is not an action, it is half an `<answer>` that the
    environment then parses.

    Asserting `max_new_tokens() >= floor` would prove nothing -- it is literally
    `max(floor, left)`. What has to hold is that the harness *stops asking* once the room
    is under the floor, so the clamp is never the thing keeping the promise.
    """
    h = CompactHarness(budget=None, summary_budget=100, response_len=1000,
                       summary_request_len=REQ, floor=200)
    h.begin(SYS, OBS)
    room = 1000 - 100 - REQ
    for resp, should_compact in ((0, False), (room - 200 - 58, False),
                                 (room - 199 - 58, True), (900, True)):
        h.note_room(resp, 58)
        assert h._should_compact() is should_compact, (
            f"resp={resp}: left={h._left()} against floor={h.floor}"
        )


def test_the_opening_call_charges_no_observation():
    """The system prompt and the opening observation become the *prompt* region, so
    charging them against the response budget over-reserves by most of it."""
    h = CompactHarness(budget=None, summary_budget=100, response_len=1000,
                       summary_request_len=REQ, floor=1)
    h.begin(SYS, OBS)
    h.note_room(0, 0)
    assert h.max_new_tokens() == 1000 - 100 - REQ


def test_an_episode_always_makes_progress():
    """Every turn must produce exactly one environment action, however tight the numbers."""
    for n_r, g, k in ((1024, 256, 128), (600, 100, 50), (20000, 512, 325)):
        if k + REQ + g > n_r:
            continue
        _, per_conv, _ = simulate(n_r=n_r, g=g, k=k, turns=30, system=589, obs=58, gen=g)
        assert sum(per_conv) == 30, f"n_r={n_r} g={g} k={k}: {sum(per_conv)} of 30 turns"


def test_a_configuration_that_cannot_buy_turns_is_reported():
    # 900 - 239 reserve = 661 of room. One 350-token generation plus a 58-token
    # observation leaves 253, under the 350 floor -- so every conversation closes after
    # exactly one turn, which is no_concat paying for a summary as well.
    with pytest.raises(CompactionMakesNoProgress, match="single turn"):
        simulate(n_r=900, g=350, k=200, turns=20, system=589, obs=58, gen=350)


# --------------------------------------------------------- the second trigger is a lever
def test_without_the_optional_budget_a_wide_region_never_compacts():
    """Why compact_budget survives. 'Compact when the next turn does not fit' only fires
    when the region is nearly full, so on Sokoban's 6144-token region a 20-turn episode
    compacts zero times -- byte-identical to concat, which is the symptom that started
    this. Narrowing the region would fix it and would also narrow the training row."""
    assert simulate(turns=20, **SOKOBAN)[2] == 0


def test_the_optional_budget_restores_it_without_touching_the_region():
    peak, per_conv, sums = simulate(turns=20, budget=1300, **SOKOBAN)
    assert sums > 0, "the second trigger did not fire"
    assert peak <= SOKOBAN["n_r"], "the second trigger broke the invariant"
    assert max(per_conv) > 1, "conversations held only one turn, which is no_concat"


def test_the_two_triggers_cannot_contradict_each_other():
    """Whichever fires first wins, and the region one is always the backstop -- so a
    compact_budget set absurdly high cannot push a conversation past its row.

    The budget is set past anything reachable *and* the region is narrowed, so the region
    trigger is demonstrably the one doing the work. Left at Sokoban's width this drove the
    same path as the plain invariant test five above.
    """
    narrow = {**SOKOBAN, "n_r": 1600}
    peak, per_conv, sums = simulate(turns=200, budget=10**9, **narrow)
    assert sums > 0, "the region trigger never fired, so the backstop is untested"
    assert peak <= narrow["n_r"], f"peak {peak} escaped the region"
    assert max(per_conv) > 1, "conversations held one turn, so this is no_concat"
