"""The accounting held against episodes that actually run.

``test_budget_arithmetic`` checks the relations against numbers. This drives the real
``run_episode`` / harness / client with a backend where one character is one token, so
every trigger, ceiling and region decision below is production code and only the *sizes*
are the test's. That is the difference that matters here: each of these passed the
arithmetic and then failed the episode.
"""

from __future__ import annotations

import asyncio

import pytest

from vagen.core.client import BackendOutput, InferenceClient
from vagen.core.runner import run_episode
from vagen.harness import build_harness
from vagen.harness.budget import Budgets, check, context_limits
from vagen.harness.compact import CompactHarness

#: The harness sends this string, so a Budgets that declares a different length has the
#: continuation ceiling reject a request the harness itself issued.
REQ = len(CompactHarness.SUMMARY_REQUEST)


def _text(m):
    c = m.get("content")
    return c if isinstance(c, str) else "".join(p.get("text", "") for p in c)


class _Client(InferenceClient):
    """One token per character, and it remembers the largest conversation it held."""

    tokenizer = None

    def __init__(self, gen, cap):
        super().__init__()
        self.gen, self.cap, self.i, self.peak = gen, cap, 0, 0

    def encode(self, msgs):
        return [0] * sum(len(_text(m)) for m in msgs)

    async def generate(self, prompt_ids, **kw):
        sp = kw.get("sampling_params") or {}
        n = min(self.gen(self.i), min(sp.get("max_new_tokens") or self.cap, self.cap))
        self.i += 1
        return BackendOutput(text="x" * n, token_ids=[1] * n)

    async def send(self, msgs, cid=None, **kw):
        r = await super().send(msgs, cid, **kw)
        self.peak = max(self.peak, len(self._conversations[r.conversation_id].token_ids))
        return r


class _Env:
    def __init__(self, system, obs):
        self.system, self.obs, self.i = system, obs, 0

    async def reset(self, seed=None):
        return {"obs_str": "o" * self.obs(0)}, {}

    async def system_prompt(self):
        return {"role": "system", "content": "s" * self.system}

    async def step(self, action, **kw):
        self.i += 1
        return {"obs_str": "o" * self.obs(self.i)}, 0.0, False, False, {}

    async def close(self):
        self.closed = True


def episode(mode, b, *, system, obs, gen):
    """Run one episode under ``mode``, wired exactly as ``gym_loop`` wires it."""
    opening, continuation = context_limits(mode, b)
    kw = (dict(budget=b.compact_budget, summary_budget=b.summary_budget)
          if mode == "compact" else {})
    c = _Client(gen, b.per_turn)
    c.opening_limit, c.continuation_limit = opening, continuation
    asyncio.run(run_episode(_Env(system, obs), build_harness(mode, **kw), c,
                            max_turns=b.max_turns))
    return c


# Sokoban vision as measured: S=589, observations 44-58, real turns about 80 against a
# configured ceiling of 512.
SOKOBAN = Budgets(prompt_len=4000, response_len=8000, per_turn=512, max_turns=12,
                  env_response=58, compact_budget=1300, summary_budget=325,
                  summary_request_len=REQ)


def test_one_long_but_legal_generation_does_not_kill_a_healthy_run():
    """A turn inside the configured ceiling is not a misconfiguration.

    The estimate exists to predict the next turn. Kept as a running maximum it stops
    predicting and starts remembering: one 512-token response -- legal, it is exactly
    response_length_per_turn -- sets the estimate for the rest of the episode, every
    later conversation is cut off after one turn, and the run dies claiming the budget
    cannot hold a turn that costs 825 against a budget of 1300.
    """
    check("compact", SOKOBAN)
    episode("compact", SOKOBAN, system=589, obs=lambda i: 58, gen=lambda i: 80)
    episode("compact", SOKOBAN, system=589, obs=lambda i: 58,
            gen=lambda i: 512 if i == 1 else 80)


def test_the_response_region_alone_fits_when_the_checks_pass():
    """``peak <= n_p + n_r`` does not imply the row fits: almost all of a conversation
    lands in the response region, and a large n_p hides that rather than absorbing it.

    Both halves matter. The first config is refused now and produced 5376-token rows
    against a 2048 region before; the second passes, and the rows it produces have to be
    inside the region the check promised.
    """
    from vagen.harness.budget import BudgetError

    hidden = Budgets(prompt_len=32768, response_len=2048, per_turn=1024, max_turns=5,
                     env_response=64, compact_budget=20000, summary_budget=256,
                     summary_request_len=REQ)
    with pytest.raises(BudgetError, match="lands in the response region"):
        check("compact", hidden)

    b = Budgets(prompt_len=32768, response_len=8192, per_turn=1024, max_turns=5,
                env_response=64, compact_budget=4000, summary_budget=256,
                summary_request_len=REQ)
    check("compact", b)
    c = episode("compact", b, system=600, obs=lambda i: 64, gen=lambda i: 1024)
    worst = max(len(r.response_ids) for r in c.rows())
    assert worst <= b.response_len, f"response region {worst} > max_response_length {b.response_len}"


def test_env_response_length_is_enforced_in_the_mode_whose_arithmetic_needs_it():
    """compact's continuation ceiling was the budget, so E bounded nothing there -- in
    the one mode where every relation is written in terms of it."""
    b = Budgets(prompt_len=2000, response_len=16000, per_turn=1024, max_turns=3,
                env_response=200, compact_budget=4000, summary_budget=512,
                summary_request_len=REQ)
    check("compact", b)
    assert context_limits("compact", b)[1] <= b.env_response


def test_a_legal_episode_keeps_the_peak_inside_the_guarantee():
    """The guarantee holds only while a turn costs at most E + g, so the ceiling that
    enforces E is what makes it a guarantee rather than a hope. An observation over E is
    refused at the call rather than allowed to blow the bound silently."""
    from vagen.core.client import ContextTooLarge

    b = Budgets(prompt_len=2000, response_len=16000, per_turn=1024, max_turns=8,
                env_response=200, compact_budget=4000, summary_budget=512,
                summary_request_len=REQ)
    check("compact", b)
    bound = (b.compact_budget + b.env_response + b.per_turn
             + b.summary_request_len + b.summary_budget)

    c = episode("compact", b, system=600, obs=lambda i: 200, gen=lambda i: 1024)
    assert c.peak <= bound, f"peak {c.peak} > the guaranteed {bound}"

    with pytest.raises(ContextTooLarge, match="an observation came to 3000"):
        episode("compact", b, system=600, obs=lambda i: 3000 if i == 1 else 64,
                gen=lambda i: 1024)


def test_the_defaults_are_the_largest_values_that_pass():
    """A default its own checker rejects fails every run that did not configure it."""
    from dataclasses import replace

    from vagen.harness.budget import default_env_response, default_summary_budget

    b = Budgets(prompt_len=1024, response_len=8192, per_turn=1024, max_turns=5,
                compact_budget=4096, summary_budget=default_summary_budget(4096, 1024),
                summary_request_len=REQ)
    check("compact", replace(b, env_response=default_env_response("compact", b)))

    c = Budgets(prompt_len=512, response_len=2048, per_turn=256, max_turns=2)
    check("concat", replace(c, env_response=default_env_response("concat", c)))


def test_an_impossible_band_says_so_once_rather_than_sending_you_round_it():
    """Reporting only the ceiling when the floor is above it gives advice that fails the
    next check, and the next: 800 is "more than half", 1000 is "at most 445", 445 is
    "more than half" again."""
    from vagen.harness.budget import BudgetError

    b = Budgets(prompt_len=512, response_len=1024, per_turn=512, max_turns=3,
                env_response=64, compact_budget=800, summary_budget=500,
                summary_request_len=REQ)
    with pytest.raises(BudgetError, match="no compact_budget"):
        check("compact", b)


def test_an_environment_is_closed_even_when_the_episode_raises():
    """Every guard added to this path raises mid-episode, and a leaked gym env is held
    for the rest of the batch."""
    from vagen.core.client import ContextTooLarge

    b = Budgets(prompt_len=100, response_len=8000, per_turn=64, max_turns=3,
                env_response=50)
    env = _Env(system=5000, obs=lambda i: 10)          # a system prompt over the ceiling
    c = _Client(lambda i: 8, 64)
    c.opening_limit, c.continuation_limit = context_limits("concat", b)
    with pytest.raises(ContextTooLarge):
        asyncio.run(run_episode(env, build_harness("concat"), c, max_turns=3))
    assert getattr(env, "closed", False), "the environment was left open"
