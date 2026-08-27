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

from vagen.rollout.client import BackendOutput, InferenceClient
from vagen.rollout.runner import run_episode
from vagen.harness import build_harness
from vagen.harness._common.budget import Budgets, check, context_limits
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
    # Mirrors gym_loop._build_harness. Leaving response_len out here is the same mistake
    # the production wiring made: the harness then accounts for nothing and the test
    # cannot see it.
    kw = dict(response_len=b.response_len, floor=b.per_turn)
    if mode == "compact":
        kw.update(budget=b.compact_budget, summary_budget=b.summary_budget,
                  summary_request_len=b.summary_request_len)
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


def test_the_response_region_holds_what_the_checks_let_through():
    """The region is the bound now, so a config that passes must produce rows inside it."""
    b = Budgets(prompt_len=32768, response_len=8192, per_turn=1024, max_turns=5,
                env_response=64, compact_budget=4000, summary_budget=256,
                summary_request_len=REQ)
    check("compact", b)
    c = episode("compact", b, system=600, obs=lambda i: 64, gen=lambda i: 1024)
    worst = max(len(r.response_ids) for r in c.rows())
    assert worst <= b.response_len, f"response region {worst} > max_response_length {b.response_len}"



def test_the_env_response_ceiling_is_what_bounds_a_continuation():
    """E's one job: the size an observation is cut to. compact's continuation ceiling
    used to be the compaction budget instead, which bounded nothing worth bounding."""
    b = Budgets(prompt_len=2000, response_len=16000, per_turn=1024, max_turns=3,
                env_response=200, compact_budget=4000, summary_budget=512,
                summary_request_len=REQ)
    check("compact", b)
    assert context_limits("compact", b)[1] <= b.env_response


def test_a_legal_episode_keeps_the_peak_inside_the_guarantee():
    """The guarantee holds only while a turn costs at most E + g, so the ceiling that
    enforces E is what makes it a guarantee rather than a hope. An observation over E is
    refused at the call rather than allowed to blow the bound silently."""
    from vagen.rollout.client import ContextTooLarge

    b = Budgets(prompt_len=2000, response_len=16000, per_turn=1024, max_turns=8,
                env_response=200, compact_budget=4000, summary_budget=512,
                summary_request_len=REQ)
    check("compact", b)
    bound = (b.compact_budget + b.env_response + b.per_turn
             + b.summary_request_len + b.summary_budget)

    c = episode("compact", b, system=600, obs=lambda i: 200, gen=lambda i: 1024)
    assert c.peak <= bound, f"peak {c.peak} > the guaranteed {bound}"

    # ★ And it holds through an observation that overruns E, because the ceiling now cuts
    # rather than refuses. That is the whole point of E being a ceiling: the guarantee is
    # meant to survive a badly behaved environment, not to be voided by one.
    over = episode("compact", b, system=600, obs=lambda i: 3000 if i == 1 else 64,
                   gen=lambda i: 1024)
    assert over.peak <= bound, f"peak {over.peak} > the guaranteed {bound} after a cut"


def test_the_defaults_pass_their_own_checks():
    """A default its own checker rejects fails every run that did not configure it. E
    is no longer among the values being solved for -- it is a flat ceiling -- but it
    still has to be a number the rest of the arithmetic tolerates."""
    from dataclasses import replace

    from vagen.harness._common.budget import default_env_response, default_summary_budget

    b = Budgets(prompt_len=1024, response_len=8192, per_turn=1024, max_turns=5,
                compact_budget=4096, summary_budget=default_summary_budget(4096, 1024),
                summary_request_len=REQ)
    check("compact", replace(b, env_response=default_env_response("compact", b)))

    c = Budgets(prompt_len=512, response_len=2048, per_turn=256, max_turns=2)
    check("concat", replace(c, env_response=default_env_response("concat", c)))


def test_a_region_that_cannot_hold_one_turn_says_so_once():
    """The band search is gone with compact_budget's demotion. What is left is the one
    thing no runtime can recover from: no room for a summary, its request and a
    generation, so every conversation closes before its first turn."""
    from vagen.harness._common.budget import BudgetError

    b = Budgets(prompt_len=512, response_len=1024, per_turn=512, max_turns=3,
                env_response=64, compact_budget=800, summary_budget=500,
                summary_request_len=REQ)
    with pytest.raises(BudgetError, match="no room to buy a turn"):
        check("compact", b)



def test_an_environment_is_closed_even_when_the_episode_raises():
    """Every guard added to this path raises mid-episode, and a leaked gym env is held
    for the rest of the batch."""
    from vagen.rollout.client import ContextTooLarge

    b = Budgets(prompt_len=100, response_len=8000, per_turn=64, max_turns=3,
                env_response=50)
    env = _Env(system=5000, obs=lambda i: 10)          # a system prompt over the ceiling
    c = _Client(lambda i: 8, 64)
    c.opening_limit, c.continuation_limit = context_limits("concat", b)
    with pytest.raises(ContextTooLarge):
        asyncio.run(run_episode(env, build_harness("concat"), c, max_turns=3))
    assert getattr(env, "closed", False), "the environment was left open"


# ------------------------------------------- an interrupted generation is not an answer
def test_an_interrupted_generation_is_re_asked_and_the_environment_never_sees_it():
    """A generation with no tokens is an interruption, so the turn has not happened yet.

    Retrying is safe precisely because it is empty: the environment steps on the action
    the call returns, so no action means no step, and the state being re-asked about is
    the state that was asked about. Under compaction the retry re-sends the summary that
    opened the conversation -- nothing downstream of it happened either.

    That premise was false before this. ``accept`` forwards ``response.text``, which is
    ``""`` and not ``None``, so the episode advanced a turn on an empty action, the
    environment moved, and the turn's reward had nowhere to land.
    """
    class _Interrupting(_Client):
        async def generate(self, prompt_ids, **kw):
            self.i += 1
            if self.i == 3:
                return BackendOutput(text="", token_ids=[])
            return BackendOutput(text="act", token_ids=[1] * 8)

    class _Watching(_Env):
        def __init__(self, *a):
            super().__init__(*a)
            self.actions = []

        async def step(self, action, **kw):
            self.actions.append(action)
            self.i += 1
            return {"obs_str": "o" * self.obs(self.i)}, float(self.i), False, False, {}

    env, c = _Watching(20, lambda i: 10), _Interrupting(lambda i: 8, 64)
    res = asyncio.run(run_episode(env, build_harness("no_concat"), c, max_turns=6))

    assert "" not in env.actions, f"the environment was stepped on an empty action: {env.actions}"
    assert len(env.actions) == 6, f"a turn went missing: {env.actions}"
    assert len(c.rows()) == 6, "a conversation was dropped, so the retry did not happen"
    assert sum(sum(r.scores) for r in c.rows()) == res.total_reward, (
        "reward the environment paid did not all reach the rows"
    )


def test_a_persistently_empty_generation_is_not_retried_forever():
    """Bounded, or an engine that has genuinely stopped answering hangs the rollout."""
    class _Never(_Client):
        async def generate(self, prompt_ids, **kw):
            self.i += 1
            return BackendOutput(text="", token_ids=[])

    c = _Never(lambda i: 8, 64)
    c.empty_generation_retries = 2
    asyncio.run(run_episode(_Env(20, lambda i: 10), build_harness("no_concat"), c, max_turns=1))
    assert c.i == 3, f"expected 1 attempt plus 2 retries, got {c.i}"


def test_asking_for_zero_tokens_is_refused_not_silently_maximised():
    """`or` treats 0 as absent, so the one case where the answer matters most inverts.

    A budget with no room left computes max_new_tokens=0. Under `x or limit` that becomes
    the *whole* limit -- the opposite of what was asked -- and the conversation overflows
    by a full generation. A negative passes through untouched.
    """
    from vagen.training.agent_loop.verl_client import VerlClient

    c = VerlClient.__new__(VerlClient)
    c.sampling_params, c.response_limit = {}, 512
    c.server_manager = c.tokenizer = c.processor = None
    c.mm_processor_kwargs, c.apply_chat_template_kwargs, c.request_id = {}, {}, "r"
    c._images, c._active = {}, None

    for asked in (0, -5):
        with pytest.raises(ValueError, match="no room left to generate"):
            asyncio.run(c.generate([1, 2, 3], sampling_params={"max_new_tokens": asked}))


def test_a_persistently_empty_generation_ends_the_episode_rather_than_faking_an_action():
    """After the retries, an empty response is an engine that has stopped answering.

    `accept` returns "" -- not None -- so the loop would take it for an action and step
    the environment on nothing. Measured before this: three env steps on '' and zero
    trainable rows, the whole episode gone from the batch while the environment moved
    three times.
    """
    class _Never(_Client):
        async def generate(self, prompt_ids, **kw):
            self.i += 1
            return BackendOutput(text="", token_ids=[])

    class _Watch(_Env):
        def __init__(self, *a):
            super().__init__(*a)
            self.actions = []

        async def step(self, action, **kw):
            self.actions.append(action)
            self.i += 1
            return {"obs_str": "o" * 10}, 1.0, False, False, {}

    c = _Never(lambda i: 8, 64)
    c.empty_generation_retries = 2
    env = _Watch(20, lambda i: 10)
    res = asyncio.run(run_episode(env, build_harness("no_concat"), c, max_turns=3))

    assert env.actions == [], f"the environment was stepped on an empty action: {env.actions}"
    assert res.truncated, "an episode cut short by a dead engine must not look terminal"


# ------------------------------------------------- the region reaches every harness
@pytest.mark.parametrize("mode", ["concat", "no_concat", "compact"])
def test_every_harness_is_given_the_room_it_has_to_work_in(mode):
    """Passing the region to compaction alone left the other two unbounded.

    `_left()` came back None, so nothing capped their generation and nothing stopped them
    when the region ran out. concat then filled past it and the batch-boundary cut took
    model turns with it -- with the reward on them. Measured before this: 62 of 182
    admitted concat configs lost reward.
    """
    from omegaconf import OmegaConf

    from vagen.training.agent_loop.gym_loop import GymLoop

    loop = GymLoop.__new__(GymLoop)
    loop.prompt_length, loop.response_length = 2048, 6144
    loop.tokenizer = type("T", (), {"encode": lambda self, t: [0] * 13,
                                    "apply_chat_template": lambda self, *a, **k: [0] * 34})()
    loop.apply_chat_template_kwargs = {}
    loop.config = OmegaConf.create({
        "trainer": {"harness": mode, "compact_budget": 1300, "compact_summary_budget": None},
        "actor_rollout_ref": {"rollout": {"max_model_len": None}},
    })

    harness, _ = GymLoop._build_harness(loop, per_turn=512, max_turns=20)
    assert harness.response_len == 6144, f"{mode} was not told the region"
    assert harness.floor == 512, f"{mode} was not told the floor"

    harness.begin({"role": "system", "content": "s"}, {"role": "user", "content": "o"})
    harness.note_room(6144 - 100, 50)
    assert harness._left() is not None, f"{mode} is not accounting at all"


def test_concat_stops_instead_of_overrunning_its_region():
    """The failure S1 describes, end to end: the reward the environment paid has to
    reach the row, and it cannot if the region overflows and the cut lands in a turn."""
    class _Paying(_Env):
        async def step(self, action, **kw):
            self.i += 1
            return {"obs_str": "o" * self.obs(self.i)}, 1.0, False, False, {}

    b = Budgets(prompt_len=500, response_len=400, per_turn=100, max_turns=8,
                env_response=50)
    c = episode("concat", b, system=20, obs=lambda i: 10, gen=lambda i: 100)
    rows = c.rows()
    assert rows, "the episode produced no trainable row"
    for r in rows:
        assert len(r.response_ids) <= b.response_len, (
            f"the response region reached {len(r.response_ids)} against {b.response_len}"
        )


def test_no_concat_measures_the_conversation_it_is_about_to_open():
    """Each turn opens a new one, so the room is a whole region every time. Measuring the
    conversation just left reports a region about to be discarded, and the episode stops
    after one turn."""
    from vagen.harness import build_harness

    h = build_harness("no_concat", response_len=200, floor=10)
    h.begin({"role": "system", "content": "s"}, {"role": "user", "content": "o"})
    h._conversation_id = "c1"           # a conversation has been used and left
    assert h.continues_conversation() is False
    h.note_room(0, 0)
    assert not h.exhausted()


def test_the_observation_is_actually_measured():
    """"Measured, not estimated" is half the design, and nothing tested it.

    A client whose `measure` returns 0 passed the entire suite: every decision then runs
    on a pending observation of size zero, which is the estimate-shaped failure the
    measurement replaced.
    """
    seen = []

    class _Counting(_Client):
        def measure(self, messages):
            n = super().measure(messages)
            seen.append(n)
            return n

    b = Budgets(prompt_len=500, response_len=4000, per_turn=100, max_turns=4,
                env_response=200)
    opening, continuation = context_limits("concat", b)
    c = _Counting(lambda i: 50, b.per_turn)
    c.opening_limit, c.continuation_limit = opening, continuation
    asyncio.run(run_episode(_Env(20, lambda i: 37), build_harness(
        "concat", response_len=b.response_len, floor=b.per_turn), c, max_turns=4))

    assert seen, "the observation was never measured"
    assert any(n > 0 for n in seen), (
        f"every measurement came back zero, so the harness decided on nothing: {seen}"
    )


def test_the_room_the_harness_sees_tracks_the_conversation_it_will_use():
    """The other half: response_len has to be read, and read for the right conversation."""
    rooms = []

    class _Recording(CompactHarness := __import__(
            "vagen.harness.compact", fromlist=["CompactHarness"]).CompactHarness):
        def note_room(self, response_len, obs_len):
            rooms.append((response_len, obs_len))
            super().note_room(response_len, obs_len)

    b = Budgets(prompt_len=500, response_len=3000, per_turn=100, max_turns=6,
                env_response=200, compact_budget=None, summary_budget=100,
                summary_request_len=REQ)
    opening, continuation = context_limits("compact", b)
    c = _Client(lambda i: 100, b.per_turn)
    c.opening_limit, c.continuation_limit = opening, continuation
    asyncio.run(run_episode(_Env(20, lambda i: 37), _Recording(
        summary_budget=100, summary_request_len=REQ,
        response_len=b.response_len, floor=b.per_turn), c, max_turns=6))

    assert rooms[0] == (0, 0), f"the opening call was charged something: {rooms[0]}"
    assert any(r > 0 for r, _ in rooms[1:]), (
        f"the spent region was never read as non-zero: {rooms}"
    )
    assert any(o > 0 for _, o in rooms[1:]), (
        f"the observation was never charged: {rooms}"
    )


# ------------------------------------------------------ two regressions, pinned
def test_an_unset_per_turn_budget_does_not_collapse_concat_to_one_turn():
    """`response_length_per_turn` is optional, and unset it falls back to the whole
    response length. Using that as the floor makes exhausted() true after a single token,
    so every episode stopped at turn one -- marked truncated, well-formed row, nothing
    reporting it. Erring small is the safe direction: too small only allows a squeezed
    generation, too large deletes the episode.
    """
    from omegaconf import OmegaConf

    from vagen.training.agent_loop.gym_loop import GymLoop

    def floor_for(per_turn, configured):
        loop = GymLoop.__new__(GymLoop)
        loop.prompt_length, loop.response_length = 2048, 8000
        loop.tokenizer = type("T", (), {"encode": lambda self, t: [0] * 13,
                                        "apply_chat_template": lambda self, *a, **k: [0] * 34})()
        loop.apply_chat_template_kwargs = {}
        loop.config = OmegaConf.create({
            "trainer": {"harness": "concat", "compact_budget": 1300,
                        "compact_summary_budget": None},
            "actor_rollout_ref": {"rollout": {"max_model_len": None}}})
        h, _ = GymLoop._build_harness(loop, per_turn=per_turn, max_turns=20,
                                      per_turn_configured=configured)
        return h.floor

    assert floor_for(512, True) == 512, "a configured per-turn budget should be the floor"
    unset = floor_for(8000, False)          # the fallback: per_turn == response_length
    assert unset < 8000, "the floor is the whole region, so one token exhausts it"
    assert unset <= 8000 // 4, f"the floor is {unset} of an 8000 region"

    h = build_harness("concat", response_len=8000, floor=unset)
    h.begin({"role": "system", "content": "s"}, {"role": "user", "content": "o"})
    h.note_room(100, 50)                    # one turn in
    assert not h.exhausted(), "the episode would stop after its first turn"


def test_the_compact_budget_lever_works_in_the_range_it_is_for():
    """compact_budget is a trigger, not a ceiling. Treating it as one bounded the opening
    call by it, so a small budget -- the whole point of the lever -- died on the episode's
    first call: the opening is the system prompt plus the first observation, and there is
    no summary in it yet to compact away."""
    from vagen.harness._common.budget import context_limits

    b = Budgets(prompt_len=2048, response_len=6144, per_turn=512, max_turns=20,
                env_response=1000, compact_budget=400, summary_budget=100,
                summary_request_len=REQ)
    check("compact", b)
    opening, _ = context_limits("compact", b)
    assert opening == 2048, f"the opening is bounded by the trigger, not the region: {opening}"
    assert opening > 589 + 58, "a Sokoban opening call would be refused before it ran"


def test_an_unusable_episode_costs_one_episode_not_the_batch():
    """verl's asyncio.gather has no return_exceptions, so anything escaping run_episode
    takes the whole rollout step with it. A too-large observation is evidence about this
    rollout; a bad configuration is evidence about every one, and still stops the run."""
    from omegaconf import OmegaConf

    from vagen.training.agent_loop.gym_loop import GymLoop
    from vagen.rollout.client import ContextTooLarge, EpisodeUnusable
    from vagen.harness._common.budget import BudgetError
    from vagen.models import ImagePlaceholderMismatch

    assert issubclass(ContextTooLarge, EpisodeUnusable)
    from vagen.harness.compact import CompactionMakesNoProgress
    assert issubclass(CompactionMakesNoProgress, EpisodeUnusable)
    # These say something about the configuration, not about one rollout.
    assert not issubclass(BudgetError, EpisodeUnusable)
    assert not issubclass(ImagePlaceholderMismatch, EpisodeUnusable)

    import inspect
    src = inspect.getsource(GymLoop.run)
    assert "except EpisodeUnusable" in src, "run() no longer isolates an unusable episode"
    assert "return []" in src, "a dropped episode must yield no rows rather than raise"


def test_the_opening_ceiling_is_checked_against_what_the_engine_actually_ran():
    """Measuring before adoption let a longer sequence through.

    The engine expands multimodal placeholders its own way, and `adopt_prompt` takes its
    version -- so a ceiling checked on our render passes while the batch boundary sees
    something bigger. That is the end-of-episode surprise the per-call ceilings exist to
    replace.
    """
    from vagen.rollout.client import BackendOutput, ContextTooLarge, InferenceClient

    class _Expanding(InferenceClient):
        """Renders 10 tokens; the engine says it ran 40."""

        tokenizer = None

        def encode(self, messages): return [0] * 10

        async def generate(self, prompt_ids, **kw):
            return BackendOutput(text="a", token_ids=[1],
                                 prompt_token_ids=[0] * 40)

    c = _Expanding()
    c.opening_limit, c.continuation_limit = 20, 20
    with pytest.raises(ContextTooLarge, match="came to 40"):
        asyncio.run(c.send([{"role": "user", "content": "x"}]))
