"""Tests for the text-space context policies.

The claim being tested is that concat, no-concat and compaction differ only in when the
conversation id is dropped. So the tests read the sequence of calls a harness produces
and check the ids, not the prose.
"""

import inspect
import types

import pytest

from vagen.harness import CompactHarness, ConcatHarness, NoConcatHarness

SYS = {"role": "system", "content": "sys"}


def _resp(text, conversation_id):
    return types.SimpleNamespace(text=text, conversation_id=conversation_id)


def _drive(harness, turns):
    """Run `turns` exchanges against a stub client, returning every call made."""
    ids = iter(f"c{i}" for i in range(1, 99))
    harness.begin(SYS, {"role": "user", "content": "obs0"})
    calls = []
    for t in range(turns):
        while True:
            call = harness.next_call()
            calls.append(call)
            live = call.conversation_id or next(ids)
            if harness.accept(_resp(f"act{t}", live)) is not None:
                break          # forwarded; the environment would act now
        harness.add_observation({"role": "user", "content": f"obs{t + 1}"})
    return calls


# ------------------------------------------------------------------------ concat


def test_concat_keeps_one_conversation():
    calls = _drive(ConcatHarness(), turns=3)

    assert calls[0].conversation_id is None          # opens it
    assert [c.conversation_id for c in calls[1:]] == ["c1", "c1"]


def test_concat_sends_only_what_is_new():
    """The conversation already holds the history; resending it would duplicate every
    earlier turn in the prompt."""
    calls = _drive(ConcatHarness(), turns=2)

    assert [m["content"] for m in calls[0].messages] == ["sys", "obs0"]
    assert [m["content"] for m in calls[1].messages] == ["obs1"]


# --------------------------------------------------------------------- no-concat


def test_no_concat_opens_a_conversation_every_turn():
    """★ One training row per turn falls out of this, with no separate mechanism."""
    calls = _drive(NoConcatHarness(), turns=3)

    assert [c.conversation_id for c in calls] == [None, None, None]


def test_no_concat_shows_only_the_latest_observation():
    calls = _drive(NoConcatHarness(), turns=2)

    assert [m["content"] for m in calls[1].messages] == ["sys", "obs1"]


# ---------------------------------------------------------------------- compact


def test_compaction_does_not_fire_under_budget():
    h = CompactHarness(budget=100)
    h.note_usage(10)
    calls = _drive(h, turns=2)

    assert [c.conversation_id for c in calls] == [None, "c1"]


def test_compaction_asks_for_a_summary_then_starts_over():
    """★ The whole of compaction: one extra call whose answer the harness keeps, then a
    new conversation seeded with it."""
    h = CompactHarness(budget=5)
    h.begin(SYS, {"role": "user", "content": "obs0"})

    assert h.next_call().conversation_id is None
    h.accept(_resp("act0", "c1"))
    h.add_observation({"role": "user", "content": "obs1"})

    h.note_usage(99)
    summary_call = h.next_call()
    assert summary_call.conversation_id == "c1", "the model must see what it summarises"
    assert "Summarise" in summary_call.messages[0]["content"]

    assert h.accept(_resp("the story so far", "c1")) is None, "a summary is consumed, not forwarded"

    fresh = h.next_call()
    assert fresh.conversation_id is None, "compaction starts a new conversation"
    assert fresh.messages[0] is SYS
    assert any("the story so far" in m["content"] for m in fresh.messages)


def test_the_environment_never_acts_on_a_summary():
    """★ A forwarded summary would make the env step on text that is not an action, so
    the episode would advance by a turn that never happened. The turn that trips the
    budget costs two calls: a summary the harness keeps, and the action it forwards.

    Usage grows and resets the way a real conversation's does, rather than sitting over
    the budget forever -- pinned over it, this exercises a configuration that cannot buy
    a single turn, which the harness now refuses outright.
    """
    h = CompactHarness(budget=100)
    h.begin(SYS, {"role": "user", "content": "obs0"})
    ids = iter(f"c{i}" for i in range(1, 99))
    calls, actions, used = [], [], 0
    for t in range(6):
        while True:
            used = 0 if h._conversation_id is None else used + 40
            h.note_usage(used)
            call = h.next_call()
            calls.append(call)
            live = call.conversation_id or next(ids)
            action = h.accept(_resp(f"act{t}", live))
            if action is not None:
                actions.append(action)
                break
        h.add_observation({"role": "user", "content": f"obs{t + 1}"})

    assert len(actions) == 6, "every turn must still produce exactly one env action"
    assert len(calls) > 6, "no summary call was issued, so nothing was consumed"


def test_the_budget_resets_after_compacting():
    h = CompactHarness(budget=5)
    h.begin(SYS, {"role": "user", "content": "obs0"})
    h.next_call(); h.accept(_resp("a", "c1"))
    h.add_observation({"role": "user", "content": "obs1"})
    h.note_usage(99)
    h.next_call(); h.accept(_resp("summary", "c1"))
    h.next_call()

    assert h._used == 0, "a stale usage figure would compact again immediately"


# -------------------------------------------------------------------- the axis


@pytest.mark.parametrize(
    "cls,kwargs", [(ConcatHarness, {}), (NoConcatHarness, {}), (CompactHarness, {"budget": 10**9})]
)
def test_every_harness_is_text_only(cls, kwargs):
    """★ The property that lets one harness serve both training and a closed API: no
    tokenizer, no client, no env anywhere in it."""
    # Against the code, not the prose: a docstring may well explain what the client
    # does without the class ever touching one.
    import ast

    tree = ast.parse(inspect.getsource(cls))
    for node in ast.walk(tree):
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant):
            node.value.value = ""          # blank out docstrings and bare string exprs
    code = ast.unparse(tree)

    for forbidden in ("tokenizer", "token_ids", "client", "env.", "reward"):
        assert forbidden not in code, f"{cls.__name__} reaches for {forbidden}"


# ------------------------------------------------------------------- the registry


def test_every_policy_is_reachable_by_name():
    """★ Which policy a run uses is a config value, so adding one must not mean editing
    the agent loop -- the same reason advantage estimators are registered."""
    from vagen.harness import HARNESSES, build_harness

    assert set(HARNESSES) == {"concat", "no_concat", "compact"}
    assert isinstance(build_harness("concat"), ConcatHarness)
    assert isinstance(build_harness("compact", budget=10), CompactHarness)


def test_an_unknown_name_lists_what_is_available():
    """A bare KeyError leaves the reader guessing which spelling was wanted."""
    from vagen.harness import build_harness

    with pytest.raises(ValueError, match="choose from"):
        build_harness("concatenate")


def test_the_contract_holds_no_implementations():
    """★ core/ is the contract and harness/ the implementations; letting one policy
    live in the base module is how the split stops being real."""
    import inspect

    from vagen.harness._common import base as contract

    classes = [
        name
        for name, obj in vars(contract).items()
        if inspect.isclass(obj) and issubclass(obj, contract.BaseHarness) and obj is not contract.BaseHarness
    ]
    assert classes == [], f"{classes} belong in vagen/harness/"


# --------------------------------------- a custom harness reaches the training path too
def test_the_estimator_check_asks_a_custom_harness_whether_it_splits():
    """★ The check keys off `splits_episode_across_rows`, and it used to look the harness
    up by name only -- so an import-path harness came back None and was assumed to split,
    which makes the estimator guard reject default_gae on a concat-like policy. Now it
    resolves the class, and only a genuinely unresolvable name falls back to `True`."""
    from vagen.harness import ConcatHarness, resolve_harness

    class Mine(ConcatHarness):
        pass

    import sys
    sys.modules[__name__].Mine = Mine
    resolved = resolve_harness(f"{__name__}:Mine")
    assert resolved is Mine
    assert resolved.splits_episode_across_rows is ConcatHarness.splits_episode_across_rows
