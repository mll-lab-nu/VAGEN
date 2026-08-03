"""Unit tests for vagen/agent_loop/multi_output.py.

The two overrides are tested against a stubbed ``AgentLoopWorker`` base rather than a
live worker: instantiating a real one needs an LLM server, a tokenizer and a Ray actor,
and none of that is what changed. Stubbing the base also makes the assertions say
exactly what we claim -- that we flatten and stretch correctly and delegate everything
else untouched.

Not covered here (needs a real rollout, tracked in Phase 1): that the gym no-concat loop
actually emits one output per environment turn, and end-to-end shapes through Ray.
"""

import types
from unittest.mock import patch

import numpy as np
import pytest
from verl.experimental.agent_loop.agent_loop import AgentLoopManager, AgentLoopWorker

import vagen.agent_loop.multi_output as mo
from vagen.agent_loop.multi_output import (
    MultiOutputAgentLoopManager,
    MultiOutputAgentLoopWorker,
)


@pytest.fixture
def worker():
    """A worker with no __init__ run -- only our two overrides are under test."""
    return MultiOutputAgentLoopWorker.__new__(MultiOutputAgentLoopWorker)


@pytest.fixture
def base_postprocess():
    """Stub AgentLoopWorker._postprocess, recording what our override forwards."""
    seen = {}

    def fake(self, inputs, input_non_tensor_batch=None, validate=False):
        seen["inputs"] = inputs
        seen["non_tensor"] = input_non_tensor_batch
        seen["validate"] = validate
        return "batch"

    with patch.object(AgentLoopWorker, "_postprocess", fake):
        yield seen


@pytest.fixture
def base_agent_loop_postprocess():
    async def fake(self, output, validate, **kwargs):
        return f"padded({output})"

    with patch.object(AgentLoopWorker, "_agent_loop_postprocess", fake):
        yield


# ------------------------------------------------- _agent_loop_postprocess


@pytest.mark.asyncio
async def test_single_output_is_passed_straight_through(worker, base_agent_loop_postprocess):
    """Concat agent loops still return one output; they must be unaffected."""
    assert await worker._agent_loop_postprocess("a", validate=False) == "padded(a)"


@pytest.mark.asyncio
async def test_list_output_is_mapped_elementwise(worker, base_agent_loop_postprocess):
    """★ Also the guard for the zero-arg super() trap: written as a comprehension over
    ``super()._agent_loop_postprocess(...)`` this raises "super(): no arguments",
    because a comprehension's frame does not carry ``self``."""
    out = await worker._agent_loop_postprocess(["a", "b", "c"], validate=False)

    assert out == ["padded(a)", "padded(b)", "padded(c)"]


@pytest.mark.asyncio
async def test_empty_list_stays_empty(worker, base_agent_loop_postprocess):
    """A rollout that produced no turns is reported at _postprocess, not here."""
    assert await worker._agent_loop_postprocess([], validate=False) == []


# ------------------------------------------------------------- _postprocess


def test_groups_are_flattened_in_order(worker, base_postprocess):
    worker._postprocess([["a1", "a2"], ["b1"], ["c1", "c2", "c3"]])

    assert base_postprocess["inputs"] == ["a1", "a2", "b1", "c1", "c2", "c3"]


def test_bare_outputs_are_accepted(worker, base_postprocess):
    """Mixed single/list keeps one worker class usable for concat and no-concat."""
    worker._postprocess(["a", ["b1", "b2"]])

    assert base_postprocess["inputs"] == ["a", "b1", "b2"]


def test_input_columns_are_stretched_to_match_rows(worker, base_postprocess):
    """★ The core of the override. Two rollouts producing 2 and 3 turns must yield 5
    rows, and each row must keep the input column value of the rollout it came from."""
    worker._postprocess(
        [["a1", "a2"], ["b1", "b2", "b3"]],
        input_non_tensor_batch={"uid": np.array(["A", "B"]), "index": np.array([7, 9])},
    )

    assert base_postprocess["non_tensor"]["uid"].tolist() == ["A", "A", "B", "B", "B"]
    assert base_postprocess["non_tensor"]["index"].tolist() == [7, 7, 9, 9, 9]
    assert len(base_postprocess["inputs"]) == 5


def test_object_columns_survive_stretching(worker, base_postprocess):
    """raw_prompt / multi_modal_data are object arrays; np.repeat must copy the
    references rather than trying to broadcast their contents."""
    prompts = np.empty(2, dtype=object)
    prompts[:] = [{"role": "user"}, {"role": "system"}]

    worker._postprocess([["a1"], ["b1", "b2"]], input_non_tensor_batch={"raw_prompt": prompts})

    got = base_postprocess["non_tensor"]["raw_prompt"]
    assert [p["role"] for p in got] == ["user", "system", "system"]
    assert got[1] is got[2], "repeat must share the reference, not deep-copy"


def test_a_rollout_yielding_nothing_drops_only_its_own_rows(worker, base_postprocess):
    """An episode that ends before any turn completes must not shift the others'
    columns -- the classic off-by-one when counts and rows are tracked separately."""
    worker._postprocess(
        [["a1"], [], ["c1", "c2"]],
        input_non_tensor_batch={"uid": np.array(["A", "B", "C"])},
    )

    assert base_postprocess["non_tensor"]["uid"].tolist() == ["A", "C", "C"]


def test_validate_flag_is_forwarded(worker, base_postprocess):
    worker._postprocess([["a"]], validate=True)
    assert base_postprocess["validate"] is True


def test_absent_input_columns_stay_absent(worker, base_postprocess):
    worker._postprocess([["a"]])
    assert base_postprocess["non_tensor"] is None


def test_all_rollouts_empty_raises_with_the_cause(worker, base_postprocess):
    """verl would fail on inputs[0] with an IndexError; say what actually happened."""
    with pytest.raises(RuntimeError, match="zero outputs"):
        worker._postprocess([[], [], []])


def test_column_length_mismatch_is_loud(worker, base_postprocess):
    """3 rollouts but a 2-row input column means the batch was mis-assembled upstream;
    silently truncating would corrupt the group/traj indices."""
    with pytest.raises(ValueError):
        worker._postprocess([["a"], ["b"], ["c"]], input_non_tensor_batch={"uid": np.array(["A", "B"])})


# ----------------------------------------------------------------- manager


def test_manager_selects_the_multi_output_worker():
    """Asserts what we hand to ray.remote rather than poking at Ray's ActorClass
    internals, which are not a stable interface."""
    manager = MultiOutputAgentLoopManager.__new__(MultiOutputAgentLoopManager)
    with (
        patch.object(mo.ray, "remote", lambda cls: ("remote", cls)),
        patch.object(AgentLoopManager, "__init__", lambda self, *a, **k: None),
    ):
        manager.__init__(None, None)

    assert manager.agent_loop_workers_class == ("remote", MultiOutputAgentLoopWorker)


def test_base_default_worker_is_never_constructed():
    """★ Claiming the slot before delegating means verl's default ActorClass is never
    built. Asserting the *final* value instead would pass either way, since the base
    only assigns under `if not hasattr(...)` and nothing reads it until later."""
    built = []

    def guarded_init(self, *args, **kwargs):
        if not hasattr(self, "agent_loop_workers_class"):
            self.agent_loop_workers_class = mo.ray.remote(AgentLoopWorker)

    manager = MultiOutputAgentLoopManager.__new__(MultiOutputAgentLoopManager)
    with (
        patch.object(mo.ray, "remote", lambda cls: built.append(cls) or ("remote", cls)),
        patch.object(AgentLoopManager, "__init__", guarded_init),
    ):
        manager.__init__(None, None)

    assert built == [MultiOutputAgentLoopWorker], f"expected one ActorClass, got {built}"


def test_manager_is_reachable_by_fqn():
    """The config passes a string; a rename here would only surface at launch time."""
    from verl.utils.import_utils import load_class_from_fqn

    fqn = "vagen.agent_loop.multi_output.MultiOutputAgentLoopManager"
    assert load_class_from_fqn(fqn, "AgentLoopManager") is MultiOutputAgentLoopManager


# --------------------------------------------------------- index assignment


class _FakeProto:
    def __init__(self, uids):
        self.non_tensor_batch = {"uid": np.array(uids, dtype=object)}


def _manager(rollout_n):
    m = MultiOutputAgentLoopManager.__new__(MultiOutputAgentLoopManager)
    m.rollout_config = types.SimpleNamespace(n=rollout_n)
    return m


def test_indices_follow_the_interleaved_repeat():
    """★ gen_batch.repeat(interleave=True) lays rows out [A A B B]; traj_idx has to
    cycle within each group, since turn-level GAE groups on (group_idx, traj_idx)."""
    p = _FakeProto(["A", "A", "B", "B"])
    _manager(2)._vagen_assign_indices(p)

    assert p.non_tensor_batch["group_idx"].tolist() == ["A", "A", "B", "B"]
    assert p.non_tensor_batch["traj_idx"].tolist() == [0, 1, 0, 1]


def test_assignment_happens_before_the_chunking_call():
    """★ Placement guard. The base's first act is prompts.chunk(...); labelling per
    worker instead would restart traj_idx at 0 inside a split group."""
    seen = {}

    def fake(self, prompts):
        seen["traj_idx"] = prompts.non_tensor_batch.get("traj_idx")
        return "generated"

    m = _manager(2)
    p = _FakeProto(["A", "A"])
    with patch.object(AgentLoopManager, "generate_sequences", fake):
        assert m.generate_sequences(p) == "generated"

    assert seen["traj_idx"] is not None, "indices must be set before delegating"
    assert seen["traj_idx"].tolist() == [0, 1]
