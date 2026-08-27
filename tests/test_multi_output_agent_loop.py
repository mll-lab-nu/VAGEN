"""Unit tests for vagen/training/agent_loop/multi_output.py.

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

import vagen.training.agent_loop.multi_output as mo
from vagen.training.agent_loop.multi_output import (
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
        # verl returns a DataProto; the override reads non_tensor_batch off it.
        return types.SimpleNamespace(non_tensor_batch=dict(input_non_tensor_batch or {}))

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

    fqn = "vagen.training.agent_loop.multi_output.MultiOutputAgentLoopManager"
    assert load_class_from_fqn(fqn, "AgentLoopManager") is MultiOutputAgentLoopManager


# --------------------------------------------------------- index assignment


class _FakeProto:
    def __init__(self, uids, validate=False):
        self.non_tensor_batch = {"uid": np.array(uids, dtype=object)}
        self.meta_info = {"validate": validate} if validate else {}


def _manager(rollout_n, val_n=None):
    m = MultiOutputAgentLoopManager.__new__(MultiOutputAgentLoopManager)
    m.rollout_config = types.SimpleNamespace(
        n=rollout_n, val_kwargs=types.SimpleNamespace(n=val_n or rollout_n)
    )
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


# ------------------------------------------------------------ validation merging


def _proto(uids, validate):
    return _FakeProto(uids, validate=validate)


def test_training_output_is_not_merged():
    """Training wants one row per turn; folding them back would undo the whole point."""
    m = _manager(2)
    with patch.object(AgentLoopManager, "generate_sequences", lambda self, p: "raw"):
        assert m.generate_sequences(_proto(["A", "A"], validate=False)) == "raw"


def test_validation_output_is_merged_to_one_row_per_trajectory():
    """★ verl unpads the generated batch positionally against the padded input and then
    unions the two, so validation needs one output row per input row. Merging here is
    what lets _validate stay untouched."""
    seen = {}
    m = _manager(1)
    m._vagen_tokenizer = lambda: "tok"
    def record(out, prompts):
        seen["merged"] = (out, prompts)
        return "merged"

    m._vagen_merge_for_validation = record

    with patch.object(AgentLoopManager, "generate_sequences", lambda self, p: "per_turn_rows"):
        result = m.generate_sequences(_proto(["A"], validate=True))

    assert result == "merged"
    assert seen["merged"][0] == "per_turn_rows"


def test_validation_uses_the_validation_rollout_count():
    """val_kwargs.n usually differs from rollout.n; using the training one would
    mislabel traj_idx and merge the wrong rows together."""
    m = MultiOutputAgentLoopManager.__new__(MultiOutputAgentLoopManager)
    m.rollout_config = types.SimpleNamespace(n=1, val_kwargs=types.SimpleNamespace(n=2))
    p = _proto(["A", "A"], validate=True)

    m._vagen_assign_indices(p)

    assert p.non_tensor_batch["traj_idx"].tolist() == [0, 1]


def test_postprocess_reports_how_many_rows_each_rollout_produced(worker, base_postprocess, capsys):
    """★ Observability, not decoration. Every no-concat row reports num_turns=1 by
    construction and the batch dimensions look identical to a single-turn run, so a
    rollout that silently stops after one turn is invisible in the metrics -- which is
    exactly what happened before this line existed."""
    worker._postprocess([["a1", "a2", "a3"], ["b1"]])

    out = capsys.readouterr().out
    assert "2 rollouts produced 4 rows" in out, out


def test_index_columns_are_restored_when_verl_drops_them(worker):
    """★ verl's _postprocess skips input_non_tensor_batch entirely when streaming
    reward is enabled. The no-concat loop survives that because it emits group_idx per
    turn via extra_fields; the concat loop does not, so the trajectory estimators
    failed with KeyError under the concat layout alone."""

    class _Out:
        def __init__(self):
            self.non_tensor_batch = {}  # verl kept nothing

    with patch.object(AgentLoopWorker, "_postprocess", lambda self, *a, **k: _Out()):
        out = worker._postprocess(
            [["a"], ["b"]],
            input_non_tensor_batch={"group_idx": np.array(["A", "B"], dtype=object), "traj_idx": np.array([0, 0])},
        )

    assert out.non_tensor_batch["group_idx"].tolist() == ["A", "B"]
    assert out.non_tensor_batch["traj_idx"].tolist() == [0, 0]


def test_restored_indices_do_not_overwrite_per_turn_values(worker):
    """The no-concat loop's per-turn group_idx is the authoritative one; the input
    column is the pre-fan-out value and must not shadow it."""

    class _Out:
        def __init__(self):
            self.non_tensor_batch = {"group_idx": np.array(["kept"], dtype=object)}

    with patch.object(AgentLoopWorker, "_postprocess", lambda self, *a, **k: _Out()):
        out = worker._postprocess([["a"]], input_non_tensor_batch={"group_idx": np.array(["input"], dtype=object)})

    assert out.non_tensor_batch["group_idx"].tolist() == ["kept"]
