"""The loop must stamp the identity chain on every row it emits.

    group_id > episode_id > conversation_id > turn_id

Each level is a strict child of the one above, and the episode log needs all of them to
put an episode back together. Missing any one does not fail: the log just groups wrongly
and reports every episode as a single turn.

This calls _outputs for real. A NameError in it survived the whole suite once, because
nothing here executed the method -- only asserted things about its source text.
"""

from __future__ import annotations

import pytest

from vagen.training.agent_loop.gym_loop import GymLoop

class _NoCompaction:
    """A harness that never summarised. `_outputs` asks it which conversations ended at a
    compaction seam rather than because the environment stepped; only CompactHarness ever
    answers non-empty."""

    summarised_conversations: set = set()




class _Row:
    response_spans = [(0, 1), (1, 2)]   # two turns inside this conversation
    prompt_ids = [1, 2, 3]
    response_ids = [4, 5]
    response_mask = [1, 1]
    logprobs = [0.0, 0.0]
    scores = [0.5, 0.5]

    def __init__(self, ordinal):
        # Carried on the row because it is decided when the conversation is opened. Read
        # off the position in rows() instead, a conversation the model never spoke in --
        # dropped there -- would renumber every one after it, with no hole to notice.
        self.ordinal = ordinal
        self.conversation_id = f"conv-{ordinal + 1}"


class _Client:
    def rows(self):
        return [_Row(0), _Row(1)]

    def images(self, conversation_id):
        return []


class _Env:
    success = True
    state_scores = {"state_estimation": 0.1, "format": 0.0}


class _Result:
    turns = 4


def _outputs():
    loop = GymLoop.__new__(GymLoop)
    loop.prompt_length = 100
    loop.response_length = 100
    kwargs = {"group_idx": "g-1", "traj_idx": 0}
    return GymLoop._outputs(loop, _Client(), _Env(), _Result(), kwargs, "ep-abc", _NoCompaction())


def test_every_row_carries_the_whole_identity_chain():
    for out in _outputs():
        f = out.extra_fields
        for key in ("group_idx", "episode_id", "conversation_id", "turn_idx"):
            assert key in f, f"the loop stopped publishing {key}"
        assert f["episode_id"] == "ep-abc"
        assert f["group_idx"] == "g-1"


def test_every_per_row_column_the_loop_publishes_survives_the_trip_to_the_trainer():
    """★ Publishing a per-row column is only half of it.

    verl drops ``input_non_tensor_batch`` wholesale under streaming reward, so anything
    the loop emits has to be named in ``INDEX_COLUMNS`` or ``ROW_COLUMNS`` to be put back.
    A column that is emitted but not listed simply is not there by the time an estimator
    reads it -- and every reader of these columns treats absence as a legitimate default
    ("no seams", "one conversation"), so nothing raises and the estimator quietly runs
    the version of itself that the column existed to correct.

    Listing them one by one is what let ``ends_with_summary`` be added to the loop and
    forgotten here, so this asks the loop what it publishes rather than being told.
    """
    from vagen.training.agent_loop.multi_output import MultiOutputAgentLoopWorker as _M

    carried = set(_M.INDEX_COLUMNS) | set(_M.ROW_COLUMNS)
    # Not per-row identity: these are either per-rollout (restored from the input batch),
    # consumed inside the loop, or turned into tensors before the trip.
    not_identity = {"traj_idx", "reward_extra_info", "image_data", "per_token_reward",
                    "last_turn", "response_mask", "logprobs", "metrics"}

    published = set()
    for out in _outputs():
        published |= set(out.extra_fields)

    missing = published - carried - not_identity
    assert not missing, (
        f"the loop publishes {sorted(missing)} but nothing restores them after verl "
        f"drops the non-tensor batch -- add them to ROW_COLUMNS, or to this test's "
        f"`not_identity` if they genuinely do not need to survive"
    )


def test_conversations_are_numbered_from_zero_in_order():
    """group / episode ids only identify; conversations are a sequence and read as
    0,1,2. Enumerating rows and calling the result turn_idx numbered conversations and
    labelled them turns -- only the same thing under no_concat."""
    ids = [o.extra_fields["conversation_id"] for o in _outputs()]
    assert ids == list(range(len(ids))), ids


def test_each_conversation_reports_the_turns_inside_it():
    for out in _outputs():
        assert out.extra_fields["response_spans"] == [(0, 1), (1, 2)]


def test_the_real_turn_count_travels_with_the_rows():
    """num_turns is 1 per row by construction; concat puts a whole episode in one row."""
    for out in _outputs():
        assert out.extra_fields["episode_turns"] == 4
        assert out.num_turns == 1


def test_one_episode_id_for_all_rows_of_an_episode():
    ids = {o.extra_fields["episode_id"] for o in _outputs()}
    assert len(ids) == 1, "rows of one episode disagree about which episode they are"


def test_last_turn_is_marked_once():
    flags = [o.extra_fields["last_turn"] for o in _outputs()]
    assert flags == [False, True], flags


def test_a_dropped_conversation_does_not_renumber_the_ones_after_it():
    """The id says which conversation this is, so it cannot be the array index.

    A conversation the model never spoke in is dropped from ``rows()`` -- correctly, it
    carries no gradient. Numbering the survivors by position then moves everything after
    the gap down by one, and nothing reveals it: the ids stay contiguous, so there is no
    hole to notice. Under no_concat the id *is* the environment step, so turn n+1's
    behaviour would be recorded against turn n.

    Not reachable on today's configuration -- a row is only empty when a generation
    returns no tokens, which needs an abort, and fully-async absorbs those a layer below
    the agent loop (FullyAsyncLLMServerClient resumes from prompt_ids + token_ids). This
    pins the invariant rather than a live failure.
    """
    class _Dropping(_Client):
        def rows(self):
            # c2 produced nothing and was dropped; c1, c3, c4 survive.
            return [_Row(0), _Row(2), _Row(3)]

    loop = GymLoop.__new__(GymLoop)
    loop.prompt_length = loop.response_length = 100
    outs = GymLoop._outputs(loop, _Dropping(), _Env(), _Result(),
                            {"group_idx": "g-1", "traj_idx": 0}, "ep-abc", _NoCompaction())
    ids = [o.extra_fields["conversation_id"] for o in outs]
    assert ids == [0, 2, 3], f"the gap was closed up and everything after it renumbered: {ids}"
    # ★ And last_turn has to survive the same gap. Comparing the ordinal against
    # `len(rows) - 1` marks ordinal 2 -- the middle row -- and leaves the real last one
    # unflagged. The whole suite passed with that bug in place.
    flags = [o.extra_fields["last_turn"] for o in outs]
    assert flags == [False, False, True], f"last_turn followed the gap: {flags}"
