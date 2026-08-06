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

from vagen.agent_loop.gym_loop import GymLoop


class _Row:
    conversation_id = "conv-1"
    response_spans = [(0, 1), (1, 2)]   # two turns inside this conversation
    prompt_ids = [1, 2, 3]
    response_ids = [4, 5]
    response_mask = [1, 1]
    logprobs = [0.0, 0.0]
    scores = [0.5, 0.5]


class _Client:
    def rows(self):
        return [_Row(), _Row()]

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
    return GymLoop._outputs(loop, _Client(), _Env(), _Result(), kwargs, "ep-abc")


def test_every_row_carries_the_whole_identity_chain():
    for out in _outputs():
        f = out.extra_fields
        for key in ("group_idx", "episode_id", "conversation_id", "turn_idx"):
            assert key in f, f"the loop stopped publishing {key}"
        assert f["episode_id"] == "ep-abc"
        assert f["group_idx"] == "g-1"


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
