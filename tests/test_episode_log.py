"""An episode has to survive being logged as one thing.

A row is one model call. An episode is several, and once the context is compacted it is
several conversations -- so the naive table shows one trajectory as unrelated rows with
nothing saying they belong together or in what order. These pin the regrouping, the
ordering, and that the frames the agent saw actually make it into the cell.
"""

from __future__ import annotations

import pytest

from vagen.utils.episode_log import episode_html, episode_rows, group_turns

PIL = pytest.importorskip("PIL.Image")


def _frame(color=(10, 20, 30)):
    return PIL.new("RGB", (8, 8), color)


def _row(g, t, turn, out, conversation="c0", images=None, score=None, success=None):
    return {
        "group_idx": g, "traj_idx": t, "turn_idx": turn, "conversation_id": conversation,
        "input": "PROMPT", "output": out, "images": images or [], "score": score,
        "traj_success": success,
    }


def test_turns_of_one_trajectory_group_together():
    rows = [_row(0, 0, 0, "a"), _row(0, 1, 0, "b"), _row(0, 0, 1, "c")]
    groups = group_turns(rows)
    assert set(groups) == {(0, 0), (0, 1)}
    assert [t["output"] for t in groups[(0, 0)]] == ["a", "c"]


def test_turns_are_ordered_even_when_rows_arrive_shuffled():
    """Row order out of a batch is not turn order, and a scrambled transcript is worse
    than no transcript -- it reads as a coherent episode that never happened."""
    rows = [_row(0, 0, 2, "third"), _row(0, 0, 0, "first"), _row(0, 0, 1, "second")]
    assert [t["output"] for t in group_turns(rows)[(0, 0)]] == ["first", "second", "third"]


def test_every_turns_text_reaches_the_cell():
    html = episode_html(group_turns([_row(0, 0, i, f"turn-body-{i}") for i in range(4)])[(0, 0)])
    for i in range(4):
        assert f"turn-body-{i}" in html


def test_frames_are_embedded_not_dropped():
    html = episode_html([_row(0, 0, 0, "x", images=[_frame(), _frame((99, 0, 0))])])
    assert html.count("data:image/png;base64,") == 2


def test_a_compaction_is_visible_as_a_seam():
    """Without the marker the transcript reads as a model that inexplicably forgot."""
    turns = group_turns([
        _row(0, 0, 0, "before", conversation="c0"),
        _row(0, 0, 1, "after", conversation="c1"),
    ])[(0, 0)]
    html = episode_html(turns)
    assert "compacted" in html
    assert html.index("before") < html.index("compacted") < html.index("after")


def test_no_seam_when_the_conversation_never_restarted():
    html = episode_html(group_turns([_row(0, 0, 0, "a"), _row(0, 0, 1, "b")])[(0, 0)])
    assert "compacted" not in html


def test_summary_counts_turns_and_conversations():
    rows = [
        _row(0, 0, 0, "a", conversation="c0", score=0.5, success=0.0),
        _row(0, 0, 1, "b", conversation="c1", score=0.25, success=1.0),
    ]
    (e,) = episode_rows(rows)
    assert (e["episode"], e["turns"], e["conversations"]) == ("0/0", 2, 2)
    assert e["score"] == 0.75 and e["success"] == 1.0


def test_rows_without_episode_ids_still_log_one_each():
    """A loop that publishes no ids must not have every row collapsed into one bogus
    episode -- that would silently show eight trajectories as a single garbled one."""
    rows = [{"output": "a", "images": []}, {"output": "b", "images": []}]
    assert len(episode_rows(rows)) == 2


def test_text_is_escaped():
    html = episode_html([_row(0, 0, 0, "<script>alert(1)</script>")])
    assert "<script>" not in html and "&lt;script&gt;" in html


def test_an_unencodable_frame_does_not_take_the_text_with_it():
    class Bad:
        def convert(self, mode):
            raise RuntimeError("not an image")

    html = episode_html([_row(0, 0, 0, "the-text-still-matters", images=[Bad()])])
    assert "the-text-still-matters" in html
