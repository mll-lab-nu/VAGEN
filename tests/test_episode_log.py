"""An episode has to survive being logged as one thing.

A row is one model call. An episode is several, and once the context is compacted it is
several conversations -- so the naive table shows one trajectory as unrelated rows with
nothing saying they belong together or in what order. These pin the regrouping, the
ordering, and that the frames the agent saw actually make it into the cell.
"""

from __future__ import annotations

import re

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
    html = episode_html([{"conversations": [{
        "conversation_id": 0,
        "prompt": [{"text": "p"}, {"image": _frame()}],
        "turns": [{"turn_id": 0, "response": [{"text": "r"}],
                   "observation": [{"text": "o"}, {"image": _frame((99, 0, 0))}]}]}]}])
    assert html.count("data:image/png;base64,") == 2




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

    html = episode_html([{"conversations": [{
        "conversation_id": 0, "prompt": [{"image": Bad()}],
        "turns": [{"turn_id": 0, "response": [{"text": "the-text-still-matters"}],
                   "observation": []}]}]}])
    assert "the-text-still-matters" in html


# ------------------------------------------------- the columns must survive the batch
def test_the_manager_restores_the_per_row_columns():
    """turn_idx and conversation_id are per row, not per rollout.

    They cannot be recovered from input_non_tensor_batch, which is per rollout and gets
    repeated. If verl drops the extra_fields carrying them, every turn sorts equal and
    the transcript reads as a coherent episode that never happened -- silently.
    """
    import numpy as np

    from vagen.training.agent_loop.multi_output import MultiOutputAgentLoopWorker as W

    class _Out:
        def __init__(self, turn, conversation):
            self.extra_fields = {"turn_idx": turn, "conversation_id": conversation}

    class _Batch:
        def __init__(self):
            self.non_tensor_batch = {}

    flat = [_Out(1, "c1"), _Out(0, "c0"), _Out(2, "c1")]
    out = W._vagen_restore_row_columns(W, _Batch(), flat)
    assert list(out.non_tensor_batch["turn_idx"]) == [1, 0, 2]
    assert list(out.non_tensor_batch["conversation_id"]) == ["c1", "c0", "c1"]


def test_restore_does_not_clobber_columns_that_survived():
    import numpy as np

    from vagen.training.agent_loop.multi_output import MultiOutputAgentLoopWorker as W

    class _Out:
        extra_fields = {"turn_idx": 99, "conversation_id": "x"}

    class _Batch:
        def __init__(self):
            self.non_tensor_batch = {"turn_idx": np.array([7], dtype=object)}

    out = W._vagen_restore_row_columns(W, _Batch(), [_Out()])
    assert list(out.non_tensor_batch["turn_idx"]) == [7], "overwrote the real column"


def _one_frame_html(img):
    return episode_html([{"conversations": [{"conversation_id": 0,
                                             "prompt": [{"image": img}], "turns": []}]}])


def test_a_large_frame_is_downscaled_before_encoding():
    """The width used to be CSS only, so the payload was full resolution.

    Invisible at sokoban's 192px; for a renderer producing 1024px frames it is the
    difference between a few KB and a few hundred per turn, every validation.
    """
    big = PIL.new("RGB", (1600, 900), (7, 7, 7))
    small = PIL.new("RGB", (160, 90), (7, 7, 7))
    big_len = len(_one_frame_html(big))
    small_len = len(_one_frame_html(small))
    assert big_len < 4 * small_len, f"large frame not downscaled: {big_len} vs {small_len}"


def test_a_small_frame_is_not_upscaled():
    tiny = PIL.new("RGB", (16, 16), (1, 2, 3))
    h = episode_html([{"conversations": [{"conversation_id": 0,
                                          "prompt": [{"image": tiny}], "turns": []}]}])
    assert "base64," in h


# ------------------------------------------------------------------ sampling
from vagen.utils.episode_log import select_episodes  # noqa: E402


def _ep(name, success, source="s1"):
    return {"episode": name, "success": success, "data_source": source, "html": "", "turns": 1}


def test_the_sample_is_half_successes_half_failures():
    """First-n at a 12% success rate is eight failures, and a log with nothing to
    compare against teaches nothing."""
    eps = [_ep(f"f{i}", 0.0) for i in range(20)] + [_ep(f"s{i}", 1.0) for i in range(20)]
    got = select_episodes(eps, 8)
    assert len(got) == 8
    assert sum(1 for e in got if e["success"]) == 4


def test_a_shortfall_of_one_class_is_filled_by_the_other():
    """A run with almost no successes must still log a full table, not half of one."""
    eps = [_ep(f"f{i}", 0.0) for i in range(20)] + [_ep("s0", 1.0)]
    got = select_episodes(eps, 8)
    assert len(got) == 8
    assert sum(1 for e in got if e["success"]) == 1


def test_no_successes_at_all_still_fills_the_table():
    got = select_episodes([_ep(f"f{i}", 0.0) for i in range(20)], 6)
    assert len(got) == 6


def test_sources_are_sampled_across_not_from_the_first():
    """With several validation sets, a table drawn entirely from whichever sorted first
    hides the others completely."""
    eps = ([_ep(f"a{i}", i % 2, "alpha") for i in range(20)]
           + [_ep(f"b{i}", i % 2, "beta") for i in range(20)])
    got = select_episodes(eps, 8)
    assert {e["data_source"] for e in got} == {"alpha", "beta"}
    assert sum(1 for e in got if e["data_source"] == "alpha") == 4


def test_asking_for_more_than_exists_returns_what_exists():
    assert len(select_episodes([_ep("a", 1.0), _ep("b", 0.0)], 50)) == 2


def test_zero_and_empty():
    assert select_episodes([_ep("a", 1.0)], 0) == []
    assert select_episodes([], 5) == []


def test_selection_is_deterministic():
    """The table has to read the same way step to step to be a progression."""
    eps = [_ep(f"f{i}", 0.0) for i in range(10)] + [_ep(f"s{i}", 1.0) for i in range(10)]
    assert [e["episode"] for e in select_episodes(eps, 6)] == [
        e["episode"] for e in select_episodes(list(eps), 6)
    ]


# ------------------------------------------------ the three context policies
def test_concat_shape_one_conversation_many_turns():
    """concat: the whole episode is one conversation, and one row."""
    rows = [_row(0, 0, 0, "everything", conversation="c0")]
    rows[0]["episode_turns"] = 5
    (e,) = episode_rows(rows)
    assert (e["turns"], e["conversations"]) == (5, 1), (
        "concat episode reported as a single turn: the row count is not the turn count"
    )


def test_no_concat_shape_many_conversations_one_turn_each():
    rows = [_row(0, 0, i, f"t{i}", conversation=f"c{i}") for i in range(4)]
    for r in rows:
        r["episode_turns"] = 4
    (e,) = episode_rows(rows)
    assert (e["turns"], e["conversations"]) == (4, 4)


def test_compact_shape_several_conversations_many_turns():
    rows = [_row(0, 0, i, f"t{i}", conversation="c0" if i < 3 else "c1") for i in range(6)]
    for r in rows:
        r["episode_turns"] = 6
    (e,) = episode_rows(rows)
    assert (e["turns"], e["conversations"]) == (6, 2)


# ------------------------------------------------------- selection strategies
def test_the_ratio_is_a_parameter_not_a_constant():
    eps = ([_ep(f"s{i}", 1.0) for i in range(10)] + [_ep(f"f{i}", 0.0) for i in range(10)])
    assert sum(1 for e in select_episodes(eps, 8, success_ratio=0.25) if e["success"]) == 2
    assert sum(1 for e in select_episodes(eps, 8, success_ratio=0.75) if e["success"]) == 6
    assert sum(1 for e in select_episodes(eps, 8, success_ratio=1.0) if e["success"]) == 8
    assert sum(1 for e in select_episodes(eps, 8, success_ratio=0.0) if e["success"]) == 0


@pytest.mark.parametrize("strategy", ["first", "failures", "successes", "worst", "best"])
def test_every_named_strategy_returns_something(strategy):
    eps = ([_ep(f"s{i}", 1.0) for i in range(5)] + [_ep(f"f{i}", 0.0) for i in range(5)])
    for e, r in zip(eps, range(10)):
        e["reward"] = float(r)
    got = select_episodes(eps, 3, strategy)
    assert 0 < len(got) <= 3


def test_worst_and_best_are_opposite_ends():
    eps = [dict(_ep(f"e{i}", i % 2), reward=float(i)) for i in range(6)]
    assert [e["reward"] for e in select_episodes(eps, 2, "worst")] == [0.0, 1.0]
    assert [e["reward"] for e in select_episodes(eps, 2, "best")] == [5.0, 4.0]


def test_an_unknown_strategy_says_what_the_options_are():
    with pytest.raises(ValueError, match="balanced"):
        select_episodes([_ep("a", 1.0)], 1, "whatever-i-typed")


# ---------------------------------------------- the conversation-first layout
def _conv(cid, prompt, turns):
    return {"conversation_id": cid, "prompt": prompt, "prompt_image": None,
            "turns": [{"turn_id": i, "response": r, "observation": o,
                       "observation_image": None} for i, (r, o) in enumerate(turns)]}


def test_a_conversation_heading_precedes_its_own_system_prompt():
    """The heading marks where a conversation starts, so it goes before that
    conversation's system prompt -- not after the response that preceded it. Walking the
    batch attaches the next row's prompt to the previous turn, which is what put each new
    system prompt a whole turn late."""
    h = episode_html([{"conversations": [
        _conv(0, "SYSTEM-ZERO", [("R0", "")]),
        _conv(1, "SYSTEM-ONE", [("R1", "")]),
    ]}])
    assert h.index("conversation 0") < h.index("SYSTEM-ZERO") < h.index("R0")
    assert h.index("R0") < h.index("conversation 1") < h.index("SYSTEM-ONE") < h.index("R1")


def test_each_turn_shows_as_an_assistant_block():
    h = episode_html([{"conversations": [
        _conv(0, "S0", [("a", ""), ("b", "")]),
        _conv(1, "S1", [("c", "")]),
    ]}])
    for body in ("a", "b", "c"):
        assert f">{body}</div>" in h, f"turn body {body} missing"


def test_the_summary_exchange_is_just_the_last_turn_of_its_conversation():
    """Compaction is an ordinary user/assistant pair: 'summarise' then the summary. It
    needs no special rendering, and inventing one would misrepresent what happened."""
    h = episode_html([{"conversations": [
        _conv(0, "S0", [("act", "please summarise"), ("THE-SUMMARY", "")]),
        _conv(1, "S1 carrying THE-SUMMARY", [("next act", "")]),
    ]}])
    assert h.index("please summarise") < h.index("THE-SUMMARY") < h.index("conversation 1")


def test_a_single_conversation_episode_still_gets_one_heading():
    h = episode_html([{"conversations": [_conv(0, "S", [("a", ""), ("b", "")])]}])
    assert h.count("conversation 0") == 1 and "conversation 1" not in h


def test_the_transcript_adds_only_roles_and_conversation_rules():
    """It should read like the sequence that was trained on. Turn numbers and per-turn
    rules are inferable from the text; where a conversation restarts is not."""
    h = episode_html([{"conversations": [
        _conv(0, "SYS", [("a", "obs")]),
        _conv(1, "SYS2", [("b", "")]),
    ]}])
    # Only the conversation heading is ours. The decoded text carries the template's
    # own role markers; adding a second set put two of each on the screen, and ours were
    # a guess at the block's contents while the template's are what the model read.
    assert "turn 0" not in h and "turn 1" not in h
    assert h.count("conversation 0") == 1 and h.count("conversation 1") == 1
    for ours in ("system + user", ">assistant<", ">user<"):
        assert ours not in h, f"still labelling {ours!r}"


def test_an_absent_observation_prints_no_empty_user_block():
    h = episode_html([{"conversations": [_conv(0, "SYS", [("only-response", "")])]}])
    assert "only-response" in h
    assert h.count("<div>") == 2, "an empty block was rendered"


def test_the_frame_sits_where_its_placeholder_was():
    """The merge splits a span at its placeholder run and puts the picture there, so the
    renderer has nothing to decide. The decoder cue is ordinary text on either side."""
    h = episode_html([{"conversations": [{
        "conversation_id": 0,
        "prompt": [{"text": "system\nrules\nuser\n[obs]"}, {"image": _frame()},
                   {"text": "\n\nassistant"}],
        "turns": [{"turn_id": 0, "response": [{"text": "<answer>Up</answer>"}],
                   "observation": []}],
    }]}])
    assert h.index("[obs]") < h.index("base64,") < h.index("assistant")
    assert "assistant" in h, "the decoder cue is part of the sequence and stays"


def test_a_frame_appears_between_the_texts_that_surrounded_its_placeholder():
    """Position comes from the sequence, so the renderer only walks the parts in order.

    Superseded the previous version of this test, which asserted that a frame was
    appended at the end of a turn -- a rule that existed only because the position had
    been thrown away by decoding.
    """
    h = episode_html([{"conversations": [{
        "conversation_id": 0,
        "prompt": [{"text": "BEFORE-FRAME"}, {"image": _frame()}, {"text": "AFTER-FRAME"}],
        "turns": [{"turn_id": 0, "response": [{"text": "RESPONSE"}], "observation": []}],
    }]}])
    assert h.index("BEFORE-FRAME") < h.index("base64,") < h.index("AFTER-FRAME")
    assert h.index("AFTER-FRAME") < h.index("RESPONSE")


def test_a_block_with_no_frame_is_left_whole():
    """The split exists only to place a frame. With none, nothing is rearranged."""
    h = episode_html([{"conversations": [{
        "conversation_id": 0, "prompt": "p", "prompt_image": None,
        "turns": [{"turn_id": 0, "response": "I am the assistant here, and I act",
                   "observation": "", "observation_image": None}]}]}])
    assert "I am the assistant here, and I act" in h


def test_best_does_not_rank_the_unscored_first():
    """`reverse=True` flips every component of the sort key, including "has no reward"."""
    eps = [dict(_ep("a", 1.0), reward=1.0), dict(_ep("b", 1.0), reward=5.0),
           dict(_ep("c", 0.0), reward=None)]
    assert [e["episode"] for e in select_episodes(eps, 2, "best")] == ["b", "a"]
    assert [e["episode"] for e in select_episodes(eps, 2, "worst")] == ["a", "b"]
