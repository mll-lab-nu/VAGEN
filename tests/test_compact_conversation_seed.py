"""What a conversation opened by compaction actually contains.

The environment resets once per episode, so only the first conversation ever sees an
initial observation. Every conversation after it opens on whatever the last ``step``
returned -- and it opens as one user turn, not two in a row.
"""

from __future__ import annotations

import pytest

from vagen.harness.compact import CompactHarness, _with_summary


class _Resp:
    def __init__(self, text, conversation_id="c0"):
        self.text, self.conversation_id = text, conversation_id


def _harness(budget=10):
    h = CompactHarness(budget=budget)
    h.begin({"role": "system", "content": "SYSTEM"},
            {"role": "user", "content": "INIT-OBS", "images": ["frame0"]})
    return h


def test_the_first_conversation_opens_on_the_initial_observation():
    h = _harness()
    call = h.next_call()
    assert call.conversation_id is None
    assert [m["content"] for m in call.messages] == ["SYSTEM", "INIT-OBS"]


def test_a_conversation_opened_by_compaction_carries_the_latest_step_not_the_initial():
    h = _harness(budget=1)
    h.accept(_Resp("act 0"))                       # turn 0 in conversation 0
    h.add_observation({"role": "user", "content": "STEP-OBS-1", "images": ["frame1"]})
    h.note_usage(999)                              # over budget -> summarise

    ask = h.next_call()
    assert ask.messages[0]["content"] == CompactHarness.SUMMARY_REQUEST
    assert ask.conversation_id is not None, "the summary must be asked inside the old context"
    assert h.accept(_Resp("THE SUMMARY")) is None, "a summary must not be sent to the env"

    seed = h.next_call()
    assert seed.conversation_id is None, "compaction must open a new conversation"
    assert len(seed.messages) == 2, f"expected system + one user turn, got {len(seed.messages)}"
    assert seed.messages[0]["content"] == "SYSTEM"
    body = seed.messages[1]["content"]
    assert "THE SUMMARY" in body
    assert "STEP-OBS-1" in body, "the new conversation lost the latest observation"
    assert "INIT-OBS" not in body, (
        "re-sent the initial observation; the environment only produces one, at reset"
    )


def test_the_new_conversation_keeps_the_frame_of_that_observation():
    h = _harness(budget=1)
    h.accept(_Resp("act 0"))
    h.add_observation({"role": "user", "content": "STEP-OBS-1", "images": ["frame1"]})
    h.note_usage(999)
    h.next_call(); h.accept(_Resp("THE SUMMARY"))
    seed = h.next_call()
    assert seed.messages[1].get("images") == ["frame1"], "the frame went missing"


def test_it_is_one_user_turn_not_two():
    """system / user / assistant. Two consecutive user messages are not something a chat
    template is obliged to render the same way, and an episode should not depend on it."""
    h = _harness(budget=1)
    h.accept(_Resp("act 0"))
    h.add_observation({"role": "user", "content": "STEP-OBS-1"})
    h.note_usage(999)
    h.next_call(); h.accept(_Resp("S"))
    roles = [m["role"] for m in h.next_call().messages]
    assert roles == ["system", "user"], roles


def test_multimodal_content_keeps_its_parts_in_order():
    msg = _with_summary("SUM", {"role": "user",
                                "content": [{"type": "image"}, {"type": "text", "text": "OBS"}]})
    assert msg["content"][0] == {"type": "text", "text": "SUM\n\n"}
    assert msg["content"][1:] == [{"type": "image"}, {"type": "text", "text": "OBS"}]


def test_the_summary_does_not_run_into_the_observation():
    """A chat template joins the parts of a content list with nothing between them, so a
    separator added only for string content leaves
    '...align it with the target.After your answer, the extracted valid action is...'
    -- no boundary at all between the story so far and where the agent now is."""
    for content in ("PLAIN-OBS", [{"type": "image"}, {"type": "text", "text": "PART-OBS"}]):
        msg = _with_summary("THE SUMMARY", {"role": "user", "content": content})
        rendered = (
            msg["content"] if isinstance(msg["content"], str)
            else "".join(p.get("text", "") for p in msg["content"])
        )
        assert "THE SUMMARY\n\n" in rendered, f"no blank line for {type(content).__name__}"
        assert "SUMMARYAfter" not in rendered and "SUMMARYPART" not in rendered
