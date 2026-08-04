"""Tests for the per-conversation token record.

The record's whole job is that the mask keeps describing the token sequence. That can
only fail silently -- a misplaced mask still trains, still produces a finite loss, and
only shows up as a score that does not move -- so the tests are about the invariant
rather than about outputs looking plausible.
"""

import pytest

from vagen.core.tape import Conversation, MaskMisaligned, Row


def _conv(prompt=(1, 2, 3)):
    c = Conversation(conversation_id="c")
    c.add_context(list(prompt))
    return c


# --------------------------------------------------------------------- structure


def test_opening_context_is_not_trainable():
    """Nothing before the model's first token belongs to the trainable region."""
    c = _conv()

    assert c.mask == [] and not c.is_trainable()


def test_a_response_starts_the_trainable_region():
    c = _conv()
    c.add_response([7, 8], [0.1, 0.2])

    row = c.row()
    assert row.prompt_ids == [1, 2, 3]
    assert row.response_ids == [7, 8]
    assert row.response_mask == [1, 1]
    assert row.logprobs == [0.1, 0.2]


def test_observations_between_turns_are_masked_out():
    """★ The distinction the whole record exists for: the model is credited for what it
    produced, not for what the environment said back."""
    c = _conv()
    c.add_response([7, 8])
    c.add_context([9])
    c.add_response([10])

    assert c.row().response_mask == [1, 1, 0, 1]


def test_a_conversation_the_model_never_spoke_in_is_dropped():
    """A new conversation immediately followed by a terminal step carries no gradient."""
    c = _conv()

    assert not c.is_trainable()
    with pytest.raises(MaskMisaligned, match="no model output"):
        c.row()


def test_row_rejects_an_inconsistent_record():
    with pytest.raises(MaskMisaligned, match="inconsistent"):
        Row(prompt_ids=[1], response_ids=[2, 3], response_mask=[1], logprobs=[0.0, 0.0])


# ---------------------------------------------------------------------- adoption


def test_adopting_before_any_response_just_replaces():
    """The first call has no trainable region to disturb."""
    c = _conv()
    c.adopt_prompt([1, 2, 3, 4, 5])

    assert c.token_ids == [1, 2, 3, 4, 5] and c.mask == []


def test_a_grown_observation_grows_its_run_of_zeros():
    """★ The engine expanded the newest image into more tokens than we did, so the mask
    must describe more observation positions -- otherwise the prompt/response split lands
    in the wrong place."""
    c = _conv()
    c.add_response([7, 8])
    c.add_context([9])                      # 6 tokens total, mask [1,1,0]
    c.adopt_prompt([1, 2, 3, 7, 8, 9, 9, 9])  # engine made that observation 3 tokens

    assert c.mask == [1, 1, 0, 0, 0]
    assert c.row().response_ids == [7, 8, 9, 9, 9]


def test_a_shrunken_observation_shrinks_it():
    c = _conv()
    c.add_response([7, 8])
    c.add_context([9, 9, 9])
    c.adopt_prompt([1, 2, 3, 7, 8, 9])

    assert c.mask == [1, 1, 0]


def test_model_output_is_never_reweighted_by_an_adoption():
    """Only context spans carry images, so the count of trained-on positions must be
    invariant -- a shifted response mask is exactly the corruption being avoided."""
    c = _conv()
    c.add_response([7, 8])
    c.add_context([9])
    before = sum(c.mask)
    c.adopt_prompt([1, 2, 3, 7, 8, 9, 9, 9, 9])

    assert sum(c.mask) == before == 2


def test_logprobs_track_the_mask_through_an_adoption():
    c = _conv()
    c.add_response([7, 8], [0.5, 0.5])
    c.add_context([9])
    c.adopt_prompt([1, 2, 3, 7, 8, 9, 9])

    assert len(c.logprobs) == len(c.mask)
    assert c.logprobs[:2] == [0.5, 0.5]


def test_a_delta_too_large_to_absorb_raises():
    c = _conv()
    c.add_response([7, 8])
    c.add_context([9])
    with pytest.raises(MaskMisaligned, match="not confined"):
        c.adopt_prompt([1, 2])


def test_a_shifted_opening_context_raises():
    """★ The load-bearing assumption is that everything before the newest context is
    already in the engine's form and re-expands identically. Assert it, do not trust it."""
    c = _conv()
    c.add_response([7, 8])

    # The tail is a response, so no context span can absorb the delta. Rather than
    # silently shifting where the prompt ends, the invariant has to catch it.
    with pytest.raises(MaskMisaligned, match="other than the newest context"):
        c.adopt_prompt([1, 2, 3, 4, 7, 8])


def test_unchanged_length_leaves_everything_alone():
    c = _conv()
    c.add_response([7, 8])
    c.add_context([9])
    c.adopt_prompt([1, 2, 3, 7, 8, 9])

    assert c.mask == [1, 1, 0] and c.row().prompt_ids == [1, 2, 3]


# ------------------------------------------------------------------------ reward


def test_a_scalar_reward_lands_on_the_last_model_token():
    """Not the last token: an observation may follow the final response, and crediting
    the environment's own words is meaningless."""
    c = _conv()
    c.add_response([7, 8])
    c.add_context([9])

    assert c.place_reward(1.5) == [0.0, 1.5, 0.0]


def test_a_vector_reward_is_used_as_given():
    c = _conv()
    c.add_response([7, 8])

    assert c.place_reward([0.25, 0.75]) == [0.25, 0.75]


def test_a_misaligned_vector_reward_raises():
    """★ An env that re-encoded the response to build this would misalign it, and a
    reward one token off is invisible in every metric."""
    c = _conv()
    c.add_response([7, 8])

    with pytest.raises(MaskMisaligned, match="align them to response_token_ids"):
        c.place_reward([0.1, 0.2, 0.3])


def test_reward_on_a_silent_conversation_is_all_zero():
    c = _conv()
    assert c.place_reward(1.0) == []


def test_the_record_needs_none_of_the_training_stack():
    """★ The claim that eval needs only the harness rests on this layer staying
    importable without torch, verl or transformers. Checked against the source, because
    other tests will have those packages loaded already and sys.modules cannot tell us
    who imported them."""
    import vagen.core.tape as tape_mod

    source = open(tape_mod.__file__).read()

    for package in ("torch", "verl", "transformers", "numpy"):
        assert f"import {package}" not in source, f"the token record now depends on {package}"
