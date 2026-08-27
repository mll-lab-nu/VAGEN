"""Tests for the per-conversation token record.

The record's whole job is that the mask keeps describing the token sequence. That can
only fail silently -- a misplaced mask still trains, still produces a finite loss, and
only shows up as a score that does not move -- so the tests are about the invariant
rather than about outputs looking plausible.
"""

import pytest

from vagen.rollout.trajectory import Conversation, MaskMisaligned, Row


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
    assert row.logprobs_complete is True


def test_logprob_provenance_distinguishes_real_zero_from_missing():
    supplied = _conv()
    supplied.add_response([7, 8], [0.0, 0.0])
    assert supplied.row().logprobs == [0.0, 0.0]
    assert supplied.row().logprobs_complete is True

    missing = _conv()
    missing.add_response([7, 8])
    assert missing.row().logprobs == [0.0, 0.0]
    assert missing.row().logprobs_complete is False


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
        Row(conversation_id='c', prompt_ids=[1], response_ids=[2, 3], response_mask=[1], logprobs=[0.0, 0.0], scores=[0.0, 0.0])


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
    with pytest.raises(MaskMisaligned, match="before the newest context|not confined"):
        c.adopt_prompt([1, 2])


def test_a_shifted_opening_context_raises():
    """★ The load-bearing assumption is that everything before the newest context is
    already in the engine's form and re-expands identically. Assert it, do not trust it."""
    c = _conv()
    c.add_response([7, 8])

    # The tail is a response, so no context span can absorb the delta. Rather than
    # silently shifting where the prompt ends, the invariant has to catch it.
    with pytest.raises(MaskMisaligned, match="before the newest context|other than the newest"):
        c.adopt_prompt([1, 2, 3, 4, 7, 8])


def test_unchanged_length_leaves_everything_alone():
    c = _conv()
    c.add_response([7, 8])
    c.add_context([9])
    c.adopt_prompt([1, 2, 3, 7, 8, 9])

    assert c.mask == [1, 1, 0] and c.row().prompt_ids == [1, 2, 3]


# ------------------------------------------------------------------------ reward


def test_a_scalar_reward_lands_on_the_turn_that_earned_it():
    """★ Not on the conversation's last token. A concat episode is many turns in one
    conversation, and an observation may follow the final response -- crediting the
    environment's own words, or the wrong turn, erases what the turn structure is for."""
    c = _conv()
    c.add_response([7, 8])
    c.add_reward(1.5)
    c.add_context([9])
    c.add_response([10])
    c.add_reward(2.0)

    assert c.row().scores == [0.0, 1.5, 0.0, 2.0]


def test_rewards_accumulate_rather_than_overwrite():
    c = _conv()
    c.add_response([7])
    c.add_reward(1.0)
    c.add_reward(0.5)

    assert c.row().scores == [1.5]


def test_a_vector_reward_covers_that_turns_response():
    c = _conv()
    c.add_response([7, 8])
    c.add_context([9])
    c.add_response([10, 11])
    c.add_reward([0.25, 0.75])

    assert c.row().scores == [0.0, 0.0, 0.0, 0.25, 0.75]


def test_a_misaligned_vector_reward_raises():
    """★ An env that re-encoded the response to build this would misalign it, and a
    reward one token off is invisible in every metric."""
    c = _conv()
    c.add_response([7, 8])

    with pytest.raises(MaskMisaligned, match="align them to response_token_ids"):
        c.add_reward([0.1, 0.2, 0.3])


def test_crediting_before_any_response_raises():
    c = _conv()
    with pytest.raises(MaskMisaligned, match="acted on nothing"):
        c.add_reward(1.0)


def test_scores_survive_an_adoption():
    c = _conv()
    c.add_response([7, 8])
    c.add_reward(1.0)
    c.add_context([9])
    c.adopt_prompt([1, 2, 3, 7, 8, 9, 9, 9])

    assert len(c.scores) == len(c.mask)
    assert c.scores[:2] == [0.0, 1.0], "credit must stay on the token that earned it"


def test_a_delta_split_between_the_opening_and_the_tail_is_caught():
    """The length check cannot see this: the mask grows by exactly the delta that grew
    the tokens, so the difference it compares is unchanged wherever the change was. The
    consequence is prompt_len pointing short of the real boundary, which puts prompt
    tokens at the head of response_ids with mask 1 -- trained on as if generated."""
    c = Conversation(conversation_id="c")
    c.add_context([1, 2, 3])
    c.add_response([7, 8])
    c.add_context([9])
    with pytest.raises(MaskMisaligned, match="before the newest context"):
        # opening grew by 2 (90, 91) and the observation by 1
        c.adopt_prompt([1, 2, 3, 90, 91, 7, 8, 9, 9])


def test_a_change_confined_to_the_newest_context_is_still_accepted():
    c = Conversation(conversation_id="c")
    c.add_context([1, 2, 3])
    c.add_response([7, 8])
    c.add_context([9])
    c.adopt_prompt([1, 2, 3, 7, 8, 9, 9, 9])          # only the tail re-expanded
    assert c.token_ids == [1, 2, 3, 7, 8, 9, 9, 9]
    assert len(c.mask) == len(c.token_ids) - c.prompt_len


def test_an_untouched_prompt_is_accepted():
    c = Conversation(conversation_id="c")
    c.add_context([1, 2, 3])
    c.add_response([7, 8])
    c.adopt_prompt([1, 2, 3, 7, 8])
    assert c.prompt_len == 3


def test_the_opening_prompt_may_still_be_re_expanded():
    """The engine expands an image placeholder in the middle of the opening prompt, so
    requiring a byte-identical prefix there rejects the only case adoption exists for --
    and fatally, since nothing catches MaskMisaligned."""
    c = Conversation(conversation_id="c")
    c.add_context([1, 2, 100, 3])                 # one placeholder
    c.adopt_prompt([1, 2, 100, 100, 100, 3])      # engine expanded it to three
    assert c.token_ids == [1, 2, 100, 100, 100, 3]
    assert c.prompt_len is None and c.mask == []


def test_an_aborted_generation_earns_nothing_rather_than_crediting_the_observation():
    """vLLM returns no tokens on abort. `end - 1` then points at the token before the
    turn -- the environment's own text -- or off the start of the list entirely."""
    c = Conversation(conversation_id="c")
    c.add_context([1, 2, 3])
    c.add_response([])
    c.add_reward(1.0)                              # must not raise
    assert c.scores == []

    c2 = Conversation(conversation_id="c2")
    c2.add_context([1, 2, 3])
    c2.add_response([7, 8])
    c2.add_context([9])
    c2.add_response([])
    c2.add_reward(5.0)
    assert c2.scores == [0.0, 0.0, 0.0], f"credited an observation token: {c2.scores}"
