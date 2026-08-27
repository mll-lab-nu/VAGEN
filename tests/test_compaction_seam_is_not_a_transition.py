"""A compaction seam must not be discounted as if the environment had stepped.

Under ``compact`` a conversation ends because the context filled up: the model is asked
to summarise, writes one, and the next conversation opens on the *same* world state. Two
model emissions are separated by that seam -- the summary and the next action -- so the
token stream shows two turn endings where one environment transition happened.

An estimator that spends ``algorithm.lam`` crossing a turn, and spends it at the seam too,
attenuates credit by ``lam ** 2`` where an ordinary turn costs ``lam`` -- and how often that
happens is set by how often the policy compacts, which is set by how much it writes. The
effective horizon then moves as the policy's verbosity moves, which is not a hyperparameter
anybody chose.

★ No estimator in the tree consults the seam today: the one that did,
``removed_estimator_gae_varlam``, has been removed, and the remaining ones either run at ``lam = 1``
where the question is moot or deliberately treat every boundary alike. What is tested here
is that the flag is *computed correctly and reaches the estimator inputs* -- the harness
knows which conversation it summarised, that reaches ``AdvantageInputs.ends_with_summary``,
and ``TrajectoryView.seam`` turns it into a per-position mask. That is the contract a custom
estimator picks up, and it is the half that cannot be re-derived downstream.

The seam also joins two critic evaluations of the same world state rendered as two
different pieces of text -- the conversation being closed, and the summary standing in
for it. ``lam = 1`` is the only value that telescopes those two values away.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

import vagen.algorithms  # noqa: F401  -- registers the estimators
from vagen.algorithms._common.packing import _is_turn_boundary, _pack


class _View:
    """The two-row compact layout, gathered: row 0 is a conversation, row 1 the next."""

    def __init__(self, mask):
        self.mask = mask
        self.rows = torch.arange(mask.shape[0])
        self.trajectories = [list(range(mask.shape[0]))]

    def gather(self, x):
        return x

    def broadcast(self, x):
        return x


class _Cfg(dict):
    __getattr__ = dict.__getitem__


class _Inputs:
    def __init__(self, mask, ends_with_summary, rewards=None, values=None,
                 gamma=1.0, lam=0.95):
        self.view = _View(mask)
        zeros = torch.zeros(mask.shape, dtype=torch.float32)
        self.rewards = zeros if rewards is None else rewards
        self.values = zeros.clone() if values is None else values
        self.ends_with_summary = ends_with_summary
        self.config = _Cfg(gamma=gamma, lam=lam)


#: The one string the writer and the reader have to agree on.
COLUMN_THE_LOOP_WRITES = "ends_with_summary"

WIDTH = 8


def _compact_layout():
    """Row 0: action A, gap, SUMMARY at the row's end. Row 1: action B, then C.

    Two rows of one episode, which is what compaction produces.
    """
    mask = torch.zeros((2, WIDTH), dtype=torch.long)
    mask[0, 0:2] = 1          # action A
    mask[0, 6:8] = 1          # the summary, running to the row's end
    mask[1, 0:2] = 1          # action B
    mask[1, 4:6] = 1          # action C
    return mask


def test_the_summarys_ending_is_a_turn_boundary_like_any_other():
    """The premise. A summary is a model emission and an action, so the token stream
    genuinely does end a turn there -- the seam is not a mis-detection to be removed."""
    packed = _pack(_Inputs(_compact_layout(), [True, False]))
    boundary = packed.boundary()[0].tolist()
    # tokens: A0 A1 S0 S1 | B0 B1 C0 C1
    assert boundary == [False, True, False, True, False, True, False, True]


def test_only_the_summarys_last_token_is_a_seam_not_every_turn_in_its_row():
    """★ A conversation holds several turns. Marking the whole flagged row would switch
    inter-turn discounting off for all of them, which is a larger error than the one
    being fixed and points the same way, so it would not show up as a regression."""
    packed = _pack(_Inputs(_compact_layout(), [True, False]))
    seam = packed.seam([True, False])[0].tolist()
    assert seam == [False, False, False, True, False, False, False, False], (
        "exactly one seam: the last token of row 0, which is the summary's"
    )


def test_a_row_that_did_not_summarise_has_no_seam():
    """no_concat also puts each turn in its own row, and there the row change *is* an
    environment step. The flag, not the row change, is what distinguishes them."""
    packed = _pack(_Inputs(_compact_layout(), [False, False]))
    assert not packed.seam([False, False]).any()


def test_an_absent_column_means_no_seams_rather_than_an_error():
    """Every harness but compact leaves it unset, as does any batch made before the
    column existed."""
    packed = _pack(_Inputs(_compact_layout(), None))
    assert not packed.seam(None).any()


# ------------------------------------------------------------------ what it buys

def _lambdas(ends_with_summary, lam_low, lam_high):
    """The per-position lambda an estimator would build from the seam mask."""
    packed = _pack(_Inputs(_compact_layout(), ends_with_summary))
    seq_lam = torch.where(packed.boundary(), lam_high, lam_low)
    return torch.where(packed.seam(ends_with_summary), 1.0, seq_lam)[0]


def test_the_seam_costs_nothing_while_a_real_turn_still_costs_lam():
    """★ The fix, stated as the quantity that was wrong. Credit crossing the seam is
    multiplied by 1.0; credit crossing an environment turn is still multiplied by lam."""
    lam_low, lam_high = 1.0, 0.95
    lam = _lambdas([True, False], lam_low, lam_high)

    assert lam[3].item() == pytest.approx(1.0), "the seam must be free"
    assert lam[1].item() == pytest.approx(lam_high), "A -> summary is a real transition"
    assert lam[5].item() == pytest.approx(lam_high), "B -> C is a real transition"


def test_without_the_fix_a_seam_costs_lam_squared():
    """The defect this pins, measured. Between action A and action B there is exactly one
    environment step, so credit should be attenuated once. Left unfixed it is attenuated
    twice -- and the second factor is charged for compacting, not for acting."""
    lam_high = 0.95
    packed = _pack(_Inputs(_compact_layout(), [True, False]))

    unfixed = torch.where(packed.boundary(), lam_high, 1.0)[0]
    fixed = _lambdas([True, False], 1.0, lam_high)

    # Positions 1 and 3 sit between action A and action B.
    assert unfixed[1].item() * unfixed[3].item() == pytest.approx(lam_high**2)
    assert fixed[1].item() * fixed[3].item() == pytest.approx(lam_high)


def test_more_compactions_do_not_mean_less_credit():
    """★ Why this is not a small constant factor. The number of seams in an episode is
    set by how verbosely the policy writes, and the policy changes that as it trains, so
    an unfixed run's effective horizon drifts with a quantity nobody configured."""
    lam_high = 0.95
    survives_unfixed = []
    survives_fixed = []
    for seams in (1, 3, 6):
        mask = torch.zeros((seams + 1, WIDTH), dtype=torch.long)
        for r in range(seams + 1):
            mask[r, 0:2] = 1
            mask[r, 6:8] = 1
        flags = [True] * seams + [False]
        packed = _pack(_Inputs(mask, flags))
        boundary, seam = packed.boundary()[0], packed.seam(flags)[0]
        valid = packed.valid[0]
        survives_unfixed.append(
            float(torch.where(boundary, lam_high, 1.0)[valid].prod())
        )
        survives_fixed.append(
            float(torch.where(seam, 1.0, torch.where(boundary, lam_high, 1.0))[valid].prod())
        )

    # Each row holds one action and one summary. Fixed, only the action is a transition,
    # so a row costs one lam and the last row -- whose summary is not a seam, the episode
    # simply ends -- costs two. Unfixed, every emission is charged: two per row.
    for i, seams in enumerate((1, 3, 6)):
        assert survives_fixed[i] == pytest.approx(lam_high ** (seams + 2), rel=1e-6)
        assert survives_unfixed[i] == pytest.approx(lam_high ** (2 * (seams + 1)), rel=1e-6)

    # ★ The exponent is what matters. Unfixed it grows at 2 per compaction and fixed at
    # 1, so the gap widens without bound as the policy gets more verbose -- this is a
    # drifting horizon, not a constant factor that a tuned lam could absorb.
    assert survives_unfixed[2] / survives_fixed[2] == pytest.approx(lam_high**6, rel=1e-6)


# ------------------------------------------ the column reaches the estimator

def test_the_property_reads_the_column_the_agent_loop_writes():
    """★ Everything above builds the view by hand. This is the join: the agent loop writes
    a `ends_with_summary` column, `AdvantageInputs.ends_with_summary` reads one, and if the
    two strings ever stop matching then `seam()` returns all-zeros and every compaction seam
    is charged as a real environment transition -- silently, on every episode.

    The first version of this test asserted `hasattr(AdvantageInputs, "ends_with_summary")`
    and grepped the writer's source for the literal. That is one-sided: renaming the string
    on the READER side (`inputs.py`) left the whole suite green while no estimator could see
    a seam again. So assert on the value that comes back.
    """
    import numpy as np

    from vagen.algorithms._common.inputs import AdvantageInputs

    written = COLUMN_THE_LOOP_WRITES
    inputs = AdvantageInputs(
        batch={},
        non_tensor_batch={written: np.asarray([True, False])},
        config=_Cfg(gamma=1.0, lam=1.0),
        estimator="token_level_gae",
    )
    got = inputs.ends_with_summary
    assert got is not None, (
        f"AdvantageInputs.ends_with_summary does not read {written!r}, the column the "
        f"agent loop writes -- so seam() is all-zeros and every compaction seam is "
        f"discounted as an environment step")
    assert list(got) == [True, False]


def test_the_agent_loop_still_writes_that_column():
    """The other end of the same join. Cheap, and it is what makes the test above a join
    rather than a test of AdvantageInputs against itself."""
    import inspect

    from vagen.training.agent_loop import gym_loop, multi_output

    lit = f'"{COLUMN_THE_LOOP_WRITES}"'
    assert lit in inspect.getsource(gym_loop), (
        "the agent loop stopped writing the column the seam is derived from")
    assert lit in inspect.getsource(multi_output), (
        "the column is not carried through multi_output, so it never reaches the batch")


# --------------------------------------------------------- the signal at its source

def test_the_harness_records_which_conversation_it_summarised():
    """The estimator can only be right if the flag is. Only the harness knows the
    difference between "the conversation ended" and "the environment stepped"."""
    from vagen.harness.compact import CompactHarness

    h = CompactHarness(budget=10, summary_budget=4, summary_request_len=1)
    h.begin({"role": "system", "content": "s"}, {"role": "user", "content": "o"})
    h._conversation_id = "conv-0"
    h._awaiting_summary = True

    class _Resp:
        text = "a summary"
        conversation_id = "conv-0"

    h.accept(_Resp())
    assert h.summarised_conversations == {"conv-0"}
    assert h._conversation_id is None


def test_a_new_episode_forgets_the_previous_ones_seams():
    """Conversation ids are per-episode ordinals, so a carried-over set marks unrelated
    conversations in the next episode."""
    from vagen.harness.compact import CompactHarness

    h = CompactHarness(budget=10, summary_budget=4, summary_request_len=1)
    h.begin({"role": "system", "content": "s"}, {"role": "user", "content": "o"})
    h.summarised_conversations.add("conv-0")
    h.begin({"role": "system", "content": "s"}, {"role": "user", "content": "o"})
    assert h.summarised_conversations == set()


def test_every_harness_answers_the_question_not_only_the_compacting_one():
    """`_outputs` asks unconditionally; a harness without the attribute would fail the
    run under concat and no_concat, which never compact."""
    from vagen.harness.concat import ConcatHarness
    from vagen.harness.no_concat import NoConcatHarness

    for cls in (ConcatHarness, NoConcatHarness):
        assert cls().summarised_conversations == set()
