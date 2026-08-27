"""V(s_t) must be the critic's output from *before* token t, not at it.

The critic emits one number per position, and the number at position p is conditioned on
tokens up to and including p. The value of the state the model acted *from* at response
token i is therefore the output at p = i-1 -- the left shift verl applies when it slices
a model output down to the response region.

Our turn-level estimators anchor a turn's value at its first response token, which is
only V(s_t) because of that shift. Nothing in either repo pinned it. If the `- 1` ever
goes, every turn's value becomes the state *after* the first token was emitted, the
advantage is biased by one position, and nothing fails: shapes match, the loss is finite,
and the run looks normal.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from verl.workers.utils.padding import no_padding_2_padding  # noqa: E402


def _data(prompt_len: int, resp_len: int):
    """One sequence, fully attended, so positions are easy to reason about."""
    from tensordict import TensorDict

    total = prompt_len + resp_len
    return TensorDict(
        {
            "prompts": torch.zeros(1, prompt_len, dtype=torch.long),
            "responses": torch.zeros(1, resp_len, dtype=torch.long),
            "attention_mask": torch.ones(1, total, dtype=torch.long),
        },
        batch_size=(1,),
    )


def test_the_value_for_a_response_token_comes_from_the_position_before_it():
    prompt_len, resp_len = 3, 4
    # The critic's output at absolute position p is simply p, so the assertion reads
    # directly as "which position did this come from".
    flat = torch.arange(prompt_len + resp_len, dtype=torch.float32)

    out = no_padding_2_padding(flat, _data(prompt_len, resp_len))[0]

    # response token i sits at absolute position prompt_len + i; its value must be the
    # critic's output at prompt_len + i - 1.
    expected = torch.tensor([prompt_len + i - 1 for i in range(resp_len)], dtype=torch.float32)
    assert torch.equal(out, expected), (
        f"values are not left-shifted: got {out.tolist()}, want {expected.tolist()}. "
        f"Anchoring a turn at its first token would then read V(s_t+1), not V(s_t)."
    )


def test_the_first_response_token_reads_the_last_prompt_position():
    """The state acted from at the start of the response is the end of the prompt."""
    prompt_len, resp_len = 5, 3
    flat = torch.arange(prompt_len + resp_len, dtype=torch.float32)
    out = no_padding_2_padding(flat, _data(prompt_len, resp_len))[0]
    assert out[0].item() == prompt_len - 1


def test_no_value_is_read_from_beyond_the_response():
    """The shift must drop the last position, not extend past it."""
    prompt_len, resp_len = 2, 3
    flat = torch.arange(prompt_len + resp_len, dtype=torch.float32)
    out = no_padding_2_padding(flat, _data(prompt_len, resp_len))[0]
    assert out.max().item() < prompt_len + resp_len - 1


def test_both_extraction_paths_keep_the_shift():
    """padding.py has two of these. A change to one and not the other is worse than
    a change to both, because half the runs would still look right."""
    import inspect

    from verl.workers.utils import padding

    src = inspect.getsource(padding)
    assert src.count("seq_offset - resp_len - 1 : seq_offset - 1") == 2, (
        "one of the two value-extraction paths no longer left-shifts"
    )


def test_the_turn_estimator_reads_the_value_at_the_turn_start():
    """Behavioural, because the source-scanning version could not tell the two apart:
    a version anchoring at the LAST token of a turn also contains "start" and "gather".

    Two turns in one row with distinct values. The turn's value must be the one at the
    position it begins, not the one where it ends.
    """
    import numpy as np

    import vagen.algorithms  # noqa: F401  importing is what registers them
    from verl.trainer.ppo.core_algos import get_adv_estimator_fn

    # Through the registry, so this exercises the same dispatch verl uses rather than
    # the undecorated function -- which takes an AdvantageInputs, not verl's kwargs.
    # The import above is load-bearing: without it this file passes only when an
    # alphabetically earlier one happens to have imported the package first.
    compute_turn_level_gae = get_adv_estimator_fn("turn_level_gae")

    width = 6
    # turn 0 covers 0..1, turn 1 covers 3..5 (a gap marks the boundary)
    mask = torch.tensor([[1, 1, 0, 1, 1, 1]], dtype=torch.float32)
    values = torch.tensor([[10.0, 99.0, 0.0, 20.0, 99.0, 99.0]])
    scores = torch.zeros(1, width)
    scores[0, 1] = 1.0      # turn 0 earns 1
    scores[0, 5] = 1.0      # turn 1 earns 1

    batch = {"token_level_scores": scores, "responses": torch.zeros(1, width, dtype=torch.long),
             "response_mask": mask, "values": values}
    nt = {"group_idx": np.array(["g"], dtype=object),
          "traj_idx": np.array([0], dtype=object),
          "turn_idx": np.array([0], dtype=object)}

    adv, ret = compute_turn_level_gae(batch=batch, non_tensor_batch=nt,
                                     config=type("C", (), {"gamma": 1.0, "lam": 1.0})())

    # Real returns sit only at anchors; everywhere else carries the sentinel that says
    # "no supervision here".
    from vagen.training.trainer.logic import IGNORE_RETURN

    anchors = (ret[0] != IGNORE_RETURN).nonzero().flatten().tolist()
    assert anchors == [0, 3], (
        f"returns are anchored at {anchors}, not at the turn starts [0, 3]"
    )
    # The 99.0 values sit at the turns' last positions. If those were read, they would
    # dominate the returns -- which is how an end-of-turn anchor shows up.
    assert ret[0][anchors].abs().max().item() < 50.0, (
        f"an end-of-turn value leaked in: {ret[0].tolist()}"
    )
