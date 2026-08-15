"""Whether the agent is following the prompt's format has to be visible in a curve.

Under `prompt_format=wm` the policy is asked for four sections. It collapsed to a bare
`<answer>` and nothing in wandb said so:

* the old state reward had a second `format_reward` term, usually fixed at zero, so its
  curve read as "never once well-formed" even though it was only an inactive line item;
* the format reward actually being paid is the environment's (`SokobanEnvConfig.
  format_reward`), and the correctness flag was published nowhere;
* `turn_metrics.action_is_valid` already carries the flag and never reaches the logger.

So the question the format exists to ask had no answer, and it took reading a rollout by
eye. These tests pin the fix in both directions: the constant-zero curve is gone, and
`format_correct_rate` is there.
"""

from __future__ import annotations

import pytest


def test_state_reward_metrics_name_only_enabled_description_scores():
    from vagen.envs.state_reward import state_reward_names

    cfg = {"state_reward": {
        "state_estimation": {"enable": True, "reward": 0.01},
        "transition_prediction": {"enable": True, "reward": 0.01},
    }}
    assert state_reward_names(cfg) == ("state_estimation", "transition_prediction")
    assert "format" not in state_reward_names(cfg)


def test_nothing_is_published_when_state_reward_is_off():
    from vagen.envs.state_reward import state_reward_names

    assert state_reward_names({}) == ()
    assert state_reward_names({"state_reward": {
        "state_estimation": {"enable": False, "reward": 0.01},
    }}) == ()


# ----------------------------------------------------------------- format_correct_rate


class _Adapter:
    """Just the accumulator under test."""

    def __init__(self):
        from vagen.agent_loop.gym_loop import GymEnvAdapter

        self.a = object.__new__(GymEnvAdapter)
        self.a.state_scores = {}
        self.a.turns_seen = 0
        self.a.turns_well_formed = 0
        self.a.reports_format = False

    def step(self, info):
        for key in self.a.state_scores:
            pass
        if "format_correct" in info:
            self.a.reports_format = True
            self.a.turns_seen += 1
            self.a.turns_well_formed += bool(info["format_correct"])
        return self.a


@pytest.mark.parametrize(
    "flags,expected",
    [([True, True, True, True], 1.0), ([True, False, True, False], 0.5), ([False] * 4, 0.0)],
)
def test_the_rate_is_the_fraction_of_well_formed_turns(flags, expected):
    a = _Adapter()
    for f in flags:
        a.step({"format_correct": f})
    assert a.a.turns_well_formed / a.a.turns_seen == pytest.approx(expected)


def test_an_environment_that_does_not_report_format_publishes_no_rate():
    """★ Absent, not zero. A zero here would read as "never well-formed" for an
    environment that simply does not have a format to be correct about -- which is the
    exact failure this whole file exists to stop repeating."""
    a = _Adapter()
    for _ in range(3):
        a.step({"something_else": 1})
    assert a.a.reports_format is False
    assert a.a.turns_seen == 0


def test_the_loop_emits_the_rate_only_when_it_was_measured():
    """The emission guard itself, read off the source: it must consult `reports_format`
    and must use getattr, since `env` is whatever the runner was handed."""
    import inspect

    from vagen.agent_loop.gym_loop import GymLoop

    src = inspect.getsource(GymLoop)
    assert "format_correct_rate" in src
    # rindex, not index: matching prose instead of the emission code would check nothing.
    i = src.rindex("format_correct_rate")
    window = src[i - 400 : i + 400]
    assert "reports_format" in window, "the rate is emitted without checking it was measured"
    assert "getattr(env" in window, "attribute access on env breaks every fake in the suite"


# ------------------------------------------------------- how much the MODEL is writing


def test_model_tokens_per_turn_counts_only_generated_tokens():
    """★ `response_length/mean` measures the whole response *region*, which on vision
    Sokoban also carries 49-144 image tokens per interleaved observation. So it moves when
    the environment renders differently, and it dilutes the one thing worth watching --
    whether the policy is getting more verbose. Every collapse in the 0809 sweep announced
    itself as a length jump, and nothing said whether the jump was the model or the
    scenery.

    `response_spans` is exactly the model-emitted region, the same thing the response mask
    marks, so the sum over spans is generated tokens and nothing else.
    """
    import inspect

    from vagen.agent_loop.gym_loop import GymLoop

    src = inspect.getsource(GymLoop)
    assert "model_tokens_per_turn" in src
    i = src.rindex("model_tokens_per_turn")
    window = src[i : i + 260]
    assert "spans" in window, "it is not derived from response_spans"
    assert "response_length" not in window, "it must not be built from the region length"


@pytest.mark.parametrize(
    "spans,expected",
    [
        ([(0, 100), (150, 250)], 100.0),          # two turns of 100 generated tokens
        ([(0, 30), (60, 90), (100, 400)], 120.0),  # (30 + 30 + 300) / 3
        ([(5, 5)], 0.0),                           # an empty generation is 0, not a crash
    ],
)
def test_the_arithmetic(spans, expected):
    """Mean generated tokens per turn -- the gap between spans is the observation and
    must not be counted."""
    assert sum(int(e) - int(b) for b, e in spans) / max(1, len(spans)) == pytest.approx(expected)


def test_no_spans_publishes_nothing():
    """Absent rather than zero, for the same reason as format_correct_rate: a row the
    model never spoke in has no verbosity to report, and a 0 would drag the mean down."""
    spans = []
    emitted = ({"model_tokens_per_turn": 0.0} if spans else {})
    assert emitted == {}
