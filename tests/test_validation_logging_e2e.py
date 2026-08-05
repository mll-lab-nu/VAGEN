"""The whole validation-logging path, without a GPU.

Everything from the columns a merged validation batch carries, through regrouping and
balancing, to the table that reaches wandb. Every bug in this path so far has been silent
-- a dropped column, a truthy list of Nones, a skip-list, a queue nothing drained -- and
each cost a thirty-minute GPU run to notice. This runs in a second.

The three context policies differ only in the shape of what arrives, so all three are
exercised here as data rather than as separate runs.
"""

from __future__ import annotations

import pytest

from vagen.trainer.mixin import VagenV0Mixin
from vagen.utils.episode_log import rows_from_validation
from vagen.utils.wandb_episodes import _PER_EPISODE, EpisodeTableLogger

PIL = pytest.importorskip("PIL.Image")


class _Cfg(dict):
    def __getattr__(self, k):
        try:
            return self[k]
        except KeyError as e:
            raise AttributeError(k) from e


class _Table:
    def __init__(self, columns, data=None):
        self.columns, self.data = columns, list(data or [])

    def add_data(self, *row):
        self.data.append(list(row))


def _fake_wandb(sink):
    return type("W", (), {
        "run": object(), "Table": _Table, "Html": staticmethod(lambda h: h),
        "log": staticmethod(lambda d, step=None: sink.append((d, step))),
    })


class _Trainer(VagenV0Mixin):
    """Only what the logging path touches."""

    def __init__(self, n=4):
        self.config = _Cfg(trainer=_Cfg({
            "log_val_generations": n, "project_name": "p", "experiment_name": "e",
            "logger": ["console", "wandb"], "replace_image_tokens_for_logging": False,
        }))
        self.global_steps = 5
        self._vagen_val_logger = EpisodeTableLogger(use_ray=False)

    def _maybe_log_val_generations_base(self, *a, **k):
        raise AssertionError("fell back to verl's flat table")

    # what super() would resolve to
    def __getattr__(self, name):
        raise AttributeError(name)


def _merged_validation_batch(mode, n_episodes=4):
    """What _validate hands the logger, after the per-turn rows have been merged.

    One row per episode -- that is what the merge produces -- carrying the identity of
    the episode it stands for.
    """
    shape = {"concat": (1, 5), "no_concat": (5, 5), "compact": (2, 6)}[mode]
    n_conversations, n_turns = shape
    inputs, outputs, scores, images = [], [], [], []
    ex = {k: [] for k in ("episode_id", "group_idx", "traj_idx", "turn_idx",
                          "conversation_id", "traj_success", "episode_turns",
                          "n_conversations", "turn_records")}
    for e in range(n_episodes):
        inputs.append("system prompt")
        outputs.append(f"<observation>ep{e}</observation><answer>Up</answer>")
        scores.append(1.0 + e)
        images.append([PIL.new("RGB", (16, 16), (e, e, e)) for _ in range(n_turns)])
        ex["episode_id"].append(f"EP{e}")
        ex["group_idx"].append(f"uid{e}")
        ex["traj_idx"].append(0)
        ex["turn_idx"].append(0)
        ex["conversation_id"].append("c0")
        ex["traj_success"].append(float(e % 2))
        ex["episode_turns"].append(n_turns)
        ex["n_conversations"].append(n_conversations)
        per_conv = max(1, n_turns // n_conversations)
        ex["turn_records"].append([
            {
                "turn_idx": k,
                "conversation_id": f"c{k // per_conv}",
                "images": [PIL.new("RGB", (16, 16), (k, k, k))],
                "response": f"<answer>Up</answer> turn {k}",
                "observation": f"obs after {k}",
            }
            for k in range(n_turns)
        ])
    return inputs, outputs, scores, images, ex


@pytest.mark.parametrize("mode,want_turns,want_convs", [
    ("concat", 5, 1), ("no_concat", 5, 5), ("compact", 6, 2),
])
def test_each_context_policy_logs_its_true_shape(monkeypatch, mode, want_turns, want_convs):
    sink = []
    monkeypatch.setitem(__import__("sys").modules, "wandb", _fake_wandb(sink))
    t = _Trainer(n=4)
    t._maybe_log_val_generations(*_merged_validation_batch(mode)[:3],
                                 images=_merged_validation_batch(mode)[3],
                                 extras=_merged_validation_batch(mode)[4])
    assert sink, "nothing reached wandb"
    payload, step = sink[-1]
    assert "val/episodes" in payload, f"wrong key: {list(payload)}"
    table = payload["val/episodes"]
    assert step == 5
    assert len(table.data) == 1, "one row per validation step"
    n_ep = (len(table.columns) - 1) // len(_PER_EPISODE)
    row = table.data[0]
    # The columns you can act on: which episode, what it earned, whether it won, and
    # the transcript. Counts live inside the transcript now, where the turns are.
    for i in range(n_ep):
        assert row[table.columns.index(f"ep{i}_episode")], f"{mode}: episode id missing"
        assert row[table.columns.index(f"ep{i}_reward")] is not None, f"{mode}: reward missing"
        assert row[table.columns.index(f"ep{i}_success")] is not None, f"{mode}: verdict missing"
    html = row[table.columns.index("ep0_html")]
    assert html.count("<b>turn") == want_turns, f"{mode}: expected {want_turns} turns, got {html.count(chr(60)+chr(98)+chr(62)+chr(116))}"
    assert html.count("new conversation") == want_convs - 1, f"{mode}: conversation seams wrong"


def test_the_sample_is_balanced_and_carries_frames(monkeypatch):
    sink = []
    monkeypatch.setitem(__import__("sys").modules, "wandb", _fake_wandb(sink))
    t = _Trainer(n=4)
    i, o, s, im, ex = _merged_validation_batch("concat", n_episodes=8)
    t._maybe_log_val_generations(i, o, s, images=im, extras=ex)
    table = sink[-1][0]["val/episodes"]
    n_ep = (len(table.columns) - 1) // len(_PER_EPISODE)
    row = table.data[0]
    succ = [row[table.columns.index(f"ep{i}_success")] for i in range(n_ep)]
    assert sum(1 for x in succ if x) == n_ep // 2, f"not balanced: {succ}"
    html = row[table.columns.index("ep0_html")]
    assert html.count("base64,") == 5, "frames missing from the transcript"


def test_rows_from_validation_keeps_every_identity_field():
    i, o, s, im, ex = _merged_validation_batch("compact")
    rows = rows_from_validation(i, o, s, im, ex)
    for key in ("episode_id", "turn_idx", "conversation_id", "episode_turns", "n_conversations"):
        assert all(r[key] is not None for r in rows), f"{key} lost before grouping"


def test_the_transcript_interleaves_frames_and_text_in_order():
    """system prompt, then per turn: what the agent saw, then what it wrote.

    A gallery of frames above a wall of text is not the same artefact: a turn's response
    only means something next to the board it was looking at.
    """
    from vagen.utils.episode_log import episode_html

    turns = [{
        "input": "SYSTEM PROMPT",
        "turn_records": [
            {"turn_idx": 0, "conversation_id": "c0",
             "images": [PIL.new("RGB", (8, 8), (1, 1, 1))],
             "response": "RESPONSE-ZERO", "observation": "OBS-ZERO"},
            {"turn_idx": 1, "conversation_id": "c0",
             "images": [PIL.new("RGB", (8, 8), (2, 2, 2))],
             "response": "RESPONSE-ONE", "observation": ""},
        ],
    }]
    h = episode_html(turns)
    order = [h.index(x) for x in
             ("SYSTEM PROMPT", "base64,", "RESPONSE-ZERO", "OBS-ZERO", "RESPONSE-ONE")]
    assert order == sorted(order), f"out of order: {order}"
    assert h.index("RESPONSE-ZERO") < h.rindex("base64,") < h.index("RESPONSE-ONE"), (
        "the second turn's frame is not between the two responses"
    )


def test_conversation_boundaries_are_marked_between_turns():
    from vagen.utils.episode_log import episode_html

    turns = [{
        "input": "P",
        "turn_records": [
            {"turn_idx": 0, "conversation_id": "c0", "images": [], "response": "BEFORE", "observation": ""},
            {"turn_idx": 1, "conversation_id": "c1", "images": [], "response": "AFTER", "observation": ""},
        ],
    }]
    h = episode_html(turns)
    assert h.index("BEFORE") < h.index("new conversation") < h.index("AFTER")
    assert h.count("<b>turn") == 2
