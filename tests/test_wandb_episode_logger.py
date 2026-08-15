"""The logger has to actually publish, and not on the training thread.

The failure this pins is silent by construction: an earlier version did the wandb.log
inside the Ray actor, where wandb.run is None, so it returned without publishing and
three validation runs produced no table at all while everything reported success.
"""

from __future__ import annotations

import pytest

from vagen.utils.wandb_episodes import EpisodeTableLogger, _PER_EPISODE, _Renderer


def _rows(n_episodes=4, turns=2):
    out = []
    for ep in range(n_episodes):
        for t in range(turns):
            out.append({
                "input": "p", "output": f"ep{ep}t{t}", "score": 1.0, "images": [],
                "group_idx": 0, "traj_idx": ep, "turn_idx": t,
                "conversation_id": f"c{ep}", "traj_success": float(ep % 2),
                "data_source": "sokoban", "episode_turns": turns,
            })
    return out


def test_the_renderer_groups_and_balances_without_wandb():
    """Rendering must not need a wandb run -- that is the whole point of the split."""
    episodes, step = _Renderer().render(_rows(4, 2), 4, 7)
    assert step == 7
    assert len(episodes) == 4
    assert sum(1 for e in episodes if e["success"]) == 2, "not balanced"
    assert all(e["turns"] == 2 for e in episodes), "row count reported instead of turns"


def test_it_publishes_through_wandb_log(monkeypatch):
    logged = {}

    class _Run: ...

    class _Table:
        def __init__(self, columns, data=None):
            self.columns, self.data = columns, list(data or [])

        def add_data(self, *row):
            self.data.append(list(row))

    fake = type("W", (), {
        "run": _Run(), "Table": _Table, "Html": lambda self=None, h=None: h,
        "log": staticmethod(lambda d, step=None: logged.update({"payload": d, "step": step})),
    })
    monkeypatch.setitem(__import__("sys").modules, "wandb", fake)

    lg = EpisodeTableLogger(use_ray=False)
    lg.submit(_rows(4, 2), 4, 11)

    assert "payload" in logged, "nothing was published"
    assert "val/episodes" in logged["payload"], f"wrong key: {list(logged['payload'])}"
    table = logged["payload"]["val/episodes"]
    assert table.columns[0] == "step"
    assert len(table.columns) == 1 + 4 * len(_PER_EPISODE)
    assert len(table.data) == 1, "one row per validation step"
    assert table.data[0][0] == 11


def test_history_accumulates_a_row_per_step(monkeypatch):
    logged = []

    class _Table:
        def __init__(self, columns, data=None):
            self.columns, self.data = columns, list(data or [])

        def add_data(self, *row):
            self.data.append(list(row))

    fake = type("W", (), {
        "run": object(), "Table": _Table, "Html": lambda self=None, h=None: h,
        "log": staticmethod(lambda d, step=None: logged.append(d["val/episodes"])),
    })
    monkeypatch.setitem(__import__("sys").modules, "wandb", fake)

    lg = EpisodeTableLogger(use_ray=False)
    for step in (1, 2, 3):
        lg.submit(_rows(2, 1), 2, step)
    assert [len(t.data) for t in logged] == [1, 2, 3], "the step slider needs one row per step"


def test_no_wandb_run_is_survivable(monkeypatch):
    fake = type("W", (), {"run": None, "log": staticmethod(lambda *a, **k: None)})
    monkeypatch.setitem(__import__("sys").modules, "wandb", fake)
    EpisodeTableLogger(use_ray=False).submit(_rows(2, 1), 2, 1)  # must not raise


def test_empty_input_does_nothing(monkeypatch):
    called = []
    fake = type("W", (), {"run": object(), "log": staticmethod(lambda *a, **k: called.append(1))})
    monkeypatch.setitem(__import__("sys").modules, "wandb", fake)
    lg = EpisodeTableLogger(use_ray=False)
    lg.submit([], 4, 1)
    lg.submit(_rows(1, 1), 0, 1)
    assert called == []
