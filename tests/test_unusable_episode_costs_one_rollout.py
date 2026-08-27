"""An unusable episode must cost one rollout, not the training run.

`EpisodeUnusable` exists so that a rollout the policy made unusable -- it sampled an
image token, it blew the context, compaction stopped making progress -- is dropped and
the batch continues. verl's `asyncio.gather` has no `return_exceptions`, so anything that
escapes takes the whole step, and a step that raises takes the run.

The handler was there and correct. `_outputs` was called *after* it, and `_outputs` is
where `_refuse_sampled_vision_tokens` runs -- so the one `EpisodeUnusable` raised outside
`run_episode` escaped. `sokoban_turn_nosr_fmt` died at step 99 of 401, 2h33m in, because
a single rollout emitted a vision token. The class, its docstring and the handler all
said "drop the episode"; the placement of one line said otherwise.
"""

from __future__ import annotations

import ast
import inspect

import pytest

from vagen.training.agent_loop.gym_loop import GymLoop, SampledVisionToken
from vagen.rollout.client import EpisodeUnusable


def _run_body():
    src = inspect.getsource(GymLoop.run_episode_and_outputs) if hasattr(
        GymLoop, "run_episode_and_outputs") else None
    if src is None:
        # The method is whichever one wraps run_episode in a try/except EpisodeUnusable.
        for name, fn in vars(GymLoop).items():
            if not callable(fn):
                continue
            try:
                s = inspect.getsource(fn)
            except (OSError, TypeError):
                continue
            if "except EpisodeUnusable" in s:
                return name, s
        pytest.fail("no method guards EpisodeUnusable any more")
    return "run_episode_and_outputs", src


def test_sampled_vision_token_is_an_unusable_episode_not_a_fatal_error():
    """The classification itself. Making it a plain ValueError would kill the run even
    with the placement right."""
    assert issubclass(SampledVisionToken, EpisodeUnusable)


def test_outputs_is_built_inside_the_guard():
    """★ The bug. `_outputs` raises `SampledVisionToken`, so it has to be inside the
    `try`, not after it.

    Checked with `ast` rather than by string search: `self._outputs(...)` appears in the
    source either way, and what matters is *where* -- inside the Try body, or after it.
    """
    name, src = _run_body()
    import textwrap

    tree = ast.parse(textwrap.dedent(src))
    tries = [n for n in ast.walk(tree) if isinstance(n, ast.Try)
             and any("EpisodeUnusable" in ast.unparse(h.type or ast.Constant(None))
                     for h in n.handlers)]
    assert tries, f"{name} no longer guards EpisodeUnusable"

    guarded = any("_outputs" in ast.unparse(t.body) for t in tries)
    assert guarded, (
        f"{name} builds _outputs outside the EpisodeUnusable guard. "
        "_refuse_sampled_vision_tokens runs in there, so its exception escapes and one "
        "bad rollout kills the training step -- and with it the run."
    )

    # ...and nowhere else, or the unguarded copy is still reachable.
    fn = tree.body[0]
    after = [st for st in fn.body if not isinstance(st, ast.Try)]
    assert not any("_outputs" in ast.unparse(st) for st in after), (
        "there is still a call to _outputs outside the try; the guarded one is dead code "
        "or the unguarded one runs on the happy path"
    )


def test_the_handler_swallows_and_returns_no_rows():
    """Dropping means returning zero rows for that episode -- not None, which would fail
    downstream, and not a partial row, which would be trained on."""
    name, src = _run_body()
    import textwrap

    tree = ast.parse(textwrap.dedent(src))
    handler = next(h for n in ast.walk(tree) if isinstance(n, ast.Try)
                   for h in n.handlers
                   if "EpisodeUnusable" in ast.unparse(h.type or ast.Constant(None)))
    body = ast.unparse(handler.body)
    assert "return []" in body, f"the EpisodeUnusable handler in {name} does not drop the episode"
    assert "logger" in body, "a dropped episode must say so; silent drops shrink the batch invisibly"
