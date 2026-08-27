"""What ``@register`` actually registered.

On 2026-08-10 a helper function was inserted between the ``@register(...)`` decorators
and ``class GymLoop``. The decorators applied to the function, so ``gym_agent`` resolved
to a two-argument helper and ``GymLoop`` was registered nowhere. Every test passed --
nothing in the suite went through the registry -- and three cluster jobs died at step 0
with ``resolve_reward_placement() got an unexpected keyword argument 'trainer_config'``,
an error that names the symptom and hides the cause.

The registry is the only thing standing between the config's ``gym_agent`` string and the
class that runs an episode, and a decorator can silently land on the wrong object.
"""

from __future__ import annotations

import pytest

import vagen.training.agent_loop.gym_loop  # noqa: F401  importing is what registers
from verl.experimental.agent_loop.agent_loop import _agent_loop_registry as REGISTRY


@pytest.mark.parametrize("name", ["gym_agent", "gym_agent_v2"])
def test_the_name_resolves_to_the_loop_class(name):
    """★ The guard. ``gym_agent`` is what ``gym_agent_dataset.py`` emits and what
    ``configs/agent_v2.yaml`` dispatches on; it must reach ``GymLoop`` and nothing else."""
    entry = REGISTRY.get(name)
    assert entry is not None, f"{name} is not registered; the config will fail to resolve"
    assert entry["_target_"] == "vagen.training.agent_loop.gym_loop.GymLoop", (
        f"{name} resolves to {entry['_target_']!r}. A decorator landed on the wrong "
        "object -- check for anything inserted between @register and `class GymLoop`."
    )


def test_the_target_is_importable_and_is_a_loop():
    """`_target_` is a string, so a rename leaves the registry pointing at nothing and
    only fails once a rollout starts. Resolve it here instead."""
    import importlib

    from vagen.training.agent_loop.gym_loop import GymLoop

    module, _, attr = REGISTRY["gym_agent"]["_target_"].rpartition(".")
    obj = getattr(importlib.import_module(module), attr)
    assert obj is GymLoop
    assert isinstance(obj, type), "the registered target is not a class"
    assert hasattr(obj, "run"), "the registered target has no run(); it cannot drive an episode"


def test_no_module_level_function_sits_between_the_decorator_and_the_class():
    """The structural version of the same check, so the failure is reported where the
    mistake is rather than as a TypeError inside Ray three minutes into a job."""
    import inspect

    import vagen.training.agent_loop.gym_loop as mod

    src = inspect.getsource(mod)
    tail = src[src.index('@register("gym_agent")'):]
    head = tail[: tail.index("class GymLoop")]
    offenders = [ln for ln in head.splitlines() if ln.startswith(("def ", "async def ", "class "))]
    assert not offenders, (
        f"these are defined between @register and `class GymLoop`, so the decorators "
        f"apply to them instead: {offenders}"
    )
