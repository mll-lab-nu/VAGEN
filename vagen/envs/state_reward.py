"""How an environment says it can have its reasoning scored.

The mapping from environment to state-reward spec used to be a dict in the agent loop
(``STATE_REWARD_SPECS = {"Sokoban": ...}``). That is the drift shape this repo keeps
hitting: the loop has to be edited whenever an environment gains the capability, the
environment cannot be read to find out whether it has it, and a name typed one way in the
registry and another in the dict fails only once a run is up.

An environment declares it instead. Either inherit :class:`HasStateReward` and set
``STATE_REWARD_SPEC``, or simply define that attribute -- :func:`state_reward_spec_of`
duck-types, so a class does not have to change its bases to gain the capability.

The spec itself is env-specific and belongs next to the environment: what its objects
are, how to read their positions out of its internal state, and what to ask the judge.
``vagen/rewards/`` keeps only the parts that are not -- the judge client, the F1 scorer,
the span helpers and the wrapper that ties them together.
"""

from __future__ import annotations

from typing import Any, Optional

from vagen.rewards.state_reward import StateRewardSpec


class HasStateReward:
    """Mixin for an environment whose agent's descriptions can be scored.

    Subclasses set ``STATE_REWARD_SPEC``. Inheriting it is optional -- the attribute is
    what counts -- but the base makes the capability greppable and gives somewhere for
    this docstring to live.
    """

    #: What the reward needs from this environment. None means "not supported".
    STATE_REWARD_SPEC: Optional[StateRewardSpec] = None


def state_reward_spec_of(env_cls: Any) -> Optional[StateRewardSpec]:
    """The spec an environment class declares, or None if it declares none."""
    spec = getattr(env_cls, "STATE_REWARD_SPEC", None)
    return spec if isinstance(spec, StateRewardSpec) else None


def supports_state_reward(env_cls: Any) -> bool:
    return state_reward_spec_of(env_cls) is not None
