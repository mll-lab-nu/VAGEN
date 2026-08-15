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

from vagen.envs.turn_limit import TurnLimit
from vagen.rewards.judge import shared_judge
from vagen.rewards.state_reward import DEFAULT_SCORE_BASE, TAGS, StateRewardSpec, StateRewardWrapper

#: The key an environment's own config uses to ask for a state reward.
#:
#: It lives in ``envs[].config`` next to the environment's other settings, not under
#: ``trainer``. Whether descriptions are scored, what a turn pays for one, and which judge
#: endpoint answers are all properties of the environment being run -- the trainer has no
#: opinion about any of them, and evaluation, which has no trainer at all, needs the same
#: three answers.
STATE_REWARD_KEY = "state_reward"


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


def state_reward_names(env_config: Optional[dict]) -> tuple[str, ...]:
    """Which state rewards this environment config switches on, in publication order.

    Read straight from the config so a caller can name the metrics it is about to record
    without building the environment first.
    """
    cfg = ((env_config or {}).get(STATE_REWARD_KEY) or {})
    return tuple(name for name in TAGS if (cfg.get(name) or {}).get("enable", False))


def build_env(env_cls: Any, env_config: Optional[dict], max_turns: Optional[int] = None):
    """Construct an environment from its config, ready to run.

    One function for both callers. Training and evaluation build environments from the
    same ``envs[].config`` block, so putting the assembly here is what makes a state
    reward work identically in both instead of being a training-only feature.

    ``max_turns`` is the environment's own budget and the environment enforces it -- see
    :class:`~vagen.envs.turn_limit.TurnLimit`. It is applied outermost so it counts the
    steps that actually happened, whatever else is wrapped around the environment.

    ``state_reward`` is popped before the environment sees the dict: the environments'
    own config dataclasses take ``**config`` and would raise ``TypeError`` on a key they
    do not declare, and adding the same field to every one of them is the drift this
    module exists to avoid.
    """
    config = dict(env_config or {})
    settings = config.pop(STATE_REWARD_KEY, None) or {}
    env = _with_state_reward(env_cls, env_cls(env_config=config), settings)
    return TurnLimit(env, int(max_turns)) if max_turns else env


def _with_state_reward(env_cls: Any, env: Any, settings: dict):
    enabled = {
        name: float(settings[name].get("reward", 0.0))
        for name in TAGS
        if (settings.get(name) or {}).get("enable", False)
    }
    if not enabled:
        return env

    spec = state_reward_spec_of(env_cls)
    if spec is None:
        raise ValueError(
            f"{getattr(env_cls, '__name__', env_cls)} has a state reward switched on in its "
            f"config but declares no STATE_REWARD_SPEC. Write one next to the environment "
            f"(see vagen/envs/sokoban/state_reward_spec.py) and set it on the class."
        )
    base_url = settings.get("judge_base_url")
    model = settings.get("judge_model")
    if not base_url or not model:
        # Off is a choice; on-but-unreachable is not. A judge that never answers scores
        # every description zero, which reads as a policy that cannot describe anything.
        raise ValueError(
            "a state reward is switched on but judge_base_url/judge_model are not set in "
            "the same config block. Start a judge first (scripts/launch_judge.sh) and "
            "point these at it, the way the eval configs point at their model server."
        )

    return StateRewardWrapper(
        env=env,
        spec=spec,
        judge=shared_judge(str(base_url), str(model)),
        enabled=enabled,
        score_base=float(settings.get("score_base", DEFAULT_SCORE_BASE)),
    )
