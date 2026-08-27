"""The gym environment, in the shape ``core/runner.py`` expects.

Lives here rather than in ``agent_loop/`` because both callers need it and only one of
them may import verl: training goes through ``agent_loop/gym_loop.py``, evaluation through
``evaluate/``, and the eval path is deliberately importable without the training stack.
It was duplicated once already -- ``evaluate/vision_workflow.py`` grew its own turn loop
and its own observation handling, and the two drifted for five months.
"""

from __future__ import annotations

import inspect
import logging

from vagen.envs._common.observations import (
    _normalize_images,
    convert_obs_to_content,
    extract_success,
)

logger = logging.getLogger(__name__)


def _accepts_response(step) -> bool:
    """Whether an env's ``step`` wants the response tokens and the tokenizer."""
    try:
        return "response_token_ids" in inspect.signature(step).parameters
    except (TypeError, ValueError):
        return False


class GymEnvAdapter:
    """The gym environment, in the shape the runner expects.

    Two differences to bridge: the env reports ``done`` where the runner distinguishes
    terminated from truncated, and it speaks in observation dicts where the harness
    speaks in messages.
    """

    def __init__(self, env, env_name: str, kwargs: dict, score_names=()):
        self.env, self.env_name, self.kwargs = env, env_name, kwargs
        self.success = False
        # ★ Only the description scores that are actually switched on. Anything in here
        # is published as a `<name>_reward` extra field, and verl turns every extra field
        # into a val_aux curve -- so declaring them unconditionally draws flat zero lines
        # that read as "the reward is on and the agent scores nothing" rather than "the
        # reward is off". Format correctness has its own rate below; state reward has no
        # second format-reward line item.
        #
        # Empty when nothing is enabled, which is the point: a metric that does not exist
        # is the honest representation of a term that is not being computed.
        self.state_scores: dict[str, float] = {name: 0.0 for name in score_names}
        # ★ How often the environment judged the turn well-formed. Nothing else reports
        # this: `format_reward` (the state-reward one) is a constant zero by config, the
        # environment's format reward is not published at all, and `turn_metrics` never
        # reaches the logger. So the one question the world-modeling prompt exists to ask
        # -- is the model writing the sections? -- had no curve, and a policy that
        # collapsed to a bare `<answer>` was invisible until someone read a rollout by
        # eye. Counted only for environments that report `format_correct`, so a metric
        # appearing at all means it is being measured.
        self.turns_seen = 0
        self.turns_well_formed = 0
        self.reports_format = False

    async def reset(self, seed=None):
        obs, info = await self.env.reset(seed=seed)
        return self._message(obs), info

    async def system_prompt(self):
        return self._message(await self.env.system_prompt(), role="system")

    async def step(self, action: str, response_token_ids=None, tokenizer=None):
        try:
            # The response and tokenizer go through when the wrapped env asks for them.
            # A plain gym env does not, and a reward wrapper that scores spans of the
            # response cannot do its job without them -- dropping them here does not
            # fail, it silently pays a scalar at the last token instead of placing the
            # score on the tokens that earned it.
            kwargs = {}
            if _accepts_response(self.env.step):
                kwargs = {"response_token_ids": response_token_ids, "tokenizer": tokenizer}
            obs, reward, done, info = await self.env.step(action, **kwargs)
        except Exception as exc:  # noqa: BLE001 - one bad action must not kill the batch
            logger.error("environment %r failed on action %r: %s", self.env_name, action, exc)
            # Ends the episode rather than pretending the step happened. Reported as
            # terminated, since there is no state left to bootstrap from.
            return self._message({"obs_str": "Environment Error"}), 0.0, True, False, {"env_error": True}

        self.success = extract_success(info)
        for key in self.state_scores:
            self.state_scores[key] += float(info.get(f"state_reward/{key}", 0.0) or 0.0)
        if "format_correct" in info:
            self.reports_format = True
            self.turns_seen += 1
            self.turns_well_formed += bool(info["format_correct"])
        # An environment that stopped because it ran out of turns says so through
        # `info["truncated"]` (see vagen/envs/turn_limit.py). Truncated and terminated are
        # not interchangeable: the first should bootstrap from V, the second must not.
        # Collapsing them was how "ran out of time" became "worth zero from here on".
        truncated = bool((info or {}).get("truncated", False))
        return self._message(obs), reward, bool(done) and not truncated, truncated, info

    async def close(self):
        await self.env.close()

    def _message(self, obs: dict, role: str = "user") -> dict:
        mm = obs.get("multi_modal_input", {}) or {}
        # Only images are carried. A video would count as a picture -- image_token_ids
        # returns the video pad id too -- while nothing lifts it into multi_modal_data,
        # so the placeholder runs and the frames stop being 1:1 and the row dies deep in
        # get_rope_index with `'NoneType' object is not subscriptable`, naming nothing.
        # Refuse where the environment can be pointed at instead.
        unsupported = [k for k in mm if k not in ("<image>",) and mm.get(k)]
        if unsupported:
            raise NotImplementedError(
                f"environment {self.env_name!r} returned {unsupported} in multi_modal_input; "
                f"only '<image>' is carried into training. Supporting another modality "
                f"means lifting it here and adding it to multi_modal_data in _outputs."
            )
        images = _normalize_images(mm.get("<image>", []) or [])
        return {"role": role, "content": convert_obs_to_content(obs, **self.kwargs), "images": images}
