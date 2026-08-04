"""One gym agent loop for every context policy.

Replaces the pair of near-identical loops -- 427 lines for concat, 304 for no-concat --
whose only real differences were how much history went into the prompt and whether a
turn produced one row or many. Both are now the harness's answer to a single question
(§4), so there is one loop.

What is left here is the glue verl needs: build the environment, adapt it to the runner's
contract, and turn the client's rows into ``AgentLoopOutput``.
"""

from __future__ import annotations

import logging
import os
from typing import Any
from uuid import uuid4

from verl.experimental.agent_loop.agent_loop import AgentLoopOutput, register
from verl.utils.rollout_trace import rollout_trace_op

from vagen.agent_loop.base import VagenGymAgentLoopBase
from vagen.agent_loop.gym_agent_loop import _normalize_images, convert_obs_to_content, extract_success
from vagen.agent_loop.verl_client import VerlClient
from vagen.core.harness import CompactHarness, ConcatHarness, NoConcatHarness
from vagen.core.runner import run_episode

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

HARNESSES = {"concat": ConcatHarness, "no_concat": NoConcatHarness, "compact": CompactHarness}


class GymEnvAdapter:
    """The gym environment, in the shape the runner expects.

    Two differences to bridge: the env reports ``done`` where the runner distinguishes
    terminated from truncated, and it speaks in observation dicts where the harness
    speaks in messages.
    """

    def __init__(self, env, env_name: str, kwargs: dict):
        self.env, self.env_name, self.kwargs = env, env_name, kwargs
        self.success = False

    async def reset(self, seed=None):
        obs, info = await self.env.reset(seed=seed)
        return self._message(obs), info

    async def system_prompt(self):
        return self._message(await self.env.system_prompt(), role="system")

    async def step(self, action: str, response_token_ids=None, tokenizer=None):
        try:
            obs, reward, done, info = await self.env.step(action)
        except Exception as exc:  # noqa: BLE001 - one bad action must not kill the batch
            logger.error("environment %r failed on action %r: %s", self.env_name, action, exc)
            # Ends the episode rather than pretending the step happened. Reported as
            # terminated, since there is no state left to bootstrap from.
            return self._message({"obs_str": "Environment Error"}), 0.0, True, False, {"env_error": True}

        self.success = extract_success(info)
        return self._message(obs), reward, bool(done), False, info

    async def close(self):
        await self.env.close()

    def _message(self, obs: dict, role: str = "user") -> dict:
        images = _normalize_images(obs.get("multi_modal_input", {}).get("<image>", []) or [])
        return {"role": role, "content": convert_obs_to_content(obs, **self.kwargs), "images": images}


@register("gym_agent_v2")
class GymLoop(VagenGymAgentLoopBase):
    """Runner + harness + client. The mode comes from config, not from the class."""

    @rollout_trace_op
    async def run(self, sampling_params: dict[str, Any], **kwargs) -> list[AgentLoopOutput]:
        env_cls = self.resolve_env_class(kwargs["env_name"])
        env = GymEnvAdapter(env_cls(env_config=kwargs["config"]), kwargs["env_name"], kwargs)

        harness = self._build_harness()
        per_turn = min(int(kwargs.get("response_length_per_turn") or self.response_length), self.response_length)
        client = VerlClient(
            self.server_manager,
            self.tokenizer,
            self.processor,
            apply_chat_template_kwargs=self.apply_chat_template_kwargs,
            mm_processor_kwargs=self._get_mm_processor_kwargs(),
            sampling_params=sampling_params,
            request_id=uuid4().hex,
            response_limit=per_turn,
        )

        # No silent default: falling back to a single turn would look like a working
        # run whose episodes all stop after one step, which is nearly invisible --
        # every row is well-formed, just short.
        if not kwargs.get("max_turns"):
            raise KeyError(
                f"the dataset row for env {kwargs['env_name']!r} carries no max_turns; "
                f"available keys: {sorted(kwargs)}"
            )
        result = await run_episode(
            env, harness, client, seed=kwargs["seed"], max_turns=int(kwargs["max_turns"])
        )
        return self._outputs(client, env, result, kwargs)

    def _build_harness(self):
        mode = self.config.trainer.get("harness", None)
        if mode is None:
            # Fall back to the flag the existing scripts set, so the same run
            # configuration selects the same layout as before.
            mode = "concat" if self.config.trainer.get("concat_multi_turn", True) else "no_concat"
        if mode not in HARNESSES:
            raise ValueError(f"unknown harness {mode!r}; choose from {sorted(HARNESSES)}")
        if mode == "compact":
            return CompactHarness(budget=int(self.config.trainer.compact_budget))
        return HARNESSES[mode]()

    def _outputs(self, client, env, result, kwargs) -> list[AgentLoopOutput]:
        rows = client.rows()
        outputs = []
        for turn_idx, row in enumerate(rows):
            images = client.images(row.conversation_id)
            outputs.append(
                AgentLoopOutput(
                    prompt_ids=row.prompt_ids[-self.prompt_length :],
                    response_ids=row.response_ids[: self.response_length],
                    response_mask=row.response_mask[: self.response_length],
                    multi_modal_data={"image": images} if images else {},
                    response_logprobs=row.logprobs[: self.response_length] or None,
                    reward_score=float(sum(row.scores)),
                    num_turns=1,
                    metrics={},
                    extra_fields={
                        "reward_extra_info": {"traj_success": float(env.success)},
                        "image_data": images,
                        "last_turn": turn_idx == len(rows) - 1,
                        "group_idx": kwargs["group_idx"],
                        "traj_idx": kwargs["traj_idx"],
                        "turn_idx": turn_idx,
                    },
                )
            )
        return outputs
