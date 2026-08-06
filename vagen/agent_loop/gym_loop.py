"""One gym agent loop for every context policy.

Replaces the pair of near-identical loops -- 427 lines for concat, 304 for no-concat --
whose only real differences were how much history went into the prompt and whether a
turn produced one row or many. Both are now the harness's answer to a single question
(§4), so there is one loop.

What is left here is the glue verl needs: build the environment, adapt it to the runner's
contract, and turn the client's rows into ``AgentLoopOutput``.
"""

from __future__ import annotations

import inspect
import logging
import os
from typing import Any
from uuid import uuid4

from verl.experimental.agent_loop.agent_loop import AgentLoopOutput, cap_token_ids, register
from verl.utils.rollout_trace import rollout_trace_op

from vagen.agent_loop.base import VagenGymAgentLoopBase
from vagen.agent_loop.obs import _normalize_images, convert_obs_to_content, extract_success
from vagen.rewards import sokoban as sokoban_spec
from vagen.rewards.judge import shared_judge
from vagen.rewards.state_reward import TAGS, StateRewardWrapper
from vagen.agent_loop.verl_client import VerlClient
from vagen.harness import HARNESSES, build_harness
from vagen.harness.budget import Budgets, check as check_budgets, default_summary_budget
from vagen.harness.compact import CompactHarness
from vagen.core.runner import run_episode

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

# Environments that can score the agent's descriptions. Keyed by the registry name.
STATE_REWARD_SPECS = {"Sokoban": sokoban_spec.SPEC}


def _spans_within(spans, keep: int) -> list[tuple[int, int]]:
    """The response spans that survive a truncation to ``keep`` tokens, clipped to it."""
    out = []
    for start, end in spans or ():
        if start >= keep:
            break
        out.append((int(start), min(int(end), keep)))
    return out


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

    def __init__(self, env, env_name: str, kwargs: dict):
        self.env, self.env_name, self.kwargs = env, env_name, kwargs
        self.success = False
        # Summed over the episode. Reported on every row whether or not the agent ever
        # produced a description: verl reads the set of extra keys from the first row,
        # so a key missing there hides the metric for the whole batch.
        self.state_scores: dict[str, float] = {name: 0.0 for name in (*TAGS, "format")}

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
        # No silent default: falling back to a single turn would look like a working
        # run whose episodes all stop after one step, which is nearly invisible --
        # every row is well-formed, just short. Read here rather than at the call to
        # run_episode because the auxiliary reward budget is divided by it.
        if not kwargs.get("max_turns"):
            raise KeyError(
                f"the dataset row for env {kwargs['env_name']!r} carries no max_turns; "
                f"available keys: {sorted(kwargs)}"
            )
        max_turns = int(kwargs["max_turns"])
        # The loop is the only thing that knows what one episode is: exactly this call.
        # Identity was being taken from (group_idx, traj_idx), which is the dataset's
        # axis, not ours -- measured at validation it was unique per row, so every row
        # grouped as its own one-turn episode. Minted here, alongside conversation_id
        # and turn_idx, so the three cannot disagree about what they identify.
        episode_id = uuid4().hex

        env_cls = self.resolve_env_class(kwargs["env_name"])
        env = GymEnvAdapter(
            self._maybe_state_reward(
                env_cls(env_config=kwargs["config"]), kwargs["env_name"], max_turns
            ),
            kwargs["env_name"],
            kwargs,
        )

        per_turn = min(int(kwargs.get("response_length_per_turn") or self.response_length), self.response_length)
        harness = self._build_harness(per_turn, max_turns,
                                      per_turn_configured=bool(kwargs.get("response_length_per_turn")))
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

        result = await run_episode(env, harness, client, seed=kwargs["seed"], max_turns=max_turns)
        return self._outputs(client, env, result, kwargs, episode_id)

    def _maybe_state_reward(self, env, env_name: str, max_turns: int = 1):
        """Wrap the environment so the reasoning is scored, if configured.

        Off by default: it needs a judge endpoint, and a run that silently scored
        nothing would be indistinguishable from one that scored badly.
        """
        cfg = self.config.trainer.get("state_reward", {}) or {}
        enabled = {
            name: float(cfg[name].get("weight", 0.5))
            for name in TAGS
            if (cfg.get(name) or {}).get("enable", False)
        }
        if not enabled:
            return env

        # The configured weights are relative; the budget sets the scale. A whole episode
        # described perfectly is worth `budget` in total, against 1 for solving the level,
        # so the auxiliary signal cannot outbid the task it exists to support. Measured
        # before this: eight trajectories, every one with traj_success 0, scoring up to
        # 2.25 -- all of it description.
        #
        # Derived rather than written down, because a per-turn constant silently doubles
        # the episode's auxiliary total the day someone raises max_turns.
        budget = float(cfg.get("budget", 1.0))
        total = sum(enabled.values()) or 1.0
        enabled = {n: budget * (w / total) / max(1, int(max_turns)) for n, w in enabled.items()}

        spec = STATE_REWARD_SPECS.get(env_name)
        if spec is None:
            raise ValueError(
                f"a state reward is on but {env_name!r} has no spec; add one to STATE_REWARD_SPECS "
                f"(available: {sorted(STATE_REWARD_SPECS)})"
            )
        return StateRewardWrapper(
            env=env,
            spec=spec,
            judge=shared_judge(cfg["judge_base_url"], cfg["judge_model"]),
            enabled=enabled,
            format_reward=float(cfg.get("format_reward", 0.1)),
        )

    def _overflow_hint(self) -> str:
        """What to change, in terms of the mode that is running.

        The budget is the same knob in every mode but the reason it was hit is not, and a
        message that only names ``max_response_length`` sends you to raise a number when
        the answer is usually to change how the context is being kept.
        """
        mode = self._harness_mode()
        fix = {
            "concat": "concat keeps every turn in one conversation, so the episode grows "
                      "without bound; switch trainer.harness to compact, lower the turn "
                      "limit, or raise the budget.",
            "compact": f"compact should have summarised before this; "
                       f"trainer.compact_budget={self.config.trainer.get('compact_budget')} "
                       f"is too close to the budget to leave room for the turn that "
                       f"crosses it, or a single turn exceeds it on its own.",
            "no_concat": "no_concat sends one turn per conversation, so a single "
                         "observation and response are over the budget on their own; "
                         "shrink the observation or raise the budget.",
        }.get(mode, "")
        return f" Running trainer.harness={mode}: {fix}" if fix else ""

    def _harness_mode(self) -> str:
        return self.config.trainer.get("harness", None) or (
            "concat" if self.config.trainer.get("concat_multi_turn", True) else "no_concat"
        )

    def _build_harness(self, per_turn: int, max_turns: int, per_turn_configured: bool = True):
        """The policy, plus a check that the numbers it was given can produce an episode.

        Checked here rather than at startup because two of them -- ``max_turns`` and the
        per-turn budget -- come from the dataset row, so they are not known until an
        episode is about to run. Still before the rollout: every failure it reports is
        decidable from the numbers alone, and the alternative is a crash after the
        generation has been paid for, or a mode that quietly degenerates into a more
        expensive version of another one and reports nothing at all.
        """
        mode = self._harness_mode()
        summary_budget = None
        if mode == "compact":
            m = int(self.config.trainer.compact_budget)
            configured = self.config.trainer.get("compact_summary_budget", None)
            summary_budget = int(configured) if configured else default_summary_budget(m, per_turn)

        check_budgets(mode, Budgets(
            prompt_len=self.prompt_length,
            response_len=self.response_length,
            per_turn=per_turn,
            max_turns=max_turns,
            per_turn_configured=per_turn_configured,
            compact_budget=int(self.config.trainer.compact_budget) if mode == "compact" else None,
            summary_budget=summary_budget,
            summary_request_len=len(self.tokenizer.encode(CompactHarness.SUMMARY_REQUEST)),
        ))

        if mode == "compact":
            return build_harness(mode, budget=int(self.config.trainer.compact_budget),
                                 summary_budget=summary_budget)
        return build_harness(mode)

    def _outputs(self, client, env, result, kwargs, episode_id: str) -> list[AgentLoopOutput]:
        rows = client.rows()
        outputs = []
        # One row is one conversation. Ordered from 0 in the order they were opened --
        # group / episode ids only identify, but conversations and turns are a sequence,
        # so they read as 0,1,2. Enumerating rows as "turn_idx" was the old bug: it
        # numbered conversations and called them turns, which is only the same thing
        # under no_concat.
        for conversation_id, row in enumerate(rows):
            images = client.images(row.conversation_id)
            # Both sides can carry image placeholders: the prompt holds the opening
            # observation, and in concat mode the response region holds every later one,
            # appended between turns as unmasked context. A raw slice of either can land
            # inside a placeholder run while multi_modal_data still ships every image,
            # and the model then dies in the attention on a shape that names neither.
            # verl refuses to slice a multimodal sequence; we defer to the same rule
            # rather than keeping a second, quieter policy here.
            #
            # Text overflows raise here too, which is not verl's default. A dataset
            # prompt that does not fit is trimmed and the sample is still the sample;
            # an episode that does not fit is a different episode after trimming. The
            # window running out is precisely the condition the context policies exist
            # to answer, so hitting it means the policy and the budget disagree -- a
            # thing to fix, not to train through.
            mm = bool(images)
            # Only built when it will be read: the hint asks config what mode is running,
            # and paying for that on every row of every episode to describe a failure
            # that almost never happens is the wrong way round.
            over = (len(row.prompt_ids) > self.prompt_length
                    or len(row.response_ids) > self.response_length)
            hint = self._overflow_hint() if over else ""
            prompt_ids = cap_token_ids(
                row.prompt_ids, self.prompt_length, multimodal=mm, keep="tail",
                what="prompt", budget_name="data.max_prompt_length",
                on_overflow="raise", hint=hint,
            )
            response_ids = cap_token_ids(
                row.response_ids, self.response_length, multimodal=mm, keep="head",
                what="response", budget_name="data.max_response_length",
                on_overflow="raise", hint=hint,
            )
            keep = len(response_ids)
            outputs.append(
                AgentLoopOutput(
                    prompt_ids=prompt_ids,
                    response_ids=response_ids,
                    response_mask=row.response_mask[:keep],
                    # "images", plural. Upstream renamed this key; the singular form is
                    # silently ignored, so the processor is handed no pictures, the
                    # forward pass gets image-pad tokens with no vision features and
                    # text position ids, and Qwen skips the masked_scatter rather than
                    # complaining. The rollout still sees the frames -- only the model
                    # being optimised is blind.
                    multi_modal_data={"images": images} if images else {},
                    response_logprobs=row.logprobs[:keep] or None,
                    # The sum is what verl's own metrics read; the vector below is what
                    # actually trains. Both, because they answer different questions.
                    reward_score=float(sum(row.scores[:keep])),
                    num_turns=1,
                    metrics={},
                    extra_fields={
                        "reward_extra_info": {
                            "traj_success": float(env.success),
                            # Always present, so the metric exists even in runs where the
                            # agent never described anything.
                            **{f"{k}_reward": v for k, v in env.state_scores.items()},
                        },
                        "image_data": images,
                        "last_turn": conversation_id == len(rows) - 1,
                        # Per-token scores, capped alongside the response they index.
                        # Not named token_level_scores: extra_fields become non-tensor
                        # columns, and verl already has a *tensor* of that name, so the
                        # two collide when the batch is converted for the critic.
                        # verl otherwise places one scalar at the final token, which
                        # erases which span earned what -- the whole point of scoring
                        # <observation> and <prediction> where they are written.
                        "per_token_reward": list(row.scores[:keep]),
                        "episode_id": episode_id,
                        "group_idx": kwargs["group_idx"],
                        "traj_idx": kwargs["traj_idx"],
                        "turn_idx": conversation_id,
                        # Which conversation this row belongs to. An episode can span
                        # several: compaction ends one and opens the next, and the
                        # episode log has to show them as one story with a seam, not as
                        # unrelated rows.
                        "conversation_id": conversation_id,
                        #: (start, end) per turn within this conversation, so turn ids
                        #: restart at 0 in each one rather than running on across the
                        #: episode.
                        # Clipped with the response they index. Emitting them whole after
                        # a truncation leaves spans pointing past the end, and the only
                        # thing that noticed was a range check downstream that silently
                        # dropped the turns they described.
                        "response_spans": _spans_within(row.response_spans, keep),
                        # How many turns the episode actually ran. num_turns above is 1
                        # per row by construction, and in concat mode an episode is one
                        # row -- so without this the only turn count anything can see
                        # says every episode was a single turn.
                        "episode_turns": int(result.turns),
                    },
                )
            )
        return outputs
