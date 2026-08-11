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
from dataclasses import replace
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
from vagen.harness.budget import (
    Budgets, check as check_budgets, context_limits, default_env_response,
    default_summary_budget,
)
from vagen.harness.compact import CompactHarness
from vagen.utils.image_token_utils import (
    image_token_ids, placeholder_blocks, truncate_keeping_images_whole,
    vision_sentinel_ids,
)
from vagen.core.client import EpisodeUnusable
from vagen.core.runner import run_episode

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

# Environments that can score the agent's descriptions. Keyed by the registry name.
STATE_REWARD_SPECS = {"Sokoban": sokoban_spec.SPEC}


class SampledVisionToken(EpisodeUnusable, ValueError):
    """The policy emitted an image placeholder of its own accord. See
    ``GymLoop._refuse_sampled_vision_tokens``."""


def _spans_within(spans, keep: int) -> tuple[list[tuple[int, int]], int]:
    """The spans that survive a truncation to ``keep``, and where the survivors end.

    A span that straddles the cut is dropped entirely rather than clipped, and the caller
    unmasks what is left of it. Clipping looked like the gentler option and is the worse
    one: the surviving fragment keeps mask 1, so it is trained on as if it were an action,
    while the reward -- which ``add_reward`` writes at ``scores[end - 1]`` -- is past the
    cut and dropped. The model is optimised on half a move, at reward zero.

    It is always the *last* turn, and the last turn is where the terminal reward lands.
    The response region ends on a model span (the next observation is only appended if it
    fits), so every overflow straddles one. Measured on a solve-at-turn-5 episode with a
    four-token overflow: 10.4 earned, 0.4 trained, the whole success reward gone. Under a
    group-relative estimator the rollouts that *solved* then get the group's most negative
    advantage, because the baseline moved with them.
    """
    out, safe_end = [], 0
    for start, end in spans or ():
        start, end = int(start), int(end)
        if start >= keep:
            break
        if end > keep:
            break            # straddles the cut: the whole turn goes
        out.append((start, end))
        safe_end = end
    return out, safe_end


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
        # ★ Only the scores that are actually switched on. Anything in here is published
        # as a `<name>_reward` extra field, and verl turns every extra field into a
        # val_aux curve -- so declaring all of them unconditionally drew a flat zero line
        # for `state_estimation_reward`, `transition_prediction_reward` and
        # `format_reward` in every run that had state_reward off, which reads as "the
        # reward is on and the agent is scoring nothing" rather than "the reward is off".
        # `format` belongs to this set too: it is the state-reward gate, not a separate
        # signal, so with state_reward off there is no format reward to report.
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
        return self._message(obs), reward, bool(done), False, info

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
                f"environment {self.name!r} returned {unsupported} in multi_modal_input; "
                f"only '<image>' is carried into training. Supporting another modality "
                f"means lifting it here and adding it to multi_modal_data in _outputs."
            )
        images = _normalize_images(mm.get("<image>", []) or [])
        return {"role": role, "content": convert_obs_to_content(obs, **self.kwargs), "images": images}


def resolve_reward_placement(config, configured: str = "auto") -> str:
    """Where a turn's scores are paid, resolved from the advantage estimator by default.

    ★ Placement and estimator are one choice. An estimator whose outer chain has a single
    reward slot per turn reads a turn's reward only at the turn's last token, so a score
    paid mid-turn is credited twice (measured bias 0.177); the per-token estimators prefer
    the opposite, because a lumped score has to be remembered by ``V`` for the rest of the
    turn (-28% variance at lam 0.9 from per-span). Neither mistake raises and neither is
    visible in a curve, so the estimator decides and ``placement`` exists to be overridden
    rather than to be set.

    ``auto`` is the default for the same reason ``lam_low`` has none on ``bi_level_gae``:
    a value that must be kept in step with another setting by hand is one that eventually
    disagrees with it, silently.
    """
    from vagen.custom_advantage import wants_turn_lumped_reward

    if configured != "auto":
        return configured
    algorithm = config.get("algorithm", {}) or {}
    estimator = algorithm.get("adv_estimator", "") if hasattr(algorithm, "get") else ""
    return "turn_end" if wants_turn_lumped_reward(estimator) else "per_span"


# Registered under both names: the dataset emits "gym_agent"
# (gym_agent_dataset.py) and that is what actually dispatches, via
# configs/agent_v2.yaml. "gym_agent_v2" is the decorator's own name and
# nothing selects it -- kept so a config that does still resolves.
@register("gym_agent")
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
        scored_env = self._maybe_state_reward(
            env_cls(env_config=kwargs["config"]), kwargs["env_name"], max_turns
        )
        env = GymEnvAdapter(
            scored_env, kwargs["env_name"], kwargs,
            score_names=self._enabled_state_rewards(),
        )

        per_turn = min(int(kwargs.get("response_length_per_turn") or self.response_length), self.response_length)
        harness, budgets = self._build_harness(
            per_turn, max_turns,
            env_response=kwargs.get("env_response_length"),
            per_turn_configured=bool(kwargs.get("response_length_per_turn")),
        )
        opening_limit, continuation_limit = context_limits(self._harness_mode(), budgets)
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
        # What one call may hand the model that it did not generate. Enforced here rather
        # than left to the end of the episode, where an observation that did not fit shows
        # up as a truncated row and not as an oversized observation.
        client.opening_limit, client.continuation_limit = opening_limit, continuation_limit

        try:
            result = await run_episode(env, harness, client, seed=kwargs["seed"],
                                       max_turns=max_turns)
        except EpisodeUnusable as exc:
            # This rollout cannot be finished, and that is evidence about this rollout
            # rather than about the run. Letting it out takes the whole batch with it:
            # verl's asyncio.gather has no return_exceptions, so one unlucky environment
            # sample costs an entire training step.
            #
            # A configuration error is different -- it is evidence about every episode --
            # and BudgetError, ImagePlaceholderMismatch and the prompt overflow are
            # deliberately not EpisodeUnusable, so they still stop the run.
            logger.warning("[vagen] dropping episode %s: %s: %s",
                           episode_id, type(exc).__name__, exc)
            return []
        return self._outputs(client, env, result, kwargs, episode_id, harness)

    def _enabled_state_rewards(self) -> tuple[str, ...]:
        """The score names this run will actually compute, in publication order.

        ``format`` rides along only when it can be non-zero. It is the gate on the others
        rather than a signal of its own, and the shipped setting is
        ``state_reward.format_reward: 0.0`` -- so publishing it unconditionally drew a
        flat zero line called ``format_reward`` in every state-reward run. That reads as
        "the agent never once produced the right format", which is both alarming and
        false; the format reward that is actually paid is the *environment's*
        (``SokobanEnvConfig.format_reward``), a different knob that this curve never
        showed. ``format_correct_rate`` below is the honest version of the question.
        """
        cfg = self.config.trainer.get("state_reward", {}) or {}
        names = tuple(n for n in TAGS if (cfg.get(n) or {}).get("enable", False))
        if not names:
            return ()
        return (*names, "format") if float(cfg.get("format_reward", 0.0) or 0.0) > 0 else names

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
            placement=resolve_reward_placement(self.config, str(cfg.get("placement", "auto"))),
        )
    def _summary_request_len(self) -> int:
        """What the summary request costs as the client will actually send it.

        The bare string is not that: the client renders it as a chat turn, which adds the
        role header and the end marker -- 15 tokens against 23 on Qwen2.5-VL. The bound
        it appears in is stated exactly, so measuring the wrong thing by 8 tokens makes it
        exact and wrong. Rendered here the same way, and the wrapper the harness puts
        around the summary itself is charged too, since that also arrives as context.
        """
        turn = [{"role": "user", "content": CompactHarness.SUMMARY_REQUEST}]
        try:
            rendered = self.tokenizer.apply_chat_template(
                turn, add_generation_prompt=True, tokenize=True, return_dict=False,
                **self.apply_chat_template_kwargs,
            )
        except Exception:
            # A tokenizer with no chat template still needs a number; the bare string
            # under-counts, which is the safe direction for a ceiling and the unsafe one
            # for a bound, so say so rather than pretending the measurement happened.
            logger.warning("no chat template to measure the summary request with; "
                           "the compact peak bound is approximate")
            rendered = self.tokenizer.encode(CompactHarness.SUMMARY_REQUEST)
        return len(rendered) + len(self.tokenizer.encode(CompactHarness.SUMMARY_PREFIX + "\n\n"))

    def _placeholders(self):
        """The ids a picture sits behind, and the sentinels that bracket it.

        Cached: reading them is cheap but this runs per row, and the answer is a property
        of the model.
        """
        if getattr(self, "_ph_cache", None) is None:
            source = getattr(self, "processor", None) or getattr(self, "tokenizer", None)
            self._ph_cache = ((image_token_ids(source), vision_sentinel_ids(source))
                              if source is not None else (set(), set()))
        return self._ph_cache

    def _refuse_sampled_vision_tokens(self, row) -> None:
        """A picture the policy invented is not a picture.

        Nothing bans the vision vocabulary from a generation -- they are ordinary ids --
        so a model can sample ``<|vision_start|><|image_pad|>`` and there is no frame
        behind it. The placeholder count then exceeds the frame count and the row dies in
        ``get_rope_index`` with a bare ``IndexError: index 2 is out of bounds``, several
        layers from anything that names a cause.

        Refused as an unusable episode rather than a fatal one: it is the policy's output,
        not the configuration, and a model that does this occasionally should cost one
        rollout rather than the run. It is worth watching, though -- a model that has
        learned to emit image tokens is learning something the reward did not ask for.
        """
        placeholders, sentinels = self._placeholders()
        watched = placeholders | sentinels
        if not watched:
            return
        for start, end in row.response_spans or ():
            hit = {t for t in row.response_ids[int(start):int(end)] if int(t) in watched}
            if hit:
                raise SampledVisionToken(
                    f"the policy generated vision token(s) {sorted(hit)} inside its own "
                    f"response at [{start}, {end}). There is no frame behind them, so the "
                    f"placeholder runs and multi_modal_inputs stop being 1:1 and the row "
                    f"fails inside get_rope_index. Ban these ids at sampling "
                    f"(rollout.sampling_params) if the model keeps doing it."
                )

    def _split_frames(self, prompt_ids, images):
        """Which frames belong to the prompt region and which to the response region.

        ``client.images()`` hands back every frame in the conversation, but the two
        regions are cut separately, so each needs its own list. The boundary is a
        context/response seam -- ``prompt_len`` is set at the first response -- so it can
        never fall inside a picture.
        """
        images = list(images or [])
        if not images:
            return [], []
        placeholders, sentinels = self._placeholders()
        n = len(placeholder_blocks(prompt_ids, placeholders, sentinels))
        return images[:n], images[n:]

    def _truncate_response(self, response_ids, frames, hint):
        """Cut the response region to fit, never through a picture."""
        if len(response_ids) <= self.response_length:
            return list(response_ids), list(frames)
        placeholders, sentinels = self._placeholders()
        if frames and not placeholders:
            # No declared placeholder ids means no blocks are found, so every frame is
            # dropped and the row goes out with image-pad tokens and no pictures -- the
            # model attends to something it was never given, and nothing raises. The
            # guard below required `placeholders` to be non-empty, which is precisely
            # the case it needed to catch.
            raise ValueError(
                f"the response carries {len(frames)} image(s) but this model declares no "
                f"image placeholder ids, so the sequence cannot be cut without losing "
                f"them. Register the family in IMAGE_TOKEN_ADAPTERS.{hint}"
            )
        if frames and not sentinels and placeholders:
            # Without the sentinels a cut can orphan a run from its vision_start, which
            # rope then lays out as text -- silently, since every count still agrees.
            # Refuse rather than degrade into that.
            raise ValueError(
                f"the response is {len(response_ids)} tokens against "
                f"data.max_response_length={self.response_length} and carries images, but "
                f"this model declares no vision sentinels, so it cannot be cut safely.{hint}"
            )
        logger.warning("response of %d tokens exceeds data.max_response_length=%d; "
                       "truncating.%s", len(response_ids), self.response_length, hint)
        return truncate_keeping_images_whole(
            response_ids, self.response_length, keep="head",
            placeholders=placeholders, frames=frames, sentinels=sentinels, min_kept=1)

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

    def _build_harness(self, per_turn: int, max_turns: int, env_response=None,
                       per_turn_configured: bool = True):
        """The policy, plus a check that the numbers it was given can produce an episode.

        Checked here rather than at startup because two of them -- ``max_turns`` and the
        per-turn budget -- come from the dataset row, so they are not known until an
        episode is about to run. Still before the rollout: every failure it reports is
        decidable from the numbers alone, and the alternative is a crash after the
        generation has been paid for, or a mode that quietly degenerates into a more
        expensive version of another one and reports nothing at all.
        """
        mode = self._harness_mode()
        m = int(self.config.trainer.compact_budget) if mode == "compact" else None
        summary_budget = None
        if mode == "compact":
            configured = self.config.trainer.get("compact_summary_budget", None)
            summary_budget = int(configured) if configured else default_summary_budget(m, per_turn)

        b = Budgets(
            prompt_len=self.prompt_length,
            response_len=self.response_length,
            per_turn=per_turn,
            max_turns=max_turns,
            context=self.config.actor_rollout_ref.rollout.get("max_model_len", None),
            per_turn_configured=per_turn_configured,
            compact_budget=m,
            summary_budget=summary_budget,
            summary_request_len=self._summary_request_len(),
        )
        # Derived from what the mode has left rather than defaulted to a constant, so an
        # env config that does not declare it is still bounded -- by the largest value
        # that would have passed the checks below.
        b = replace(b, env_response=int(env_response) if env_response else default_env_response(mode, b),
                    env_response_configured=bool(env_response))
        check_budgets(mode, b)

        # Every mode gets the region and the floor. Passing them to compaction alone
        # left `_left()` as None for the other two, so nothing bounded their generation
        # by the room left and nothing stopped them when it ran out -- concat then filled
        # past its region and the batch-boundary cut took model turns with it, losing the
        # reward on them. Measured: 62 of 182 admitted concat configs lost reward.
        # The floor is the smallest generation worth making, and it must never be a large
        # fraction of the region. `per_turn` falls back to the whole response length when
        # the env config declares no per-turn budget -- a supported case -- and using that
        # as the floor makes `exhausted()` true after a single token: every concat episode
        # then stops at turn one, marked truncated, with a perfectly well-formed row and
        # nothing reporting it.
        #
        # Erring small is the safe direction. Too small only allows a squeezed generation,
        # which the truncation handles; too large silently deletes the episode.
        room = dict(response_len=self.response_length,
                    floor=min(per_turn, max(1, self.response_length // 4)))
        if mode == "compact":
            # compact_budget is an optional second trigger on top of the region.
            return build_harness(
                mode, budget=m, summary_budget=summary_budget,
                summary_request_len=b.summary_request_len, **room,
            ), b
        return build_harness(mode, **room), b

    def _outputs(self, client, env, result, kwargs, episode_id: str,
                 harness) -> list[AgentLoopOutput]:
        rows = client.rows()
        outputs = []
        # One row is one conversation. Ordered from 0 in the order they were opened --
        # group / episode ids only identify, but conversations and turns are a sequence,
        # so they read as 0,1,2. Enumerating rows as "turn_idx" was the old bug: it
        # numbered conversations and called them turns, which is only the same thing
        # under no_concat.
        for row in rows:
            conversation_id = row.ordinal
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

            # The prompt still refuses to be cut, and that is not asymmetry for its own
            # sake: this region is the *opening call* -- system prompt plus the first
            # observation, plus a summary under compaction. There is nothing old in it to
            # drop, so a left cut takes the instructions. The client's opening ceiling
            # already bounds it, so reaching here means something upstream is wrong.
            prompt_ids = cap_token_ids(
                row.prompt_ids, self.prompt_length, multimodal=mm, keep="tail",
                what="prompt", budget_name="data.max_prompt_length",
                on_overflow="raise", hint=hint,
            )
            # The response is truncated rather than refused. Budget-aware generation is
            # what should keep it inside the region; this is the backstop for when the
            # environment returns more than anyone planned for, and refusing there makes
            # a long-tail rollout impossible to debug. What gets cut is context: the
            # model's own tokens are bounded by max_new_tokens, so only observations can
            # overflow, and observations are mask 0 and carry no reward.
            self._refuse_sampled_vision_tokens(row)
            prompt_frames, response_frames = self._split_frames(row.prompt_ids, images)
            response_ids, response_frames = self._truncate_response(
                row.response_ids, response_frames, hint)
            images = prompt_frames + response_frames
            keep = len(response_ids)
            spans, safe_end = _spans_within(row.response_spans, keep)
            # Whatever is left of a dropped turn is context, not an action. Left at
            # mask 1 it trains as a decision the model never finished making.
            mask = list(row.response_mask[:keep])
            scores = list(row.scores[:keep])
            for i in range(safe_end, len(mask)):
                mask[i] = 0
                # ★ The scores go with the mask, not just the mask. A *scalar* env reward
                # sits at the turn's last token and is clipped away with the span, so that
                # half looked fixed. A *vector* reward (state_reward) is spread over tokens
                # near the turn's start, so it survives the `[:keep]` slice while the mask
                # above it is zeroed: the estimators gather only mask-1 positions and drop
                # it, but `token_level_scores` still carries it -- into critic/score/mean,
                # into the custom metrics, and into the STARPO-S filter's per-sample reward,
                # which decides which groups survive. Reported reward then exceeds trained
                # reward with nothing reporting the gap.
                scores[i] = 0.0
            outputs.append(
                AgentLoopOutput(
                    prompt_ids=prompt_ids,
                    response_ids=response_ids,
                    response_mask=mask,
                    # "images", plural. Upstream renamed this key; the singular form is
                    # silently ignored, so the processor is handed no pictures, the
                    # forward pass gets image-pad tokens with no vision features and
                    # text position ids, and Qwen skips the masked_scatter rather than
                    # complaining. The rollout still sees the frames -- only the model
                    # being optimised is blind.
                    multi_modal_data={"images": images} if images else {},
                    # None when the engine did not return logprobs. The tape fills
                    # unsupplied positions with 0.0, and `[0.0, 0.0, ...] or None` is the
                    # list -- so verl received a real rollout_log_probs tensor of zeros.
                    # Every rollout-vs-training probability metric then reads that as the
                    # rollout's actual belief, and `apply_bypass_mode` guards only on the
                    # key being present, so a zero vector sets old_log_probs to zero.
                    response_logprobs=(row.logprobs[:keep]
                                       if any(row.logprobs[:keep]) else None),
                    # The sum is what verl's own metrics read; the vector below is what
                    # actually trains. Both, because they answer different questions.
                    reward_score=float(sum(scores)),
                    num_turns=1,
                    metrics={},
                    extra_fields={
                        "reward_extra_info": {
                            "traj_success": float(env.success),
                            # Always present, so the metric exists even in runs where the
                            # agent never described anything.
                            **{f"{k}_reward": v for k, v in env.state_scores.items()},
                            # Fraction of the episode's turns the environment judged
                            # well-formed. Absent when the environment does not report it,
                            # rather than a zero that would read as "never well-formed".
                            # getattr, not attribute access: `env` here is whatever the
                            # runner was handed, and the tests' fakes are not the adapter.
                            **(
                                {"format_correct_rate": getattr(env, "turns_well_formed", 0)
                                                        / getattr(env, "turns_seen", 0)}
                                if getattr(env, "reports_format", False) and getattr(env, "turns_seen", 0)
                                else {}
                            ),
                        },
                        "image_data": images,
                        "last_turn": row is rows[-1],
                        # `row is rows[-1]`, not `conversation_id == len(rows) - 1`:
                        # conversation_id is the ordinal assigned when the conversation
                        # was opened, and a conversation the model never spoke in is
                        # dropped without returning its number. Comparing an ordinal to
                        # a count then flags the wrong row -- ordinals [0,2,3] with
                        # len(rows)==3 marks the middle one and misses the last.
                        # Per-token scores, capped alongside the response they index.
                        # Not named token_level_scores: extra_fields become non-tensor
                        # columns, and verl already has a *tensor* of that name, so the
                        # two collide when the batch is converted for the critic.
                        # verl otherwise places one scalar at the final token, which
                        # erases which span earned what -- the whole point of scoring
                        # <observation> and <prediction> where they are written.
                        "per_token_reward": scores,
                        # True when this conversation ended because the context filled
                        # up and the model was asked to summarise -- not because the
                        # environment stepped. The next conversation's first action sees
                        # the same world state this row's summary saw, so an estimator
                        # that discounts turn-to-turn must not charge this seam as a
                        # transition. See `BaseHarness.summarised_conversations`.
                        "ends_with_summary": row.conversation_id in harness.summarised_conversations,
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
                        "response_spans": spans,
                        # How many turns the episode actually ran. num_turns above is 1
                        # per row by construction, and in concat mode an episode is one
                        # row -- so without this the only turn count anything can see
                        # says every episode was a single turn.
                        "episode_turns": int(result.turns),
                    },
                )
            )
        return outputs
