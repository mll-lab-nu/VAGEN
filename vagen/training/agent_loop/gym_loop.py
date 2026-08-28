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
from dataclasses import replace
from uuid import uuid4

from verl.experimental.agent_loop.agent_loop import AgentLoopOutput, cap_token_ids, register
from verl.utils.rollout_trace import rollout_trace_op

from vagen.training.agent_loop.base import VagenGymAgentLoopBase
from vagen.envs import build_env, state_reward_names
from vagen.training.agent_loop.verl_client import VerlClient
from vagen.harness import budget_mode, build_harness, resolve_harness
from vagen.harness.compact import CompactHarness
from vagen.harness._common.budget import (
    Budgets, check as check_budgets, context_limits, default_env_response,
    default_summary_budget,
)
from vagen.models import (
    image_token_ids, placeholder_blocks, truncate_keeping_images_whole,
    vision_sentinel_ids,
)
from vagen.rollout import EpisodeUnusable, run_episode
from vagen.envs._common.adapter import GymEnvAdapter, _accepts_response

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

# Which environments can have their reasoning scored is no longer a table here: each
# environment declares its own `STATE_REWARD_SPEC`. See `envs/_common/rewards/factory.py`.


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
        # every row is well-formed, just short. The same value constructs TurnLimit and
        # remains run_episode's backstop, so the environment and runner cannot drift.
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
        # Whether descriptions are scored, and by which judge, is read from the
        # environment's own config block -- the same one evaluation builds from. The
        # trainer holds no state-reward settings and this loop passes none.
        env = GymEnvAdapter(
            build_env(env_cls, kwargs["config"], max_turns=max_turns),
            kwargs["env_name"], kwargs,
            score_names=state_reward_names(kwargs["config"]),
        )

        per_turn = min(int(kwargs.get("response_length_per_turn") or self.response_length), self.response_length)
        harness, budgets = self._build_harness(
            per_turn, max_turns,
            env_response=kwargs.get("max_env_response_per_turn") or kwargs.get("env_response_length"),
            per_turn_configured=bool(kwargs.get("response_length_per_turn")),
        )
        opening_limit, continuation_limit = context_limits(budget_mode(self._harness_mode()), budgets)
        # An extra sampling key rather than anything this layer interprets. verl builds its
        # sampling dict from a fixed list of fields and has no pass-through for the rest,
        # but the engine call is `SamplingParams(max_tokens=..., **sampling_params)` -- so
        # a key added here reaches vLLM untouched, and nothing in VAGEN has to know what a
        # reasoning block is. See EnvSpec.thinking_token_budget for why it is not the same
        # lever as response_length_per_turn.
        if kwargs.get("thinking_token_budget"):
            sampling_params = {**sampling_params,
                               "thinking_token_budget": int(kwargs["thinking_token_budget"])}
        if kwargs.get("stop_strings"):
            sampling_params = {
                **sampling_params,
                "stop": list(kwargs["stop_strings"]),
                "include_stop_str_in_output": True,
            }
        client = VerlClient(
            self.server_manager,
            self.tokenizer,
            self.processor,
            model_adapter_name=self.config.trainer.get("model_adapter", "auto"),
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
            # ★ `_outputs` is inside the guard, not after it. It calls
            # `_refuse_sampled_vision_tokens`, which raises `SampledVisionToken` -- an
            # `EpisodeUnusable` whose whole point is to cost one rollout rather than the
            # run. Built outside, it escaped instead: `sokoban_turn_nosr_fmt` died at step
            # 99 of 401 after 2h33m because one rollout sampled an image token. The
            # exception class, the docstring and the handler all said "drop the episode";
            # only the placement of one line said otherwise.
            return self._outputs(client, env, result, kwargs, episode_id, harness)
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
        """Which context policy this run uses. One key: the old ``concat_multi_turn``
        boolean is deleted rather than deprecated, so a stale override is rejected by
        hydra rather than quietly outvoted by ``harness``."""
        return self.config.trainer.get("harness", None) or "concat"

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
        # ★ compact_budget is optional, and the budget checker says so in the text it
        # prints ("leave it unset and let the response region decide"). `int(None)` is a
        # TypeError, which is not EpisodeUnusable, so it escaped GymLoop.run into
        # asyncio.gather and took the whole batch -- on every episode of every step,
        # naming nothing. The unset case is the one the code recommends.
        m = None
        summary_budget = None
        if issubclass(resolve_harness(mode), CompactHarness):
            configured_m = self.config.trainer.get("compact_budget", None)
            m = int(configured_m) if configured_m else None
            configured = self.config.trainer.get("compact_summary_budget", None)
            if configured:
                summary_budget = int(configured)
            elif m:
                summary_budget = default_summary_budget(m, per_turn)
            else:
                # No trigger of its own: the response region is the bound, so reserve
                # against that rather than against a budget that does not exist.
                summary_budget = default_summary_budget(self.response_length, per_turn)

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
        b = replace(b, env_response=int(env_response) if env_response
                    else default_env_response(budget_mode(mode), b))
        check_budgets(budget_mode(mode), b)

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
        if issubclass(resolve_harness(mode), CompactHarness):
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
                    # Numeric zero cannot distinguish "the backend omitted logprobs"
                    # from a legitimate probability-1 token.  The tape carries that
                    # provenance explicitly; ``any(logprobs)`` incorrectly dropped
                    # valid all-zero GLM responses and broke mixed batches.
                    response_logprobs=(row.logprobs[:keep]
                                       if getattr(row, "logprobs_complete", any(row.logprobs))
                                       else None),
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
                            # well-formed. NaN, not absent, when the environment does not
                            # report one -- see the note on a stable key set below.
                            # getattr, not attribute access: `env` here is whatever the
                            # runner was handed, and the tests' fakes are not the adapter.
                            "format_correct_rate": (
                                getattr(env, "turns_well_formed", 0) / getattr(env, "turns_seen", 0)
                                if getattr(env, "reports_format", False) and getattr(env, "turns_seen", 0)
                                else float("nan")
                            ),
                            # ★ How much the model itself wrote, per turn. `response_spans`
                            # is exactly the model-emitted region -- the same thing the
                            # response mask marks -- so this counts generated tokens and
                            # nothing else.
                            #
                            # `response_length/mean` cannot answer this: it measures the
                            # whole response *region*, which also carries the observations
                            # interleaved between turns. On vision Sokoban an observation
                            # is 49-144 image tokens, so that metric moves when the
                            # environment renders differently and dilutes the thing we
                            # actually watch for -- whether the policy is getting more
                            # verbose. Every collapse in the 0809 sweep showed up first as
                            # a length jump, and this is the number that would have said
                            # whether the jump was the model or the scenery.
                            "model_tokens_per_turn": (
                                sum(int(e) - int(b) for b, e in spans) / max(1, len(spans))
                                if spans else float("nan")
                            ),
                        },
                        # ★ Every row publishes the same reward_extra_info keys, always.
                        # verl reads the key set off ROW 0 and then indexes every other row
                        # with it (agent_loop.py: `reward_extra_keys =
                        # list(reward_extra_infos[0].keys())`), so a row that omits a key
                        # either loses the metric for the whole batch or raises KeyError and
                        # takes the training step -- depending on which episode happens to
                        # land at index 0, i.e. nondeterministically.
                        #
                        # Two reachable producers of a mixed key set: an environment that
                        # crashed (GymEnvAdapter.step returns {"env_error": True} with no
                        # format_correct), and spatial_gym, which never reports
                        # format_correct at all while the other four environments do. Any
                        # batch mixing them hit this every step.
                        #
                        # NaN rather than 0: "this environment does not measure it" is not
                        # "it was never well-formed", and a mean over NaN is visibly NaN
                        # rather than quietly depressed.
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
