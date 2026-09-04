"""Rewarding the reasoning, not only the outcome.

Two rewards, independently switchable:

* **state estimation** -- the agent says where things are before acting
  (``<perception>``), scored against the state it acted *from*.
* **transition prediction** -- it says where they will be after
  (``<prediction>``), scored against the state it acted *into*.

Descriptions are structured either by the configured model judge or, when an
environment exposes a closed-vocabulary parser, by the optional exact scorer. Both paths
feed the same F1 and reward-composition logic.

★ Each score is paid **per span** -- on the last token of the section that earned it. The
environment decides where a reward belongs and remains independent of the downstream
advantage estimator.

This used to be a ``placement`` setting resolved from ``algorithm.adv_estimator``, which
put the estimator's business inside the environment and made the two a single choice
spread across two configs.

Enabling either reward requires the complete canonical WM response. The switches decide
which descriptions the judge scores, while the protocol remains stable across runs. A
partial or legacy-shaped response earns no auxiliary reward -- see ``_score``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from vagen.envs._common.rewards.judge import NullJudge
from vagen.envs._common.rewards.spans import tagged_span, token_offsets, tokens_covering
from vagen.envs._common.rewards.spatial import grouped_f1
from vagen.envs._common.response_format import WM_FORMAT, parse_wm_sections

#: reward name -> the tag the agent writes it in
TAGS = {"state_estimation": "perception", "transition_prediction": "prediction"}

#: What a description that looked at nothing already scores. Subtracted before paying.
#:
#: ★ `grouped_f1` pays 0.5 for getting one of two axes right and each axis is a 3-way
#: choice, so merely *naming* a relation scores about a third. Re-measured over the
#: current 10,000-example Sokoban training seed range, the best constant description --
#: which describes nothing and looks at nothing -- scores 0.427825. The reward's usable
#: range is therefore much narrower than 0 to 1, and observed model scores around 0.45
#: can sit barely above a constant policy.
#:
#: 0.334 is the uniform-random floor rather than the 0.428 constant-answer floor, so a
#: policy that finds the most common relation and repeats it still earns a little. Raise
#: it to 0.428 to make the calibration stricter. The measurement is reproducible with
#: ``tools/calibrate_sokoban_state_reward.py``.
DEFAULT_SCORE_BASE = 0.334
CREDIT_SITES = ("section_end", "turn_end")
AGGREGATIONS = ("per_turn", "episode_mean")
SCORERS = ("judge", "exact")


class _EpisodeMeanScalar(float):
    """A scalar whose auxiliary component is finalized once episode length is known."""

    def __new__(cls, outcome: float, auxiliary: float, horizon: int):
        obj = float.__new__(cls, float(outcome) + float(auxiliary))
        obj.outcome = float(outcome)
        obj.auxiliary = float(auxiliary)
        obj.horizon = int(horizon)
        return obj

    def finalize_episode(self, turns: int) -> float:
        return self.outcome + self.auxiliary * self.horizon / max(1, int(turns))


class _EpisodeMeanVector(list):
    """A token reward retaining which entries came from state supervision."""

    def __init__(self, values, auxiliary, horizon: int):
        super().__init__(values)
        self.auxiliary = list(auxiliary)
        self.horizon = int(horizon)

    def finalize_episode(self, turns: int) -> list[float]:
        scale = self.horizon / max(1, int(turns))
        return [
            float(total) + (scale - 1.0) * float(aux)
            for total, aux in zip(self, self.auxiliary)
        ]


@dataclass
class StateRewardSpec:
    """What an environment must supply to have its agent's reasoning scored."""

    #: env -> the relations actually present, as {"object_id", "vertical_relation", ...}
    relations: Callable[[Any], list[dict]]
    #: format(content=...) -> the prompt that asks the judge to structure a description
    judge_prompt: str
    #: per object type, how much of the score it accounts for
    object_weights: dict[str, float]
    #: one worked example per section, so the agent is shown the format it is scored on
    examples: dict[str, str] = field(default_factory=dict)
    #: how to read the relations, appended once whichever sections are on
    axes: str = ""
    #: Optional environment-owned deterministic parser. It maps one description span to
    #: the same relation records returned by the judge. Environments should provide this
    #: only when their response contract has a closed vocabulary.
    exact_parser: Optional[Callable[[str], list[dict]]] = None


@dataclass
class StateRewardWrapper:
    """An environment whose reward includes how well the agent described the world."""

    env: Any
    spec: StateRewardSpec
    judge: Any = field(default_factory=NullJudge)
    #: reward name -> configured scale for a perfect description of that section.
    #: Under the default ``per_turn`` aggregation this is what one turn can earn. Under
    #: ``episode_mean`` the episode receives the average turn quality times
    #: this scale and the turn horizon, preserving the same maximum episode budget. The
    #: value itself always comes directly from the environment config rather than from
    #: the downstream advantage estimator.
    enabled: dict[str, float] = field(default_factory=dict)
    #: Subtracted from each f1 before it is paid, then the positive remainder is
    #: rescaled so a perfect description still earns the full configured reward. 0.0
    #: restores the legacy reward exactly. See `DEFAULT_SCORE_BASE` for its origin.
    score_base: float = DEFAULT_SCORE_BASE
    #: Where auxiliary credit is written. ``section_end`` is the production default;
    #: ``turn_end`` places the combined auxiliary signal at the action boundary for
    #: estimators that reason over environment turns. It is configured by the
    #: environment, never inferred from the downstream estimator.
    credit_site: str = "section_end"
    #: ``per_turn`` preserves per-turn behavior. ``episode_mean`` divides only the
    #: auxiliary part by the realized episode length and multiplies by the configured
    #: turn horizon after the final turn. This preserves the maximum shaping budget of
    #: ``per_turn`` while preventing a failed five-turn trajectory from receiving more
    #: merely because it survived longer. Credit remains on the section token that
    #: earned it.
    aggregation: str = "per_turn"
    #: ``judge`` preserves free-form parsing through the configured model. ``exact``
    #: uses the environment-owned closed-vocabulary parser.
    scorer: str = "judge"
    #: Supplied by the shared environment factory from the episode's real turn limit.
    #: It is not an independent user knob: changing it without changing the episode
    #: horizon would silently change the shaping budget.
    episode_horizon: int = 1

    def __post_init__(self):
        unknown = set(self.enabled) - set(TAGS)
        if unknown:
            raise ValueError(
                f"unknown state rewards {sorted(unknown)}; choose from {sorted(TAGS)}"
            )
        if self.credit_site not in CREDIT_SITES:
            raise ValueError(
                f"unknown state-reward credit_site {self.credit_site!r}; "
                f"choose from {CREDIT_SITES}"
            )
        if self.aggregation not in AGGREGATIONS:
            raise ValueError(
                f"unknown state-reward aggregation {self.aggregation!r}; "
                f"choose from {AGGREGATIONS}"
            )
        if self.scorer not in SCORERS:
            raise ValueError(
                f"unknown state-reward scorer {self.scorer!r}; choose from {SCORERS}"
            )
        if self.scorer == "exact" and self.spec.exact_parser is None:
            raise ValueError(
                "state-reward scorer='exact' requires the environment spec to declare "
                "an exact_parser"
            )
        if int(self.episode_horizon) <= 0:
            raise ValueError("state-reward episode_horizon must be positive")
        self.last_scores: dict[str, float] = {}

    # ------------------------------------------------------------------ env facade
    async def reset(self, seed=None):
        return await self.env.reset(seed=seed)

    async def close(self):
        await self.env.close()

    async def system_prompt(self):
        prompt = await self.env.system_prompt()
        text = prompt.get("obs_str", "") if isinstance(prompt, dict) else str(prompt)
        block = self.instructions(text)
        if not block:
            return prompt
        joined = f"{text}\n\n{block}"
        return {**prompt, "obs_str": joined} if isinstance(prompt, dict) else joined

    def instructions(self, existing: str = "") -> str:
        """Request canonical WM once when the environment has not already done so."""
        if not self.enabled:
            return ""
        required = ("perception", "reasoning", "prediction", "answer")
        if all(f"<{tag}>" in existing for tag in required):
            return ""
        sections = [
            self.spec.examples[name]
            for name in self.enabled
            if name in self.spec.examples
        ]
        body = "\n".join(sections)
        return (
            "Use the complete response format below; the order is required:\n"
            f"{WM_FORMAT}\n{body}\n{self.spec.axes}"
        ).strip()

    # ------------------------------------------------------------------- stepping
    async def step(self, action: str, response_token_ids=None, tokenizer=None):
        """Score the descriptions, then return what the env it wraps returns.

        Four values, not the five of ``BaseEnv``. This wrapper stands in for a plain gym
        environment and is consumed by ``GymEnvAdapter``, which is the thing that speaks
        the five-value contract, with this underneath it. Returning five made every step
        raise "too many values to unpack (expected 4)" -- 1006 turns of it, reaching the
        cluster as a model that earned zero reward rather than as a wrapper with the
        wrong arity.
        """
        before = self.spec.relations(self.env)
        obs, reward, done, info = await self.env.step(action)
        after = self.spec.relations(self.env)

        scored = await self._score(
            action,
            {"state_estimation": before, "transition_prediction": after},
            protocol_correct=info.get("format_correct"),
        )
        self.last_scores = {k: v for k, v in scored.items() if k != "spans"}
        info = {
            **(info or {}),
            **{f"state_reward/{k}": v for k, v in self.last_scores.items()},
        }

        if response_token_ids is None or tokenizer is None:
            # Nothing to place anything on; degrade to a scalar rather than vanish.
            auxiliary = sum(self.last_scores.values())
            value = (
                _EpisodeMeanScalar(float(reward), auxiliary, self.episode_horizon)
                if self.aggregation == "episode_mean"
                else float(reward) + auxiliary
            )
            return obs, value, done, info

        return (
            obs,
            self._place(scored, float(reward), response_token_ids, tokenizer),
            done,
            info,
        )

    async def _score(
        self,
        action: str,
        gold: dict[str, list[dict]],
        *,
        protocol_correct: Optional[bool] = None,
    ) -> dict:
        spans = {
            name: tagged_span(action, tag)
            for name, tag in TAGS.items()
            if name in self.enabled
        }
        present = [(name, span) for name, span in spans.items() if span is not None]

        scores: dict[str, Any] = {"spans": spans}
        for name in self.enabled:
            scores[name] = 0.0

        # Auxiliary supervision follows the same canonical ordering as the environment
        # format reward. Old tags or reordered fields may still be useful for diagnostics,
        # but they must not earn reward-model credit.
        if protocol_correct is False or not parse_wm_sections(
            action, allow_native_thinking=True
        ).format_correct:
            return scores

        # Format is a gate, not a line item: a turn that did not write every section it
        # was asked for scores nothing for the ones it did write. Paying per-section
        # makes the rest optional, and an agent that learns to describe well while
        # skipping a section has learned to farm the auxiliary reward.
        #
        # There is deliberately no format reward of its own here. Writing the sections is
        # what makes the descriptions scoreable; it is not separately worth money. The one
        # format knob a run has is the environment's own `format_reward`.
        if len(present) < len(self.enabled):
            return scores

        # Both descriptions of a turn go out together, so a turn costs one round trip
        # rather than two.
        contents = [action[slice(*span)] for _, span in present]
        if self.scorer == "exact":
            parser = self.spec.exact_parser
            assert parser is not None  # validated once in __post_init__
            parsed = [parser(content) for content in contents]
        else:
            parsed = await self.judge.parse_batch(
                [self.spec.judge_prompt.format(content=content) for content in contents]
            )

        for (name, _), items in zip(present, parsed):
            # A description the judge could not structure scores nothing rather than
            # zero-by-default, so an outage reads as absence, not as failure.
            quality = (
                0.0
                if items is None
                else self._above_base(
                    grouped_f1(items, gold[name], self.spec.object_weights)
                )
            )
            scores[name] = self.enabled[name] * quality
        return scores

    def _above_base(self, f1: float) -> float:
        """How much of the description was better than saying something at random.

        The rescale keeps the configured per-turn reward honest: a perfect description
        still earns the full number written in the yaml. Scores below the calibrated
        baseline are clipped to zero.
        """
        base = float(self.score_base or 0.0)
        if base <= 0.0:
            return f1
        return max(0.0, (f1 - base) / (1.0 - base))

    def _place(self, scored: dict, outcome: float, token_ids, tokenizer) -> list[float]:
        """Pay each section's score on the last token of the section that earned it.

        On the *last* token because a span's score is a property of the whole span, so it
        is only determined at the token that completes it.

        This environment does not ask what the advantage estimator is. A reward belongs
        where it was earned, and per-span placement preserves that information.

        The outcome reward is different in kind -- it is the environment's verdict on the
        whole turn, not on any span of text -- so it goes on the turn's last token.

        Cost, stated because it is the price of the decoupling and not free: locating a
        span needs ``token_offsets``, which decodes every prefix of the response, O(n)
        decodes over O(n) characters once per turn per rollout. The old ``turn_end``
        placement skipped the tokenizer entirely. It bought that by knowing which
        estimator was downstream.
        """
        offsets = token_offsets(list(token_ids), tokenizer)
        auxiliary = [0.0] * len(offsets)
        if self.credit_site == "turn_end":
            if auxiliary:
                auxiliary[-1] = sum(
                    float(scored.get(name, 0.0) or 0.0) for name in self.enabled
                )
        else:
            unplaced = 0.0
            for name in self.enabled:
                span, value = scored["spans"].get(name), scored.get(name)
                # A section with no span scored nothing anyway -- `_score` gates on every
                # section being present -- but a score without the span that justifies it
                # must never be paid, however `_score` is changed later.
                if not value:
                    continue
                if span is None:
                    unplaced += float(value)
                    continue
                covered = tokens_covering(span, offsets)
                if covered:
                    auxiliary[covered[-1]] += value
                else:
                    unplaced += float(value)
            # Centered protocol penalties can belong to a missing span. They still need
            # a causal location; the action boundary is the first point where the
            # complete response is known to be malformed.
            if auxiliary and unplaced:
                auxiliary[-1] += unplaced

        vector = list(auxiliary)
        if vector:
            vector[-1] += outcome
        return (
            _EpisodeMeanVector(vector, auxiliary, self.episode_horizon)
            if self.aggregation == "episode_mean"
            else vector
        )

    def finalize_episode_scores(
        self, totals: dict[str, float], turns: int
    ) -> dict[str, float]:
        """Report the auxiliary totals under the rule actually used for training."""
        if self.aggregation != "episode_mean":
            return dict(totals)
        scale = int(self.episode_horizon) / max(1, int(turns))
        return {name: float(value) * scale for name, value in totals.items()}
