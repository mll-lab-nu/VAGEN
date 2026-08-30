"""Rewarding the reasoning, not only the outcome.

Two rewards, independently switchable:

* **state estimation** -- the agent says where things are before acting
  (``<perception>``), scored against the state it acted *from*.
* **transition prediction** -- it says where they will be after
  (``<prediction>``), scored against the state it acted *into*.

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
#: choice, so merely *naming* a relation scores about a third. Measured over 300 real
#: Sokoban starts: uniform random 0.334, and the best constant answer ("same", "same") --
#: which describes nothing and looks at nothing -- 0.391. The reward's usable range was
#: 0.39 to 1.00, not 0 to 1, and observed model scores of 0.45-0.52 sat barely above it.
#: That is what "the description is visibly wrong but the reward is high" looks like.
#:
#: 0.334 is the uniform-random floor rather than the 0.391 constant-answer floor, so a
#: policy that finds the most common relation and repeats it still earns a little. Raise
#: it to 0.391 to take that away too.
DEFAULT_SCORE_BASE = 0.334


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


@dataclass
class StateRewardWrapper:
    """An environment whose reward includes how well the agent described the world."""

    env: Any
    spec: StateRewardSpec
    judge: Any = field(default_factory=NullJudge)
    #: reward name -> what ONE turn pays for a perfect description of that section.
    #: An absolute per-turn number, straight from the environment's config: there is no
    #: episode budget divided by a turn count anywhere. The number in the yaml is the
    #: number a turn can earn, so raising `max_turns` does not silently change what the
    #: auxiliary signal is worth relative to the task.
    enabled: dict[str, float] = field(default_factory=dict)
    #: Subtracted from each f1 before it is paid, then the remainder is rescaled so a
    #: perfect description still earns the full per-turn reward:
    #: ``max(0, (f1 - base)/(1 - base))``. Rescaled rather than merely shifted, so the
    #: number configured stays the number a perfect turn earns. 0.0 restores the legacy
    #: reward exactly. See `DEFAULT_SCORE_BASE` for where the number comes from.
    score_base: float = DEFAULT_SCORE_BASE

    def __post_init__(self):
        unknown = set(self.enabled) - set(TAGS)
        if unknown:
            raise ValueError(f"unknown state rewards {sorted(unknown)}; choose from {sorted(TAGS)}")
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
        sections = [self.spec.examples[name] for name in self.enabled if name in self.spec.examples]
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

        scored = await self._score(action, {"state_estimation": before, "transition_prediction": after})
        self.last_scores = {k: v for k, v in scored.items() if k != "spans"}
        info = {**(info or {}), **{f"state_reward/{k}": v for k, v in self.last_scores.items()}}

        if response_token_ids is None or tokenizer is None:
            # Nothing to place anything on; degrade to a scalar rather than vanish.
            return obs, float(reward) + sum(self.last_scores.values()), done, info

        return obs, self._place(scored, float(reward), response_token_ids, tokenizer), done, info

    async def _score(self, action: str, gold: dict[str, list[dict]]) -> dict:
        spans = {name: tagged_span(action, tag) for name, tag in TAGS.items() if name in self.enabled}
        present = [(name, span) for name, span in spans.items() if span is not None]

        scores: dict[str, Any] = {"spans": spans}
        for name in self.enabled:
            scores[name] = 0.0

        # Auxiliary supervision follows the same canonical ordering as the environment
        # format reward. Old tags or reordered fields may still be useful for diagnostics,
        # but they must not earn reward-model credit.
        if not parse_wm_sections(action, allow_native_thinking=True).format_correct:
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
        parsed = await self.judge.parse_batch(
            [self.spec.judge_prompt.format(content=action[slice(*span)]) for _, span in present]
        )

        for (name, _), items in zip(present, parsed):
            # A description the judge could not structure scores nothing rather than
            # zero-by-default, so an outage reads as absence, not as failure.
            scores[name] = (
                0.0 if items is None
                else self.enabled[name] * self._above_base(
                    grouped_f1(items, gold[name], self.spec.object_weights)
                )
            )
        return scores

    def _above_base(self, f1: float) -> float:
        """How much of the description was better than saying something at random.

        ``max(0, (f1 - base) / (1 - base))``. The rescale keeps the configured per-turn
        reward honest: a perfect description still earns the full number written in the
        yaml. A plain subtraction would quietly cap the achievable auxiliary reward at
        ``(1 - base)`` of what the config promises.

        Clipped at zero: a description worse than chance is not a debt, it is just worth
        nothing, and a negative auxiliary reward would push against the task reward.
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
        vector = [0.0] * len(offsets)
        for name in self.enabled:
            span, value = scored["spans"].get(name), scored.get(name)
            # A section with no span scored nothing anyway -- `_score` gates on every
            # section being present -- but a score without the span that justifies it
            # must never be paid, however `_score` is changed later.
            if span is None or not value:
                continue
            covered = tokens_covering(span, offsets)
            if covered:
                vector[covered[-1]] += value
        if vector:
            vector[-1] += outcome
        return vector
