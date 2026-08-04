"""Rewarding the reasoning, not only the outcome.

The agent is asked to say where things are before it acts (``<observation>``) and where
they will be afterwards (``<prediction>``). Each description is scored against what the
environment actually contained, and the score is paid to the tokens that carry it.

★ That placement is the point. A scalar added to the turn's reward tells the credit
assignment only that the turn went well; putting the grounding score on the grounding
tokens and the prediction score on the prediction tokens is a signal *within* a turn,
which is the level a token- or removed_estimator estimator can act on. It is available because
the environment interface returns a vector aligned to the response.

Timing follows the meaning: grounding describes the state the agent acted *from*, so it
is scored against the state before the step; prediction describes what it acted *into*,
so against the state after.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from vagen.rewards.judge import NullJudge
from vagen.rewards.spans import spread, tagged_span, token_offsets, tokens_covering
from vagen.rewards.spatial import grouped_f1


@dataclass
class StateRewardSpec:
    """What an environment must supply to be scored this way."""

    #: env -> the relations actually present, as {"object_id", "vertical_relation", ...}
    relations: Callable[[Any], list[dict]]
    #: format(content=...) -> the prompt that asks the judge to structure a description
    judge_prompt: str
    #: per object type, how much of the score it accounts for
    object_weights: dict[str, float]
    #: appended to the system prompt so the agent knows the format expected of it
    instructions: str = ""


@dataclass
class StateRewardWrapper:
    """An environment whose reward includes how well the agent described the world."""

    env: Any
    spec: StateRewardSpec
    judge: Any = field(default_factory=NullJudge)
    grounding_weight: float = 0.5
    worldmodeling_weight: float = 0.5
    format_reward: float = 0.1

    def __post_init__(self):
        self.last_scores: dict[str, float] = {}

    async def reset(self, seed=None):
        return await self.env.reset(seed=seed)

    async def system_prompt(self):
        prompt = await self.env.system_prompt()
        if not self.spec.instructions:
            return prompt
        text = prompt.get("obs_str", "") if isinstance(prompt, dict) else str(prompt)
        joined = f"{text}\n\n{self.spec.instructions}"
        return {**prompt, "obs_str": joined} if isinstance(prompt, dict) else joined

    async def close(self):
        await self.env.close()

    async def step(self, action: str, response_token_ids=None, tokenizer=None):
        before = self.spec.relations(self.env)
        obs, reward, done, info = await self.env.step(action)
        after = self.spec.relations(self.env)

        scored = await self._score(action, before, after)
        self.last_scores = {k: v for k, v in scored.items() if isinstance(v, float)}
        info = {**(info or {}), **{f"state_reward/{k}": v for k, v in self.last_scores.items()}}

        if response_token_ids is None or tokenizer is None:
            # No tokens to place anything on; fall back to the scalar the env gave plus
            # the process scores, so the wrapper degrades rather than silently vanishing.
            return obs, float(reward) + sum(self.last_scores.values()), done, False, info

        vector = self._place(scored, float(reward), response_token_ids, tokenizer)
        return obs, vector, done, False, info

    # ------------------------------------------------------------------ scoring
    async def _score(self, action: str, before: list[dict], after: list[dict]) -> dict:
        spans = {"observation": tagged_span(action, "observation"), "prediction": tagged_span(action, "prediction")}
        asked = [(tag, span) for tag, span in spans.items() if span is not None]
        if not asked:
            return {"format": 0.0, "spans": spans}

        parsed = await self.judge.parse_batch(
            [self.spec.judge_prompt.format(content=action[slice(*span)]) for _, span in asked]
        )

        gold = {"observation": before, "prediction": after}
        weights = {"observation": self.grounding_weight, "prediction": self.worldmodeling_weight}
        scores: dict[str, Any] = {"spans": spans, "format": self.format_reward if len(asked) == 2 else 0.0}
        for (tag, _), items in zip(asked, parsed):
            # A description the judge could not structure scores nothing rather than
            # zero-by-default, so a judge outage is visible as absence, not as failure.
            scores[tag] = 0.0 if items is None else weights[tag] * grouped_f1(items, gold[tag], self.spec.object_weights)
        return scores

    def _place(self, scored: dict, outcome: float, token_ids, tokenizer) -> list[float]:
        offsets = token_offsets(list(token_ids), tokenizer)
        vector = [0.0] * len(offsets)

        for tag in ("observation", "prediction"):
            span, value = scored["spans"].get(tag), scored.get(tag)
            if span is None or not value:
                continue
            for i, share in enumerate(spread(value, tokens_covering(span, offsets), len(offsets))):
                vector[i] += share

        # Outcome and format belong to the turn as a whole, so they sit on its last
        # token -- the position a return is bootstrapped from.
        if vector:
            vector[-1] += outcome + scored.get("format", 0.0)
        return vector
