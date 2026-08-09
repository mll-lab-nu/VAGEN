"""Rewarding the reasoning, not only the outcome.

Two rewards, independently switchable:

* **state estimation** -- the agent says where things are before acting
  (``<observation>``), scored against the state it acted *from*.
* **transition prediction** -- it says where they will be after
  (``<prediction>``), scored against the state it acted *into*.

★ Each score is paid to the tokens that carry it. A scalar added to the turn tells credit
assignment only that the turn went well; putting the estimation score on the estimation
tokens and the prediction score on the prediction tokens is a signal *within* a turn,
which is the level a token- or removed_estimator estimator can act on. It is available because the
environment interface returns a vector aligned to the response.

Whichever rewards are on decide the response format the agent is asked for, and the
format bonus is paid only when exactly those sections are present -- asking for a section
nobody scores, or scoring one nobody asked for, are both ways to get silent zeros.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from vagen.rewards.judge import NullJudge
from vagen.rewards.spans import spread, tagged_span, token_offsets, tokens_covering
from vagen.rewards.spatial import grouped_f1

#: reward name -> the tag the agent writes it in
TAGS = {"state_estimation": "observation", "transition_prediction": "prediction"}


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
    #: reward name -> weight; only the names present here are asked for and scored
    enabled: dict[str, float] = field(default_factory=dict)
    format_reward: float = 0.1

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
        """The response format, for whichever rewards are on and the env has not asked for.

        Sections the environment's own prompt already requests are skipped. Sokoban's
        "wm" format asks for <observation> and <prediction> in natural language, with
        worked examples; appending a second set of instructions for the same tags does
        not reinforce them, it competes with them. Adding a JSON example that way made
        the agent emit the schema and the judge a re-parser of its own output; replacing
        it with prose left two differently-worded blocks asking for one thing, and six
        of eight episodes stopped producing a usable action at all.

        Built rather than selected from a table of combinations: with two independent
        switches a table has four entries that drift apart, and asking for a section
        that nothing scores trains the agent to write text for no reason.
        """
        sections = [
            self.spec.examples[name]
            for name, tag in TAGS.items()
            if name in self.enabled and name in self.spec.examples and f"<{tag}>" not in existing
        ]
        if not sections:
            return ""
        body = "\n".join(sections)
        return f"{body}\n{self.spec.axes}".strip()

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

        scores: dict[str, Any] = {"spans": spans, "format": 0.0}
        for name in self.enabled:
            scores[name] = 0.0

        # Format is a gate, not a line item: a turn that did not write every section it
        # was asked for scores nothing for the ones it did write. Paying per-section
        # makes the rest optional, and an agent that learns to describe well while
        # skipping a section has learned to farm the auxiliary reward.
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
                0.0 if items is None else self.enabled[name] * grouped_f1(items, gold[name], self.spec.object_weights)
            )
        scores["format"] = self.format_reward
        return scores

    def _place(self, scored: dict, outcome: float, token_ids, tokenizer) -> list[float]:
        offsets = token_offsets(list(token_ids), tokenizer)
        vector = [0.0] * len(offsets)

        for name in self.enabled:
            span, value = scored["spans"].get(name), scored.get(name)
            if span is None or not value:
                continue
            covered = tokens_covering(span, offsets)
            if not covered:
                continue
            # ★ On the span's LAST token, not spread across it. A span's score is a
            # property of the whole span -- you cannot tell whether a state estimation is
            # right from half of it -- so the reward is determined at the step that
            # completes the span, and that is where it belongs. The recursion then hands
            # it back to every token that caused it.
            #
            # Spreading gave the opposite gradient: with `value/K` at each of K tokens,
            # G_j (the return from j) collects only the shares at or after j, so the
            # span's FIRST token received the whole value and its LAST received value/K.
            # That is backwards -- the last token is the one that finished the thing being
            # scored.
            #
            # Same rule as `outcome` below, which already sits on the turn's last token:
            # every score lands on the last token of whatever it scores. The total is
            # unchanged, so the length-hacking channel `spread` was guarding stays shut.
            vector[covered[-1]] += value

        # Outcome and format belong to the turn as a whole, so they sit on its last
        # token -- the position a return is bootstrapped from.
        if vector:
            vector[-1] += outcome + scored.get("format", 0.0)
        return vector
