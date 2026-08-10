"""Rewarding the reasoning, not only the outcome.

Two rewards, independently switchable:

* **state estimation** -- the agent says where things are before acting
  (``<observation>``), scored against the state it acted *from*.
* **transition prediction** -- it says where they will be after
  (``<prediction>``), scored against the state it acted *into*.

★ Both scores are paid on the turn's **last** token, together with the outcome and the
format bonus. This was per-span -- each score on the last token of the section that earned
it, a signal *within* a turn -- which is the better placement for ``token_level_gae`` and
the variable-lambda ``bi_level_gae``, and the wrong one for the paper's nested Bi-Level
GAE, whose outer chain has a single reward slot per turn. Placement and advantage
estimator are one choice, not two; ``_place`` has the measurements. The per-section
breakdown moves to ``info`` rather than disappearing.

Whichever rewards are on decide the response format the agent is asked for, and the
format bonus is paid only when exactly those sections are present -- asking for a section
nobody scores, or scoring one nobody asked for, are both ways to get silent zeros.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from vagen.rewards.judge import NullJudge
from vagen.rewards.spans import tagged_span, token_offsets, tokens_covering
from vagen.rewards.spatial import grouped_f1

#: reward name -> the tag the agent writes it in
TAGS = {"state_estimation": "observation", "transition_prediction": "prediction"}

#: where a turn's scores are paid. See `StateRewardWrapper._place`.
PLACEMENTS = ("turn_end", "per_span")


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
    #: "turn_end" -- the whole turn's reward on its last token, which is what an
    #: estimator with one reward slot per turn requires; "per_span" -- each section's
    #: score on the last token of the section that earned it. See `_place`. Resolved from
    #: the advantage estimator in use by `gym_loop`, not set independently, because the
    #: two are one choice.
    placement: str = "turn_end"

    def __post_init__(self):
        unknown = set(self.enabled) - set(TAGS)
        if unknown:
            raise ValueError(f"unknown state rewards {sorted(unknown)}; choose from {sorted(TAGS)}")
        if self.placement not in PLACEMENTS:
            raise ValueError(f"unknown placement {self.placement!r}; choose from {sorted(PLACEMENTS)}")
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
        """Pay the turn's scores, either all on its last token or each on its own section.

        ★ Which of the two is right is decided by the advantage estimator, not by taste,
        and ``gym_loop`` resolves it from the estimator in use rather than letting the two
        be configured apart:

        ``turn_end``
            Everything on the turn's final token. Required by an estimator whose outer
            chain has one reward slot per turn -- ``bi_level_gae_paper`` reads a turn's
            reward only there, so a score left mid-turn is credited once by the inner
            token chain and again by the outer turn chain (measured bias 0.177 against an
            exact policy gradient, and a critic fixed-point error of exactly the misplaced
            weight).

        ``per_span``
            Each section's score on the last token of the section that earned it -- on the
            last, because a span's score is a property of the whole span, so it is
            determined at the step that completes it. Better for ``token_level_gae`` and
            the variable-lambda ``bi_level_gae``: measured -28% variance at lam 0.9 and
            -45% at 0.8, and exactly invariant once the critic is self-consistent, because
            a lumped score otherwise has to be *remembered* by ``V`` for the rest of the
            turn.

        The per-turn total is identical either way, so nothing here opens a length-hacking
        channel, and the per-section breakdown is reported through ``info`` regardless.

        ``turn_end`` also skips the tokenizer entirely. Locating a span needs
        ``token_offsets``, which decodes every prefix of the response -- O(n) decodes over
        O(n) characters, once per turn per rollout.
        """
        if self.placement == "per_span":
            offsets = token_offsets(list(token_ids), tokenizer)
            vector = [0.0] * len(offsets)
            for name in self.enabled:
                span, value = scored["spans"].get(name), scored.get(name)
                if span is None or not value:
                    continue
                covered = tokens_covering(span, offsets)
                if covered:
                    vector[covered[-1]] += value
            # The outcome and the format bonus belong to the turn as a whole either way.
            if vector:
                vector[-1] += outcome + scored.get("format", 0.0)
            return vector

        total = 0.0
        for name in self.enabled:
            value = scored.get(name)
            # A section with no span scored nothing anyway -- `_score` gates on every
            # section being present -- but a score without the span that justifies it
            # must never be paid, however `_score` is changed later.
            if scored["spans"].get(name) is None or not value:
                continue
            total += value

        vector = [0.0] * len(token_ids)
        if vector:
            vector[-1] += total + outcome + scored.get("format", 0.0)
        return vector
