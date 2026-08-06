"""Whether the configured numbers can produce an episode at all.

Four numbers bound a run:

    n_prompt   data.max_prompt_length        a row's prompt region
    n_resp     data.max_response_length      a row's response region
    per_turn   response_length_per_turn      one generation, if the env config sets one
    max_turns  the dataset row               env steps in an episode

Only the first two are enforced anywhere, and they are enforced at the *end*, when the
episode is already over and the rollout already paid for -- see ``cap_token_ids``. Whether
they can be met is a property of the context policy, since the policy is what decides how
an episode is spread across rows, and it is decidable before the run starts.

Compaction is where the arithmetic actually bites, because it introduces two more numbers
that have to fit inside the first four:

    m   trainer.compact_budget           summarise once the conversation reaches this
    k   trainer.compact_summary_budget   how long the summary may be

and three relations between them:

    per_turn  <= n_resp                  one generation fits the region it is written to
    m + |req| + k <= n_prompt + n_resp   the conversation is largest just after it is
                                         summarised, and that is the row it becomes
    2k <= m                              a summary is a compression. Allowed to be as
                                         long as the budget it must fit inside, it buys
                                         no turns: the next conversation opens already
                                         near m, summarises again after one turn, and
                                         every environment step costs two generations
                                         plus a summary of a single turn.

The third is the one that was wrong here and produced no symptom. ``k`` was not a setting
at all -- the summary generated against ``response_length_per_turn``, so at m=400 and
per_turn=8000 the summary was permitted to be twenty times the budget it was compressing
into. It worked because the model happened to write short summaries.

Nothing here is sufficient. An observation is whatever the environment returns and no
arithmetic bounds it, so a run that passes these checks can still overflow on data; that
is what the guard in ``cap_token_ids`` is for. These are the failures that do not need
data to find.
"""

from __future__ import annotations

from dataclasses import dataclass


class BudgetError(ValueError):
    """The configured budgets cannot produce an episode.

    Raised before the run rather than during it: every one of these is decidable from the
    config alone, and the alternative is finding out from a crash several hours in, or
    from a mode that silently degenerates into a more expensive version of another one.
    """


@dataclass(frozen=True)
class Budgets:
    """The numbers a mode has to work inside. ``summary_*`` are compact-only."""

    prompt_len: int
    response_len: int
    per_turn: int
    max_turns: int
    compact_budget: int | None = None
    summary_budget: int | None = None
    summary_request_len: int = 0
    #: Whether ``per_turn`` is a real per-turn budget or just the whole response region
    #: standing in for one. Unset, it falls back to ``n_resp``, and multiplying that by
    #: max_turns says only that a turn could in principle use the whole budget -- true of
    #: every configuration, so a check on it would refuse them all.
    per_turn_configured: bool = True

    @property
    def window(self) -> int:
        """What one conversation may reach in total, prompt region plus response region."""
        return self.prompt_len + self.response_len


def default_summary_budget(compact_budget: int, per_turn: int) -> int:
    """A summary budget derived from what it must fit inside, when none is configured.

    A quarter of the budget leaves three quarters for the system prompt, the observation
    that opens the new conversation, and the turns the compaction exists to buy. Derived
    rather than a constant, so lowering ``compact_budget`` does not quietly leave a
    summary budget behind that is now larger than the thing it compresses into.
    """
    return max(1, min(per_turn, compact_budget // 4))


def check(mode: str, b: Budgets) -> None:
    """Raise if ``mode`` cannot run inside ``b``. Returns nothing when it can."""
    if b.per_turn > b.response_len:
        raise BudgetError(
            f"response_length_per_turn={b.per_turn} exceeds "
            f"data.max_response_length={b.response_len}, so a single generation does not "
            f"fit the response region it is written to. Lower the former or raise the latter."
        )

    if mode == "concat":
        # Necessary, not sufficient: this ignores every observation, and in concat the
        # observations sit in the response region too. If the responses alone do not fit,
        # no episode of this length can complete whatever the environment returns.
        floor = b.max_turns * b.per_turn
        if b.per_turn_configured and floor > b.response_len:
            raise BudgetError(
                f"concat keeps the whole episode in one conversation, so its response "
                f"region has to hold every turn: max_turns={b.max_turns} x "
                f"response_length_per_turn={b.per_turn} = {floor} tokens against "
                f"data.max_response_length={b.response_len} -- and that is before a single "
                f"observation, which concat also stores there. Use trainer.harness=compact, "
                f"lower max_turns or response_length_per_turn, or raise the budget."
            )

    if mode == "compact":
        m, k = b.compact_budget, b.summary_budget
        if not m or m <= 0:
            raise BudgetError("trainer.harness=compact needs a positive trainer.compact_budget")
        if not k or k <= 0:
            raise BudgetError("trainer.harness=compact needs a positive trainer.compact_summary_budget")
        if k > b.per_turn:
            raise BudgetError(
                f"trainer.compact_summary_budget={k} exceeds "
                f"response_length_per_turn={b.per_turn}; the summary is a generation like "
                f"any other and cannot be longer than one."
            )
        if 2 * k > m:
            raise BudgetError(
                f"trainer.compact_summary_budget={k} is more than half of "
                f"trainer.compact_budget={m}. A summary that long buys no turns: the next "
                f"conversation opens near the budget, summarises again after one turn, and "
                f"every environment step then costs two generations and a summary of a "
                f"single turn -- a more expensive no_concat. Set the summary budget to "
                f"about {default_summary_budget(m, b.per_turn)}, or raise compact_budget."
            )
        peak = m + b.summary_request_len + k
        if peak > b.window:
            raise BudgetError(
                f"a compacted conversation is largest at the moment it is summarised: "
                f"compact_budget={m} + the summary request ({b.summary_request_len} tokens) "
                f"+ compact_summary_budget={k} = {peak} tokens, against a window of "
                f"max_prompt_length={b.prompt_len} + max_response_length={b.response_len} "
                f"= {b.window}. That conversation is one training row, so it has to fit."
            )
