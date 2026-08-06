"""Token accounting: what has to hold for a mode to produce an episode.

Six quantities bound a run. Three are enforced by something, three were not:

    C     rollout.max_model_len          hard. The engine refuses past it.
    n_p   data.max_prompt_length         a row's prompt region
    n_r   data.max_response_length       a row's response region
    g     response_length_per_turn       one generation. Hard: it is max_new_tokens.
    E     env_response_length            one observation, after the processor has
                                         expanded its images. Enforced here; nothing
                                         bounded it before.
    T     max_turns                      environment steps in an episode

and compaction adds two more that have to fit inside those:

    m     trainer.compact_budget         the largest a conversation may become
    k     trainer.compact_summary_budget the largest a summary may be

``S`` is the system prompt, measured rather than configured -- it comes from the
environment, so it is not known until an episode starts. Every check below that does not
mention ``S`` is decidable before the rollout and is made there; the ones that need it are
made on the first call, which is the earliest they can be.

What each mode has to satisfy
-----------------------------
A conversation is one training row: its opening call becomes the prompt region and
everything after becomes the response region.

    no_concat   S + E <= n_p                       one turn per conversation
                g <= n_r
                S + E + g <= C

    concat      S + E <= n_p                       one conversation per episode
                T*g + (T-1)*E <= n_r               every turn lands in one response region
                S + E + T*g + (T-1)*E <= C

    compact     S + k + E <= min(n_p, m)           a conversation opens on a summary
                m + E + g + |req| + k <= window    and is largest as it is summarised,
                                                   one turn past the trigger
                k <= g                             a summary is a generation
                2k <= m                            and a compression

The last two mention only configured numbers. The first mentions ``S`` and is checked on
the opening call; that ``m`` also has to be large enough to buy more than one turn depends
on what a turn actually costs and is checked as it happens.

The trigger
-----------
``m`` is the largest a conversation may become, so compaction fires when *one more turn
would not fit*, not when the budget is already gone. The original trigger, ``used >= m``,
cannot know what the turn it is about to admit will cost, so a conversation waved through
at ``m - 1`` still grows by a whole turn before anyone looks again and ``m`` is not
actually a bound on anything.

What one more turn costs is measured, not assumed: the largest continuation seen so far
this episode. Charging the configured ceiling ``E + g`` instead looks safer and is
useless -- on Sokoban ``g`` is 512 and a real turn is about 80, so a trigger charging 512
fires after the first turn of every conversation and compaction degenerates into no_concat
with a summary attached. Ceilings bound the worst case; they do not predict the next one.

    trigger      used + observed_turn_cost >= m
    guaranteed   peak <= m + E + g + |req| + k     (the ceilings do bound the overshoot)

which is what ``compact_budget_bounds`` inverts to get the largest workable ``m``.

Nothing here is sufficient. ``S`` and ``E`` are measured, not promised, and the lower end
of ``m`` -- is it big enough to buy more than one turn? -- depends on both, so it is not
decidable here at all. The runtime companions are what close that: the per-call ceilings
in the client, ``CompactionMakesNoProgress``, and ``cap_token_ids`` at the end.

Measured on Sokoban vision, for scale: S=589, E=44..58 (with a 96x96 frame), so a
conversation opens at 633 tokens. ``compact_budget=400`` -- which is what this ran with --
cannot hold the system prompt, let alone a turn. Every conversation summarised after one
turn and the mode was no_concat at twice the price, silently, for three runs.
"""

from __future__ import annotations

from dataclasses import dataclass


class BudgetError(ValueError):
    """The configured numbers cannot produce an episode.

    Raised before the rollout wherever the failure is decidable from config alone. The
    alternative is a crash hours in, or no crash at all: a mode that quietly degenerates
    into a more expensive version of another one still finishes and still writes
    well-formed rows.
    """


@dataclass(frozen=True)
class Budgets:
    """The numbers a mode has to work inside. ``compact_*`` are compact-only."""

    prompt_len: int                      # n_p
    response_len: int                    # n_r
    per_turn: int                        # g
    max_turns: int                       # T
    env_response: int = 0                # E
    context: int | None = None           # C; None means "whatever the regions allow"
    compact_budget: int | None = None    # m
    summary_budget: int | None = None    # k
    summary_request_len: int = 0         # |req|
    #: Whether ``per_turn`` is a real per-turn budget or the whole response region
    #: standing in for one. Unset it falls back to ``n_r``, and ``T*g`` then says only
    #: that a turn could in principle use everything -- true of every configuration, so
    #: a check on it would refuse them all.
    per_turn_configured: bool = True

    @property
    def row(self) -> int:
        """A conversation's total room: prompt region plus response region."""
        return self.prompt_len + self.response_len

    @property
    def window(self) -> int:
        """The tightest hard bound on one conversation."""
        return min(self.context, self.row) if self.context else self.row

    @property
    def window_name(self) -> str:
        return ("rollout.max_model_len" if self.context and self.context < self.row
                else "max_prompt_length + max_response_length")


# ------------------------------------------------------------------------ derivations
def default_summary_budget(compact_budget: int, per_turn: int) -> int:
    """``k`` when it is not configured: a quarter of what it must fit inside.

    Derived rather than a constant, so lowering ``m`` cannot leave behind a summary budget
    that is now larger than the thing it compresses into.
    """
    return max(1, min(per_turn, compact_budget // 4))


def default_env_response(mode: str, b: Budgets) -> int:  # noqa: D401
    """``E`` when it is not configured: the room the mode has left for observations.

    Read off the same relations that check it, so the default is by construction the
    largest value that passes. A configured ``E`` is better -- it makes the static checks
    exact rather than merely self-consistent -- but an unconfigured run should still be
    bounded by something other than nothing, which is what it was bounded by before.
    """
    # Every mode: an observation shares the opening call with the system prompt, so it
    # can never exceed the prompt region. Applied to all three because the relation is,
    # and leaving it out of concat and compact let the default exceed what check() would
    # accept -- a default rejected by its own checker fails every run that relied on it.
    ceiling = b.prompt_len
    if mode == "concat":
        # The response region holds T generations and the T-1 observations between them.
        spare = b.response_len - b.max_turns * b.per_turn
        ceiling = min(ceiling, max(0, spare) // max(1, b.max_turns - 1))
    elif mode == "compact" and b.compact_budget:
        # Bounded by the room the peak leaves, which is what compact_budget_bounds
        # measures from the other side: E appears in that overhead, so solving for it
        # gives the largest value whose own ceiling still admits this budget.
        k = b.summary_budget or default_summary_budget(b.compact_budget, b.per_turn)
        room = (min(b.window, b.response_len) - b.compact_budget
                - b.summary_request_len - k - b.per_turn)
        ceiling = min(ceiling, max(0, room))
    return ceiling


def compact_budget_bounds(b: Budgets) -> tuple[int, int]:
    """``(lowest, highest)`` workable ``m``, derived from everything else.

    The top is where a conversation stops being summarisable inside the window. The
    trigger fires on a measured turn cost, so a conversation can be admitted at ``m - 1``
    and then overshoot by at most one turn before it is closed -- ``E + g``, the ceilings,
    which is the one thing they are good for. Add the summary request and the summary and
    that is the peak the window has to hold.

    The bottom is only the compression rule, ``2k <= m``. Whether ``m`` is large enough to
    buy more than one turn depends on ``S`` and on what a turn actually costs, neither of
    which is known here; ``CompactionMakesNoProgress`` answers that at runtime, when both
    have been measured.
    """
    k = b.summary_budget or default_summary_budget(b.compact_budget or 4, b.per_turn)
    lowest = 2 * k
    overhead = b.summary_request_len + k + b.env_response + b.per_turn
    # Against the response region and not the window. Almost all of a conversation lands
    # in the response region -- only the opening call becomes the prompt region -- so a
    # large max_prompt_length hides an overflow rather than absorbing one. Charging the
    # opening at zero is the safe direction, since its real size is the system prompt and
    # that is not known until an episode starts.
    return lowest, min(b.window, b.response_len) - overhead


# ----------------------------------------------------------------------------- checks
def check(mode: str, b: Budgets) -> None:
    """Raise if ``mode`` cannot run inside ``b``. Silent when it can.

    Everything here is decidable without ``S``, which is why it runs before the rollout.
    """
    if b.per_turn > b.response_len:
        raise BudgetError(
            f"response_length_per_turn={b.per_turn} exceeds "
            f"data.max_response_length={b.response_len}: a single generation does not fit "
            f"the response region it is written to."
        )
    if b.context and b.per_turn > b.context:
        raise BudgetError(
            f"response_length_per_turn={b.per_turn} exceeds rollout.max_model_len={b.context}."
        )

    if mode == "no_concat":
        _need(b, b.env_response + b.per_turn, "one turn",
              f"env_response_length={b.env_response} + response_length_per_turn={b.per_turn}")

    if mode == "concat" and b.per_turn_configured:
        episode = b.max_turns * b.per_turn + max(0, b.max_turns - 1) * b.env_response
        if episode > b.response_len:
            raise BudgetError(
                f"concat keeps the whole episode in one conversation, so its response "
                f"region holds every turn: max_turns={b.max_turns} x "
                f"response_length_per_turn={b.per_turn} + {max(0, b.max_turns - 1)} x "
                f"env_response_length={b.env_response} = {episode} tokens, against "
                f"data.max_response_length={b.response_len}. Use trainer.harness=compact, "
                f"lower max_turns / response_length_per_turn / env_response_length, or "
                f"raise the budget."
            )
        _need(b, b.env_response + episode, "the whole episode",
              f"a first observation of {b.env_response} plus {episode} tokens of turns")

    if mode == "compact":
        _check_compact(b)


def _check_compact(b: Budgets) -> None:
    m, k = b.compact_budget, b.summary_budget
    if not m or m <= 0:
        raise BudgetError("trainer.harness=compact needs a positive trainer.compact_budget")
    if not k or k <= 0:
        raise BudgetError("trainer.harness=compact needs a positive trainer.compact_summary_budget")
    if k > b.per_turn:
        raise BudgetError(
            f"trainer.compact_summary_budget={k} exceeds "
            f"response_length_per_turn={b.per_turn}: the summary is a generation like any "
            f"other and cannot be longer than one."
        )
    lowest, highest = compact_budget_bounds(b)
    if lowest > highest:
        # Reporting only the ceiling here sends you round a loop: below the floor the
        # halving rule fires, above the ceiling this one does, and the advice from each
        # lands in the other's territory.
        raise BudgetError(
            f"no compact_budget works with these numbers. The compression rule puts the "
            f"floor at {lowest} (twice compact_summary_budget={k}) and the room left in "
            f"the response region puts the ceiling at {highest} "
            f"(min(window {b.window}, max_response_length {b.response_len}) less the "
            f"summary request {b.summary_request_len}, the summary {k}, one observation "
            f"{b.env_response} and one generation {b.per_turn}). Lower "
            f"compact_summary_budget or env_response_length or response_length_per_turn, "
            f"or raise max_response_length."
        )
    if m < lowest:
        raise BudgetError(
            f"trainer.compact_summary_budget={k} is more than half of "
            f"trainer.compact_budget={m}. A summary that long buys no turns: the next "
            f"conversation opens near the budget, summarises again after one turn, and every "
            f"environment step costs two generations and a summary of a single turn -- a "
            f"more expensive no_concat. compact_budget must be at least {lowest}."
        )
    if m > highest:
        peak = m + b.env_response + b.per_turn + b.summary_request_len + k
        raise BudgetError(
            f"a compacted conversation is largest at the moment it is summarised. The "
            f"trigger fires on a measured turn cost, so compact_budget={m} can be reached "
            f"and then overshot by one more turn (env_response_length={b.env_response} + "
            f"response_length_per_turn={b.per_turn}) before the summary request "
            f"({b.summary_request_len}) and the summary ({k}) go on top: {peak} tokens. "
            f"Almost all of that lands in the response region, so it is bounded by "
            f"min(window {b.window}, max_response_length {b.response_len}) and not by the "
            f"two regions added together -- compact_budget must be at most {highest}."
        )


def _need(b: Budgets, tokens: int, what: str, breakdown: str) -> None:
    """One conversation of ``tokens`` has to fit the hard window."""
    if tokens > b.window:
        raise BudgetError(
            f"{what} needs {tokens} tokens ({breakdown}), against a window of {b.window} "
            f"({b.window_name})."
        )


def context_limits(mode: str, b: Budgets) -> tuple[int, int]:
    """``(opening, continuation)``: the most context one call may add, enforced live.

    Context is everything the model is given that it did not generate -- system prompt,
    observation, summary -- measured after the processor has expanded image placeholders,
    since a frame costs hundreds of tokens and counting it as one would make the check
    decorative.

    The two differ because the calls do. A call that opens a conversation becomes that
    row's prompt region and is bounded by ``n_p``; a call that continues one appends to
    the response region, which already has to hold every generation in the episode. This
    is where ``S`` is finally charged: the opening ceiling is the only check that sees the
    system prompt, because it is the first thing to have measured it.

    Compaction bounds its openings by ``m`` as well, which is stricter and is the point --
    a conversation that opens at the budget summarises after one turn.
    """
    opening = b.prompt_len
    if mode == "compact" and b.compact_budget:
        opening = min(opening, b.compact_budget)
    # An observation is bounded by env_response_length wherever a conversation can be
    # continued. Compaction used the budget here, which bounded nothing worth bounding:
    # every relation compaction has to satisfy is written in terms of E, and E was the
    # one quantity with no runtime enforcement in the one mode that depends on it.
    #
    # Compaction continues a conversation for two reasons, though, and only one of them
    # is an observation: the summary request goes out on the same path. It is a fixed
    # string this module already measures, so admit it rather than reporting a 70-token
    # observation the environment never returned.
    continuation = b.env_response if mode in ("concat", "compact") else opening
    if mode == "compact":
        continuation = max(continuation, b.summary_request_len)
    return opening, continuation
