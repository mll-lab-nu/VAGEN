"""Token accounting: what has to hold for a mode to produce an episode.

    C     rollout.max_model_len          hard. The engine refuses past it.
    n_p   data.max_prompt_length         a row's prompt region
    n_r   data.max_response_length       a row's response region -- the real bound
    g     response_length_per_turn       one generation, and the floor below which one
                                         is not worth making
    E     env_response_length            one observation, after the processor has
                                         expanded its images
    T     max_turns                      environment steps in an episode
    k     trainer.compact_summary_budget the largest a summary may be
    m     trainer.compact_budget         optional: close conversations earlier than the
                                         region would

``S``, the system prompt, comes from the environment, so it is measured rather than
configured and nothing here can see it.

Most of what this module used to compute is gone, and the reason is in
``core/harness.py``: a conversation is now generated against the room it has left. Every
call asks how much of ``n_r`` has been spent and how big the observation about to be sent
is, bounds the generation by what remains, and -- when a turn no longer fits -- compacts,
or stops. Accumulation is bounded by the region itself, so the arithmetic that used to
solve for a workable ``m`` from a worst-case sum has nothing left to do.

What remains here is of two kinds.

**Refusals**, for the things no runtime can recover from:

    k <= g                      a summary is a generation, and the client clamps it to
                                that limit -- reserving more reserves room for something
                                that cannot be written
    k + |req| + g <= n_r        or every conversation closes before its first turn

**Warnings**, for worst cases that a real rollout may never reach:

    T*g + (T-1)*E <= n_r        concat's episode, if every turn used its full allowance
    2k <= m                     the optional trigger set somewhere unhelpful

The distinction matters. Refusing on a worst case rules out any long episode on the
strength of a case that does not happen, and that is what makes a long-tail rollout
impossible to debug. What overflows anyway is truncated, image-aware, at the batch
boundary -- and what gets truncated is context, since the model's own tokens are bounded
by ``max_new_tokens`` and only observations can overflow.

Measured on Sokoban vision, for scale: S=589, E=44..58 with a 96x96 frame, a real turn
about 164 tokens. A 20-turn episode is ~3200 tokens against a 6144 region -- which is why
the region trigger alone never fires there, and why ``m`` survives as a second one.
"""

from __future__ import annotations

import warnings
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
    # What *one* observation may be, not what all of them may sum to. The sum stopped
    # being this number's business when generation became budget-aware: a conversation
    # is generated against the room it has left and stops -- or compacts -- when a turn
    # no longer fits, so accumulation is bounded by the region itself.
    #
    # Deriving it from the worst-case sum, as this used to, made it *negative* the moment
    # max_turns x per_turn exceeded the region, clamped to zero, and then refused every
    # observation the environment produced. Measured: max_turns=20 with per_turn=512
    # against a 6144-token region gave E=0 and killed the run on a 47-token observation.
    # A ceiling whose job is to catch one pathological observation must not be computed
    # from an aggregate a real rollout never reaches.
    if mode == "no_concat":
        # Every call opens a conversation, so the observation lands in the prompt region.
        return max(1, b.prompt_len)
    # Room for one observation and the generation that answers it.
    return max(1, b.response_len - b.per_turn)


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
            # A warning, not an error. This is the worst case -- every turn generating
            # its full allowance -- and it is now survivable two ways: the generation is
            # bounded by the room actually left, and what overflows anyway is truncated
            # rather than refused. Refusing here would rule out any long episode on the
            # strength of a worst case that a real rollout does not reach, which is the
            # bluntness that makes a long-tail run impossible to debug.
            warnings.warn(
                f"concat keeps the whole episode in one conversation, so its response "
                f"region holds every turn: max_turns={b.max_turns} x "
                f"response_length_per_turn={b.per_turn} + {max(0, b.max_turns - 1)} x "
                f"env_response_length={b.env_response} = {episode} tokens, against "
                f"data.max_response_length={b.response_len}. Turns will be generated "
                f"against the room left rather than the full allowance, and anything over "
                f"is truncated -- but if episodes really do run that long, use "
                f"trainer.harness=compact or lower max_turns.",
                stacklevel=2,
            )
        _need(b, b.env_response + episode, "the whole episode",
              f"a first observation of {b.env_response} plus {episode} tokens of turns")

    if mode == "compact":
        _check_compact(b)


def _check_compact(b: Budgets) -> None:
    """What compaction still needs from the numbers.

    Much less than it used to. ``compact_budget`` was the bound; it is now an optional
    second trigger, and the real bound is the response region -- a conversation is
    generated against the room it has left and closed before the summary would not fit.
    So the checks that solved for a workable ``m`` are gone, and what remains is the two
    things the runtime cannot recover from.
    """
    k = b.summary_budget
    if not k or k <= 0:
        raise BudgetError("trainer.harness=compact needs a positive trainer.compact_summary_budget")
    if k > b.per_turn:
        raise BudgetError(
            f"trainer.compact_summary_budget={k} exceeds "
            f"response_length_per_turn={b.per_turn}: the summary is a generation like any "
            f"other, and the client clamps it to that limit -- so the reservation would be "
            f"larger than anything that can be written into it."
        )

    # A floor has to exist: room for the summary, its request, and one generation worth
    # making. Below this no configuration works, whatever compact_budget says.
    needed = k + b.summary_request_len + b.per_turn
    if needed > b.response_len:
        raise BudgetError(
            f"a conversation has no room to work in: compact_summary_budget={k} + the "
            f"summary request ({b.summary_request_len}) + response_length_per_turn="
            f"{b.per_turn} = {needed}, against data.max_response_length={b.response_len}. "
            f"Every conversation would close before its first turn. Lower "
            f"compact_summary_budget or response_length_per_turn, or raise "
            f"max_response_length."
        )

    m = b.compact_budget
    if m and 2 * k > m:
        # Advisory now, not fatal: the region trigger is the one that has to hold, and
        # this only says the optional second trigger is set somewhere unhelpful.
        warnings.warn(
            f"trainer.compact_summary_budget={k} is more than half of "
            f"trainer.compact_budget={m}, so the optional budget trigger will close "
            f"conversations that a summary of its own size could not usefully compress. "
            f"Raise compact_budget above {2 * k}, or leave it unset and let the response "
            f"region decide.",
            stacklevel=3,
        )

def _need(b: Budgets, tokens: int, what: str, breakdown: str) -> None:
    """One conversation of ``tokens`` has to fit the hard window."""
    if tokens > b.window:
        warnings.warn(
            f"{what} needs {tokens} tokens ({breakdown}), against a window of {b.window} "
            f"({b.window_name}). This is the worst case; generation is bounded by the "
            f"room left and the overflow is truncated, but the numbers do not leave "
            f"headroom for it.",
            stacklevel=2,
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

    Compaction no longer bounds its openings by ``m``. It used to, back when ``m`` was
    what a conversation had to fit inside; ``m`` is an optional *trigger* now -- close
    earlier than the region would -- and treating a trigger as a ceiling made the lever
    unusable in exactly the range it is for. ``compact_budget=400`` is accepted by every
    static check and then dies on the first call of the episode, because the opening is
    the system prompt plus the first observation (647 tokens on Sokoban) and there is no
    summary in it yet to be compacted away. Whether a conversation opens too full to buy
    a turn is a runtime question, and ``CompactionMakesNoProgress`` is where it is asked.
    """
    opening = b.prompt_len
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
