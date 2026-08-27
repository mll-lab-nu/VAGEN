"""Inference client — the only layer that knows about tokens.

The harness works in text and the env works in text; everything token-level is here, and
it is written once. What varies between experiments is the harness and the env, and
neither can reach a token through this interface.

A conversation id is the whole protocol. Passing one continues that conversation; passing
``None`` starts a new one. Concat keeps the same id for an episode, no-concat drops it
every turn, and compaction drops it when a budget is hit — three points on one axis
rather than three mechanisms. One conversation becomes one training row.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Optional

import logging

from vagen.rollout.trajectory import Conversation, Row

logger = logging.getLogger(__name__)


@dataclass
class Response:
    """What the harness gets back. Text plus the id needed to continue."""

    text: str
    conversation_id: str
    token_ids: Optional[list[int]] = None
    logprobs: Optional[list[float]] = None
    stop_reason: Optional[str] = None
    weights_version: Optional[tuple[int, int]] = None


@dataclass
class BackendOutput:
    """What a backend reports for one generation."""

    text: str
    token_ids: list[int]
    logprobs: Optional[list[float]] = None
    # The prompt the backend actually ran. Multimodal placeholders are expanded by the
    # backend, so this is not always the prompt it was handed -- see Conversation.
    prompt_token_ids: Optional[list[int]] = None
    #: "completed", "aborted", or None. Under partial rollout the resuming client absorbs
    #: aborts below this layer, so what arrives here is the outcome of the whole retry.
    stop_reason: Optional[str] = None
    #: The policy versions this response was generated across. They differ when a partial
    #: rollout resumed after a weight update, and then the response is on-policy for
    #: neither -- which is the only thing that says so. Carried rather than used: nothing
    #: reads them yet, and an off-policy correction that wanted to would otherwise have to
    #: reach back into the client to add them.
    weights_version: Optional[tuple[int, int]] = None


class EpisodeUnusable(Exception):
    """This episode cannot be finished; other episodes are unaffected.

    The distinction that matters is *what the failure is evidence of*. A configuration
    error is evidence about every episode, so it should stop the run. An observation that
    came back too large, or a conversation that could not buy a turn, is evidence about
    this rollout -- and taking the batch down for it means one unlucky environment sample
    costs an entire training step.

    Which is what happened: these escaped ``run_episode``, nothing caught them, and
    verl's ``asyncio.gather`` has no ``return_exceptions``.
    """


class ContextTooLarge(EpisodeUnusable, ValueError):
    """One call handed the model more context than the mode has room for.

    Context is everything it did not generate: the system prompt, an observation, a
    summary. Nothing bounded it before -- an environment returns what it returns, and a
    frame costs hundreds of tokens once the processor expands its placeholder -- so an
    observation that did not fit was found at the end of the episode by ``cap_token_ids``,
    if at all, by which point it is a truncated row and not an oversized observation.
    """


class InferenceClient(ABC):
    """Conversation bookkeeping, shared by every backend."""

    #: ``None`` for closed APIs that only return text.
    tokenizer: Any = None

    #: The most context one call may add, opening a conversation and continuing one. See
    #: ``vagen.harness._common.budget.context_limits``: the two differ because the calls do -- an
    #: opening call becomes a row's prompt region, a continuation appends to its response
    #: region. ``None`` disables the check, which is what evaluation against a closed API
    #: wants, since there is no row to fit.
    opening_limit: int | None = None
    continuation_limit: int | None = None

    def __init__(self):
        self._conversations: dict[str, Conversation] = {}
        self._counter = 0
        #: One warning per client, not per turn -- an environment that overruns the
        #: ceiling overruns it on every episode.
        self._warned_truncating_context = False

    @property
    def returns_token_ids(self) -> bool:
        """Training needs the ids. Checked at construction, not mid-episode."""
        return self.tokenizer is not None

    # ------------------------------------------------------------------ backend
    @abstractmethod
    def encode(self, messages: list[Any]) -> list[int]:
        """Render messages to tokens. Called only on messages not yet sent."""

    @abstractmethod
    async def generate(self, prompt_ids: list[int], **kwargs) -> BackendOutput:
        """Run the model."""

    # -------------------------------------------------------------------- send
    async def send(self, messages: list[Any], conversation_id: str | None = None, **kwargs) -> Response:
        conversation_id = self._open(conversation_id)
        conversation = self._conversations[conversation_id]

        # Encode exactly what the harness handed over. The harness already sends only
        # what is new -- deduplicating again here silently dropped every observation
        # after the first, since it sliced a one-message delta against a count of the
        # messages already sent.
        #
        # Measured on this encode rather than a second one: encoding runs the processor,
        # which is expensive, and it records the message's images against the
        # conversation -- so measuring separately would both cost twice and ship every
        # frame twice.
        opening = conversation.prompt_len is None
        messages = self._fit_messages(messages, opening=opening)
        context = self.encode(messages)
        conversation.add_context(context)

        output = await self._generate_nonempty(conversation, **kwargs)

        # Adopt what the backend ran, so the sequence trained on is the sequence sampled
        # from. This is also what repairs any seam left by rendering messages
        # incrementally, since the correction lands on the span just added.
        if output.prompt_token_ids is not None:
            # Adopting is the point -- the engine expands multimodal placeholders its own
            # way and we want the sequence we train on to be the one it ran. But adopting
            # silently absorbs a *systematic* disagreement too: `adopt_prompt` only guards
            # the head, so a per-image tiling difference lands entirely in the tail and is
            # swallowed, while multi_modal_inputs is still rebuilt locally from the frames.
            # The first thing to notice would be masked_scatter in the actor.
            expected = len(conversation.token_ids)
            got = len(output.prompt_token_ids)
            if got != expected:
                # Re-check the ceiling against what the engine actually ran. Measuring
                # before adoption meant the ceiling passed on our render and the batch
                # boundary then saw a longer sequence -- exactly the end-of-episode
                # surprise the per-call ceilings were introduced to replace.
                if opening:
                    self._check_context(list(output.prompt_token_ids), opening=True)
                logger.warning(
                    "the engine ran a prompt of %d tokens where this client rendered %d "
                    "(%+d). Adopting the engine's, which is right for training, but a "
                    "standing difference means the two are tiling images differently and "
                    "multi_modal_inputs is built from ours.", got, expected, got - expected)
            conversation.adopt_prompt(output.prompt_token_ids)
        conversation.add_response(output.token_ids, output.logprobs)

        # A backend that returns None -- what a closed API gives for a refusal or a
        # filtered completion -- must not reach the harness as an action. `accept`
        # forwards it, `while action is None` never exits, and the loop spins.
        return Response(
            text=output.text if isinstance(output.text, str) else "",
            conversation_id=conversation_id,
            token_ids=output.token_ids,
            logprobs=output.logprobs,
            stop_reason=output.stop_reason,
            weights_version=output.weights_version,
        )

    #: How many times to re-ask when a generation comes back with no tokens. See
    #: ``_generate_nonempty``. Zero disables the retry and lets the empty result through.
    empty_generation_retries: int = 3

    async def _generate_nonempty(self, conversation, **kwargs) -> BackendOutput:
        """Generate, re-asking if the engine returns nothing.

        A generation with no tokens is an interruption, not an answer -- an aborted or
        pre-empted request. Retrying is safe *because* it is empty: the environment is
        stepped on the action this call returns, so if there is no action there was no
        step, and the state being re-asked about is the state that was asked about. In
        compaction the retry re-sends the summary that opened this conversation, for the
        same reason: nothing downstream of it happened.

        That safety is a property of the caller's order, not of this function, and it
        only holds while an empty response cannot reach ``env.step``. It could: ``accept``
        forwards ``response.text``, which is ``""`` and not ``None``, so the episode used
        to advance a turn on an empty action and the environment did move. Retrying here
        is what keeps the premise true.

        verl's fully-async client does the same thing a layer below (resuming from
        ``prompt_ids + token_ids`` rather than re-asking), so under that configuration
        this never fires. It is for every other configuration, where nothing does.
        """
        for attempt in range(self.empty_generation_retries + 1):
            output = await self.generate(conversation.token_ids, **kwargs)
            if output.token_ids or attempt == self.empty_generation_retries:
                return output
            logger.warning(
                "generation %d/%d returned no tokens (interrupted); re-asking. The "
                "environment has not been stepped, so the state is unchanged.",
                attempt + 1, self.empty_generation_retries,
            )
        raise AssertionError("unreachable")

    def _open(self, conversation_id: str | None) -> str:
        if conversation_id is not None:
            if conversation_id not in self._conversations:
                raise KeyError(f"unknown conversation {conversation_id!r}; pass None to start one")
            return conversation_id
        new_id = f"c{self._counter + 1}"
        # Numbered here, where the order is what actually happened. Numbering them at the
        # far end by position in ``rows()`` would be a different thing: a conversation the
        # model never spoke in is dropped there, and the survivors after the gap would
        # each move down one -- with no hole to notice, since the ids stay contiguous.
        self._conversations[new_id] = Conversation(conversation_id=new_id, ordinal=self._counter)
        self._counter += 1
        return new_id

    # ------------------------------------------------------------------ reading
    def _fit_messages(self, messages: list, *, opening: bool) -> list:
        """Shrink an over-large observation to what this mode has room for.

        ``max_env_response_per_turn`` is a ceiling on what the environment may hand back in
        one turn, and this is where it becomes true rather than merely declared. An
        observation over it is cut; the ceiling exists so that an episode is bounded, and a
        bound that kills the rollout when an environment exceeds it does not bound
        anything, it moves the failure.

        ★ The cut is on the message TEXT, before rendering -- not on the rendered token
        span. Cutting the span is the obvious implementation and it is wrong: ``render``
        tokenizes with ``add_generation_prompt=True``, so the span ends
        ``<|im_end|>\n<|im_start|>assistant\n``, and a head-keeping cut throws that away
        first. The engine is then handed a prompt that stops mid-observation with no role
        boundary, the model continues the user's sentence, and ``add_response`` records
        those tokens at mask 1 and hands them to ``env.step`` as an action. Trimming the
        text and re-rendering cannot produce a malformed turn, because the template builds
        the boundary either way.

        ★ Only observations. An **opening** still raises: it is the system prompt plus the
        first observation, and cutting it would truncate the instructions identically on
        every episode of the run and train on the remainder -- a config error laundered
        into silently degraded data. An opening that does not fit means the prompt region
        is too small, which no cut repairs.
        """
        limit = self.opening_limit if opening else self.continuation_limit
        if limit is None:
            return messages
        size = self.measure(messages)
        if size <= limit:
            return messages
        if opening:
            # The part that cannot be cut. If it alone is over, no trimming helps.
            fixed = [m for m in messages if m.get("role") == "system"]
            if fixed and self.measure(fixed) > limit:
                self._check_context([0] * size, opening=True)

        trimmed, final = self._shrink(messages, limit)
        if not self._warned_truncating_context:
            self._warned_truncating_context = True
            logger.warning(
                "an observation came to %d tokens, over the %d this mode has room for; "
                "its text was cut to bring it to %d. Image placeholders are counted "
                "expanded, as the model sees them. Set max_env_response_per_turn to what "
                "the environment actually returns, shrink the observation (fewer or "
                "smaller frames, shorter text), or raise the budget it has to fit inside "
                "-- see vagen/harness/budget.py. Warned once per client.",
                size, limit, final,
            )
        return trimmed

    #: How many times ``_shrink`` may re-measure. Each pass scales the text by the ratio it
    #: is over by, so it converges fast; the cap is only there to stop a pathological
    #: tokenizer looping forever.
    _SHRINK_PASSES = 12

    def _shrink(self, messages: list, limit: int) -> tuple[list, int]:
        """Trim text, then whole images, until ``measure`` fits under ``limit``.

        ★ Refuses rather than reducing an observation to nothing. Without a floor this
        happily returned the empty string: measured on the shipped sokoban eval, where the
        ceiling (256, copied from the training yaml, where a frame really is ~96 tokens)
        sat below what evaluation charges for one frame (800 by estimate), every
        continuation observation became `""` with no image -- the model played blind from
        turn 2 and the run reported a success rate, exit 0, one warning. An unusable
        ceiling has to be loud; it is a config error, not something to silently absorb.
        """
        work = [dict(m) for m in messages]
        # The system prompt is instructions, not an observation: cutting it degrades every
        # episode of the run identically and invisibly.
        cuttable = [i for i, m in enumerate(work) if m.get("role") != "system"]
        size = self.measure(work)
        for _ in range(self._SHRINK_PASSES):
            if size <= limit:
                return work, size
            # Scale by how far over we are, with a margin, rather than nibbling: a token
            # is not a fixed number of characters and one pass per token would be O(n).
            keep = max(0.0, (limit / size) * 0.95)
            shrunk = [_scale_text(m, keep) if i in cuttable else m
                      for i, m in enumerate(work)]
            if _text_len(shrunk) == _text_len(work):
                break                      # text is exhausted; only images are left
            work = shrunk
            size = self.measure(work)
        # Text alone could not do it. Drop whole images from the end -- a partial image is
        # not an image, and the placeholder/frame counts have to stay 1:1.
        while size > limit and any(work[i].get("images") for i in cuttable):
            for i in reversed(cuttable):
                m = work[i]
                if m.get("images"):
                    m["images"] = list(m["images"])[:-1]
                    m["content"] = _drop_one_image_placeholder(m.get("content", ""))
                    break
            size = self.measure(work)
        nothing_left = (not _text_len([work[i] for i in cuttable])
                        and not any(work[i].get("images") for i in cuttable))
        if size > limit or nothing_left:
            raise ContextTooLarge(
                f"an observation of {self.measure(messages)} tokens cannot be brought under "
                f"the {limit}-token ceiling without deleting it: cutting reached "
                f"{size} tokens with {sum(len(m.get('images') or []) for m in work)} "
                f"image(s) and no text left. Raise max_env_response_per_turn, or -- if this "
                f"is an evaluation -- check `tokens_per_image` against what your "
                f"environment's frames actually cost; the default estimate is deliberately "
                f"high and a ceiling copied from a training config is priced differently."
            )
        return work, size

    def _check_context(self, context: list[int], *, opening: bool) -> None:
        """Raise if ``context`` is over the ceiling. For the cases a cut cannot repair.

        Two callers. An **opening** (see ``_fit_context``), and the **post-adoption**
        re-check, which runs after the engine has already produced tokens against this
        prompt -- cutting there would train on a sequence that was never sampled from,
        the exact divergence ``adopt_prompt`` exists to prevent.
        """
        limit = self.opening_limit if opening else self.continuation_limit
        if limit is None or len(context) <= limit:
            return
        what = ("the call opening a conversation, which is the system prompt and the "
                "first observation (and under compaction the summary too)"
                if opening else "an observation")
        raise ContextTooLarge(
            f"{what} came to {len(context)} tokens, over the {limit} this mode has room "
            f"for. Image placeholders are counted expanded, as the model sees them. "
            f"Set max_env_response_per_turn to what the environment actually returns, shrink "
            f"the observation (fewer or smaller frames, shorter text), or raise the "
            f"budget it has to fit inside -- see vagen/harness/budget.py for which one."
        )

    def reward(self, conversation_id: str, value: float | list[float]) -> None:
        """Credit the turn that just happened in this conversation."""
        self._conversations[conversation_id].add_reward(value)

    def rows(self) -> list[Row]:
        """One row per conversation the model spoke in.

        A conversation with no model output — a new one opened immediately before a
        terminal step — carries no gradient and is dropped rather than padded into the
        batch as an empty sequence.
        """
        return [c.row() for c in self._conversations.values() if c.is_trainable()]

    def usage(self, conversation_id: str) -> int:
        """How large this conversation has grown, in whatever unit the backend counts.

        Tokens here; a closed API would report the prompt size from its own ``usage``,
        counted by its own tokenizer, so a budget will trigger at slightly different
        points than in training. Log where it actually fires.
        """
        return len(self._conversations[conversation_id].token_ids)

    def response_len(self, conversation_id: str) -> int:
        """How much of ``max_response_length`` this conversation has spent."""
        return self._conversations[conversation_id].response_len

    def measure(self, messages: list) -> int:
        """Tokens these messages would add, without adding them.

        Backends that cannot render without side effects may override; the default routes
        through ``render``, which exists for exactly this.
        """
        rendered = self.render(messages)
        return len(rendered[0] if isinstance(rendered, tuple) else rendered)

    def render(self, messages: list):
        """Tokens for these messages, recording nothing. Defaults to ``encode`` for
        backends where encoding has no side effects to begin with."""
        return self.encode(messages)

    def conversations(self) -> list[Conversation]:
        return list(self._conversations.values())


#: The marker an environment puts where a frame goes. Kept in step with the frames list.
IMAGE_PLACEHOLDER = "<image>"


def _text_parts(content):
    """The mutable text-bearing parts of a message's content, whatever shape it is in."""
    if isinstance(content, str):
        return None
    return [p for p in content if isinstance(p, dict) and p.get("type") == "text"]


def _text_len(messages) -> int:
    total = 0
    for m in messages:
        c = m.get("content", "")
        if isinstance(c, str):
            total += len(c.replace(IMAGE_PLACEHOLDER, ""))
        else:
            total += sum(len(p.get("text", "")) for p in _text_parts(c) or ())
    return total


def _scale_text(message: dict, keep: float) -> dict:
    """A copy of ``message`` with its text cut to ``keep`` of its length, head kept.

    The head, because an observation leads with what changed and trails with boilerplate
    the model has already seen T-1 times.
    """
    out = dict(message)
    content = out.get("content", "")
    if isinstance(content, str):
        # ★ Scale the prose *between* the image placeholders, never the placeholders. A
        # plain slice of the string deletes `<image>` markers, and then the placeholders
        # no longer match the frames list -- the 1:1 invariant this whole path exists to
        # preserve. Caught by test_a_cut_drops_whole_images_once_the_text_is_gone.
        chunks = content.split(IMAGE_PLACEHOLDER)
        out["content"] = IMAGE_PLACEHOLDER.join(c[: int(len(c) * keep)] for c in chunks)
        return out
    parts = []
    for p in content:
        if isinstance(p, dict) and p.get("type") == "text":
            p = {**p, "text": p.get("text", "")[: int(len(p.get("text", "")) * keep)]}
        parts.append(p)
    out["content"] = parts
    return out


def _drop_one_image_placeholder(content):
    """Remove the last image placeholder, so placeholders and frames stay 1:1."""
    if isinstance(content, str):
        head, sep, tail = content.rpartition(IMAGE_PLACEHOLDER)
        return head + tail if sep else content
    parts = list(content)
    for i in range(len(parts) - 1, -1, -1):
        if isinstance(parts[i], dict) and parts[i].get("type") == "image":
            del parts[i]
            break
    return parts
