"""Check that the prompt trained on is the prompt that produced the response.

Multimodal prompts exist in two forms. The agent loop tokenizes one with the HF
processor, which expands an image into its full run of placeholder tokens. The engine is
handed that prompt with the runs collapsed back to one token each, because it expands
them itself from the images it is given.

Nothing forces the two expansions to agree. When they do not, the engine samples from one
sequence while training computes log-probs over another -- silently, since both are
well-formed and neither side sees both. It is an off-policy corruption that no metric
names: the loss stays finite and the score merely fails to improve.

So the engine reports the length of the prompt it actually ran, and this compares it
against the one being kept.
"""

from __future__ import annotations

import warnings


class PromptLengthMismatch(RuntimeError):
    """Raised when the engine ran a different prompt from the one being trained on."""


def engine_prompt_ids(output) -> list[int] | None:
    """The prompt the engine actually ran, if it reported one.

    Adopting these instead of the locally tokenized ids makes the training sequence the
    sampling sequence by construction, for any model family -- no per-family expansion
    rules to reimplement and keep in step with the engine.

    Only safe where the caller rebuilds its prompt each turn. A loop that accumulates
    one prompt across turns tracks its response mask by appending counts, and adopting a
    re-expanded prompt would move tokens the mask has already been measured against.
    """
    ids = (getattr(output, "extra_fields", None) or {}).get("prompt_token_ids")
    return list(ids) if ids else None


def check_prompt_matches_engine(prompt_ids, output, *, env_name: str = "?", strict: bool = True) -> None:
    """Compare the kept prompt against the one the engine reported running.

    Silent when the engine reports nothing -- older servers, and non-vLLM backends, do
    not carry the field, and their absence is not evidence of a problem.
    """
    reported = (getattr(output, "extra_fields", None) or {}).get("prompt_token_count")
    if reported is None:
        return

    kept = len(prompt_ids)
    if kept == reported:
        return

    message = (
        f"prompt length disagrees with the inference engine in env {env_name!r}: "
        f"tokenized {kept}, engine ran {reported} ({reported - kept:+d}). "
        "The two expand image placeholders differently, so training would compute "
        "log-probs over a sequence the model never saw. Align the preprocessing "
        "config on both sides -- data.mm_processor_kwargs reaches the tokenizer and "
        "the engine."
    )
    if strict:
        raise PromptLengthMismatch(message)
    warnings.warn(message, stacklevel=2)


def adopt_engine_prompt(engine_ids, prompt_ids, response_mask, response_logprobs, tail_obs_len, prefix_len):
    """Adopt the engine's prompt in a loop that accumulates one across turns.

    Returns ``(mask, logprobs, tail_obs_len, prefix_len)`` for the adopted prompt.

    Only the newest observation can change length. Everything before it was adopted on
    an earlier turn, so it is already in the engine's own form and re-expanding it is
    idempotent -- the deduplication that precedes the round trip is exactly its inverse.
    That makes the whole delta attributable to the trailing observation, which is a run
    of zeros of known length, so the mask can be corrected without having to locate
    anything inside the token stream.

    ``tail_obs_len`` is ``None`` on the first turn, where the tail is the initial prompt
    and the mask covers none of it.

    The invariant is asserted rather than assumed: the prompt and the mask must still
    describe the same sequence afterwards.
    """
    delta = len(engine_ids) - len(prompt_ids)

    if delta and tail_obs_len is not None:
        adjusted = tail_obs_len + delta
        if adjusted < 0:
            raise PromptLengthMismatch(
                f"engine prompt is {-delta} tokens shorter than the observation it re-expanded "
                f"({tail_obs_len} tokens), so the change cannot be confined to it -- the mask "
                "can no longer be placed"
            )
        keep = len(response_mask) - tail_obs_len
        response_mask = response_mask[:keep] + [0] * adjusted
        if response_logprobs:
            response_logprobs = response_logprobs[:keep] + [0.0] * adjusted
        tail_obs_len = adjusted

    if prefix_len is None:
        # First adoption fixes the prefix; it is in engine form from here on.
        prefix_len = len(engine_ids) - len(response_mask)

    if len(engine_ids) - len(response_mask) != prefix_len:
        raise PromptLengthMismatch(
            f"after adopting the engine prompt the mask no longer lines up: "
            f"{len(engine_ids)} tokens minus {len(response_mask)} masked is not the "
            f"{prefix_len}-token prefix. The change was not confined to the trailing "
            "observation, so some earlier region was re-expanded differently."
        )

    return response_mask, response_logprobs, tail_obs_len, prefix_len
