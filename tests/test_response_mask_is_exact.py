"""``response_mask`` must mark the model's tokens and nothing else.

It is the single most load-bearing tensor in the whole pipeline, and every consumer takes
it on faith:

* the actor's policy gradient, entropy bonus and KL loss are all aggregated over it
* ``batch_num_tokens``, the denominator of every one of those, is ``loss_mask.sum()`` and
  ``loss_mask`` *is* this tensor
* the critic's value loss uses it (intersected with ``value_mask`` when one exists)
* every advantage estimator gathers only its 1 positions

So a token wrongly marked 1 is trained as an action the model never chose, and a token
wrongly marked 0 silently leaves the objective. Neither raises.

The fake processor emits one token per character, so a mask can be read straight back
against the text -- which is the only way to check "exactly the model's output" rather
than "some plausible number of ones".
"""

from __future__ import annotations

import types

import pytest

from vagen.training.agent_loop.verl_client import VerlClient


class Proc:
    """One token per character. Images add a placeholder run."""

    def apply_chat_template(self, messages, add_generation_prompt=True, tokenize=False, **kw):
        def rendered(message):
            content = message["content"]
            if isinstance(content, str):
                return content
            return "".join(str(part.get("text", "<I>")) for part in content)

        return "|" + "".join(rendered(m) for m in messages)

    def __call__(self, text=None, images=None, return_tensors=None, **kw):
        ids = [ord(c) for c in text[0]] + [7] * (len(images or []) * 3)
        return {"input_ids": types.SimpleNamespace(squeeze=lambda _: types.SimpleNamespace(tolist=lambda: ids))}


class Tok:
    def decode(self, ids, skip_special_tokens=True):
        return f"text{len(ids)}"


class Server:
    """Answers with the characters of `replies[i]`, so a generation is readable."""

    def __init__(self, replies):
        self.replies = list(replies)
        self.n = 0

    async def generate(self, request_id, prompt_ids, sampling_params, image_data=None, **kw):
        reply = self.replies[self.n]
        self.n += 1
        ids = [ord(c) for c in reply]
        return types.SimpleNamespace(token_ids=ids, log_probs=[0.0] * len(ids), extra_fields={})


def _msg(text):
    return {"role": "user", "content": [{"type": "text", "text": text}]}


def _spell(row, want):
    """The characters at mask==`want`, in order."""
    return "".join(chr(t) for t, m in zip(row.response_ids, row.response_mask) if m == want)


# ------------------------------------------------------------------ one conversation


@pytest.mark.asyncio
async def test_the_mask_marks_the_generations_and_only_the_generations():
    """★ Three turns in one conversation (the concat layout). The model wrote AAA, BB and
    C; everything else in the response region is observation text the environment
    supplied. If an observation were marked 1 it would train as an action."""
    client = VerlClient(Server(["AAA", "BB", "C"]), Tok(), Proc(), model_adapter_name="qwen")

    r = await client.send([_msg("obs1")])
    await client.send([_msg("obs2")], r.conversation_id)
    await client.send([_msg("obs3")], r.conversation_id)

    row = client.rows()[0]
    assert _spell(row, 1) == "AAABBC", "the mask does not spell the model's output"
    assert "A" not in _spell(row, 0) and "B" not in _spell(row, 0) and "C" not in _spell(row, 0)
    assert "obs2" in _spell(row, 0) and "obs3" in _spell(row, 0), (
        "the later observations must be in the response region, at mask 0"
    )


@pytest.mark.asyncio
async def test_the_opening_observation_is_not_in_the_response_region_at_all():
    """It is prompt, not masked-out response: `prompt_len` is set at the first
    `add_response`, so everything before that is outside the trainable region."""
    client = VerlClient(Server(["AAA"]), Tok(), Proc(), model_adapter_name="qwen")
    await client.send([_msg("obs1")])

    row = client.rows()[0]
    assert "obs1" not in _spell(row, 0), "the opening observation leaked into the response"
    assert "obs1" in "".join(chr(t) for t in row.prompt_ids)


@pytest.mark.asyncio
async def test_every_vector_stays_the_same_length_as_the_mask():
    """The mask, the ids, the logprobs and the scores are sliced in parallel everywhere
    downstream; a length drift turns into a silent off-by-one in the advantage."""
    client = VerlClient(Server(["AAA", "BB"]), Tok(), Proc(), model_adapter_name="qwen")
    r = await client.send([_msg("obs1")])
    await client.send([_msg("obs2")], r.conversation_id)

    row = client.rows()[0]
    n = len(row.response_ids)
    assert len(row.response_mask) == n
    assert len(row.logprobs) == n
    assert len(row.scores) == n


# ------------------------------------------------------------------ the compact seam


@pytest.mark.asyncio
async def test_a_summary_is_masked_in_and_its_request_is_masked_out():
    """★ The compaction seam, in mask terms. The summary is a model emission and is an
    action; the request that provoked it is a user message and is not."""
    client = VerlClient(Server(["AAA", "SUMMARY"]), Tok(), Proc(), model_adapter_name="qwen")

    r = await client.send([_msg("obs1")])
    await client.send([_msg("Summarise")], r.conversation_id)

    row = client.rows()[0]
    assert _spell(row, 1) == "AAASUMMARY"
    assert "Summarise" in _spell(row, 0), "the summary request must not be an action"


@pytest.mark.asyncio
async def test_a_new_conversation_puts_the_summary_in_its_prompt_not_its_response():
    """The same text appears twice -- as an action in the closing conversation and as
    context in the next one. Only the first may carry gradient."""
    client = VerlClient(
        Server(["AAA", "SUMMARY", "BB"]), Tok(), Proc(), model_adapter_name="qwen"
    )

    r = await client.send([_msg("obs1")])
    await client.send([_msg("Summarise")], r.conversation_id)
    await client.send([_msg("Summary so far: SUMMARY. obs2")])       # no id: a new one

    first, second = client.rows()
    assert _spell(first, 1) == "AAASUMMARY"
    assert _spell(second, 1) == "BB", "the carried-over summary must not be re-trained"
    assert "SUMMARY" in "".join(chr(t) for t in second.prompt_ids)


# --------------------------------------------------- what depends on it being present


def test_verl_would_mark_every_observation_trainable_if_ours_went_missing():
    """★ Why the mask must always be published, not merely usually.

    verl's fallback is `attention_mask[:, -response_length:]` -- every non-padding token
    in the response region. Under concat that region holds all the interleaved
    observations, so the fallback marks them as actions. Both call sites are guarded by
    `if "response_mask" not in ...`, which is the only thing standing between us and that.
    """
    import inspect

    from verl.experimental.separation import ray_trainer
    from verl.trainer.ppo.ray_trainer import compute_response_mask

    assert "attention_mask[:, -response_length:]" in inspect.getsource(compute_response_mask)
    src = inspect.getsource(ray_trainer.SeparateRayPPOTrainer._fit_generate)
    assert 'if "response_mask" not in batch.batch.keys():' in src, (
        "verl would recompute the mask unconditionally and every observation would "
        "become a trainable action"
    )


def test_the_actor_denominator_is_the_same_mask():
    """`batch_num_tokens` is `loss_mask.sum()`, and `loss_mask` is this tensor -- which is
    why dropping a token from `response_mask` removes it from the numerator *and* the
    denominator, and why nothing applied later in the loss can do the same."""
    import inspect

    from verl.experimental.separation import ray_trainer
    from verl.workers.engine.fsdp import transformer_impl

    assert "loss_mask=response_masks" in inspect.getsource(ray_trainer)
    assert 'batch_num_tokens = data["loss_mask"].sum()' in inspect.getsource(transformer_impl)


def test_the_critic_uses_the_same_mask_narrowed_by_value_mask():
    """The critic's counterpart. `value_mask` only ever *narrows* `response_mask`; an
    estimator cannot use it to supervise a position the model did not emit."""
    import inspect

    from verl.workers.utils import losses

    src = inspect.getsource(losses.value_loss)
    assert 'response_mask = data["response_mask"].to(bool)' in src
    assert 'response_mask & data["value_mask"].to(bool)' in src, (
        "value_mask must intersect response_mask, not replace it"
    )
