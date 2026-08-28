"""Turn an agent's free-form environment description into structured items.

The agent says "the box is just below me and the target is two to the left"; scoring that
needs it as a list of ``{"object_id", "vertical_relation", "horizontal_relation"}``. A
small instruct model does the conversion, reached over an OpenAI-compatible endpoint --
a locally hosted sglang or vLLM server, or a provider.

Deliberately thin. The judge is a parser, not part of the training loop: it holds no
state, logs nothing, and a failure returns ``None`` rather than raising, because losing
one turn's process reward is not worth losing the rollout.
"""

from __future__ import annotations

import asyncio
import json
import re
from dataclasses import dataclass
from typing import Any, Optional

_JSON_ARRAY = re.compile(r"\[.*\]", re.DOTALL)


def parse_items(text: str) -> Optional[list[dict]]:
    """The first JSON array in the reply, or None if there is not one.

    Tolerant on purpose: small models wrap output in prose or fences, and rejecting that
    would throw away usable answers.
    """
    if not text:
        return None
    match = _JSON_ARRAY.search(text)
    if not match:
        return None
    try:
        items = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    return [i for i in items if isinstance(i, dict)] if isinstance(items, list) else None


@dataclass
class StructuredJudge:
    """Batched calls to an OpenAI-compatible chat endpoint."""

    base_url: str
    model: str
    api_key: str = "EMPTY"
    temperature: float = 0.0
    max_tokens: int = 512
    concurrency: int = 32
    timeout: float = 60.0

    def __post_init__(self):
        self._client = None
        self._gate = asyncio.Semaphore(self.concurrency)

    def _ensure_client(self):
        if self._client is None:
            from openai import AsyncOpenAI

            self._client = AsyncOpenAI(base_url=self.base_url, api_key=self.api_key, timeout=self.timeout)
        return self._client

    async def _one(self, prompt: str) -> Optional[list[dict]]:
        async with self._gate:
            try:
                reply = await self._ensure_client().chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                )
                return parse_items(reply.choices[0].message.content)
            except Exception:  # noqa: BLE001 - a parser outage must not end the rollout
                return None

    async def parse_batch(self, prompts: list[str]) -> list[Optional[list[dict]]]:
        if not prompts:
            return []
        return list(await asyncio.gather(*(self._one(p) for p in prompts)))


_SHARED: dict[tuple, "StructuredJudge"] = {}


def shared_judge(base_url: str, model: str, **kwargs) -> "StructuredJudge":
    """One judge per endpoint, per worker process.

    ★ Not one per rollout. The semaphore bounds concurrency, and a fresh judge for every
    episode means the bound is per episode -- with hundreds in flight the endpoint sees
    hundreds of times the intended load and starts timing out, which shows up as the
    process reward quietly going to zero.
    """
    key = (base_url, model, tuple(sorted(kwargs.items())))
    if key not in _SHARED:
        _SHARED[key] = StructuredJudge(base_url=base_url, model=model, **kwargs)
    return _SHARED[key]


class NullJudge:
    """Stands in when no judge is configured: every description scores nothing.

    Makes the reward's absence explicit rather than leaving the wrapper half-wired.
    """

    async def parse_batch(self, prompts: list[str]) -> list[Optional[list[dict]]]:
        return [None] * len(prompts)
