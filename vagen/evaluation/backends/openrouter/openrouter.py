from __future__ import annotations
import inspect
import json
import logging
import os

import httpx
from typing import Any, Dict, List, Optional, Tuple
from openai import AsyncOpenAI
from PIL import Image
from vagen.evaluation.backends._common.base import EvaluationBackend
from vagen.evaluation.backends._common.rendering import pil_to_dataurl_png, compile_text_images_for_order
from vagen.evaluation.backends._common.registry import register_adapter, register_client


logger = logging.getLogger(__name__)

#: ``_sdk_params`` before the SDK's signature has been read.
_UNREAD = object()

_OPENROUTER_URL = "https://openrouter.ai/api/v1"


class OutputBudgetExceeded(RuntimeError):
    """The endpoint stopped at ``max_tokens`` before writing any content.

    Raised rather than returned empty so the episode is recorded as an error and
    can be re-run, instead of being written as a failure that resume then treats
    as finished work. Carries the raw response: its usage is the evidence of where
    the budget went.
    """

    def __init__(self, raw_response: Any, message: str):
        super().__init__(message)
        self.raw_response = raw_response


class ProviderError(RuntimeError):
    """OpenRouter's report, inside a 200 response, that the upstream provider failed:
    no choices and a top-level ``error``, or a choice that ended with
    ``finish_reason: "error"`` and carries its own ``error``.

    ``code`` is the provider's status from that error, so the retry policy treats it as
    it would the same status over HTTP.
    """

    def __init__(self, raw_response: Any, error: Any, what: str):
        error = error if isinstance(error, dict) else {}
        code = error.get("code")
        if isinstance(code, str) and code.isdigit():
            code = int(code)
        super().__init__(f"{what}: {error.get('message') or raw_response}")
        self.raw_response = raw_response
        self.code = code if isinstance(code, int) else None


def _transport(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Connection pool and retry budget, shared by both clients.

    Two defaults are wrong for an evaluation that runs many episodes at once
    against a hosted endpoint.

    httpx keeps 20 connections alive, so asking for 256 concurrent requests still
    sent them a few at a time -- measured 6 requests a second at 32 concurrent
    where the latency implied 20. The pool is sized to the concurrency instead.

    The SDK retries twice, which is not enough against a per-minute quota. A
    quota is a queue, not a verdict: the request is fine and will be served
    shortly. Measured against one deployment, two retries left 88% of calls
    returning 429 and 27% of episodes recorded as failures with nothing wrong
    with them; eight brought that to 1.8% without touching concurrency.

    These retries sit under ThrottledAdapter's own (``max_retries`` in the backend
    config), so the two multiply. The SDK's count therefore has its own key,
    ``sdk_max_retries``, rather than sharing ``max_retries`` and squaring it.
    """
    pool = int(cfg.get("max_connections", os.getenv("LLM_MAX_CONNECTIONS", "1024")))
    return {
        "http_client": httpx.AsyncClient(
            limits=httpx.Limits(max_connections=pool, max_keepalive_connections=pool),
            timeout=httpx.Timeout(float(cfg.get("timeout", 600.0)), connect=20.0),
        ),
        "max_retries": int(cfg.get("sdk_max_retries", os.getenv("LLM_MAX_RETRIES", "8"))),
    }


@register_client("openrouter")
def build_client_openrouter(cfg: Dict[str, Any]) -> AsyncOpenAI:
    """OpenRouter by default; ``base_url`` points it at any server that speaks the same
    format -- a local vLLM/SGLang server, or a gateway such as a LiteLLM proxy.

    OPENROUTER_API_KEY is read only for OpenRouter itself, so the key is not sent to
    whatever else base_url names; a local server needs none and gets "EMPTY".
    """
    base_url = cfg.get("base_url") or _OPENROUTER_URL
    env_key = os.getenv("OPENROUTER_API_KEY") if base_url.startswith(_OPENROUTER_URL) else None
    return AsyncOpenAI(
        api_key=cfg.get("api_key") or env_key or "EMPTY",
        base_url=base_url,
        **_transport(cfg),
    )


def _raw_dump(resp: Any) -> Any:
    """JSON-safe copy of an SDK response object (pydantic model_dump), or its repr."""
    try:
        return resp.model_dump(mode="json")
    except Exception:  # noqa: BLE001
        try:
            return json.loads(resp.model_dump_json())
        except Exception:  # noqa: BLE001
            return {"repr": repr(resp)}


@register_adapter("openrouter")
class OpenRouterAdapter(EvaluationBackend):
    """The one built-in backend: OpenRouter's request format over Chat Completions.

    Messages use content parts with {"type": "text"} and {"type": "image_url"}, and
    chat_config is an OpenRouter request body minus ``model`` and ``messages``. A model
    served some other way is reached through a program that exposes it in this format.
    """

    def __init__(
        self,
        client,
        model: str,

    ):
        self.client = client
        self.model = model
        #: Raw response of the most recent call, read by ChatClient for raw_responses.json.
        self.last_raw: Any = None
        #: Argument names ``chat.completions.create`` takes; None when it takes **kwargs.
        self._sdk_params: Any = _UNREAD

    def _request_kwargs(self, chat_config: Dict[str, Any]) -> Dict[str, Any]:
        """chat_config as SDK arguments: unset values dropped, the rest of OpenRouter's
        parameters (``reasoning``, ``top_k``, ``provider``, ...) moved into ``extra_body``.

        The SDK takes only OpenAI's parameters as arguments and raises TypeError on any
        other; ``extra_body`` is merged into the JSON body unchanged, so the request is the
        one OpenRouter documents. A key also given inside ``extra_body`` keeps that value.
        """
        if self._sdk_params is _UNREAD:
            params = inspect.signature(self.client.chat.completions.create).parameters.values()
            self._sdk_params = (None if any(p.kind is p.VAR_KEYWORD for p in params)
                                else frozenset(p.name for p in params))
        kwargs: Dict[str, Any] = {}
        extra = dict(chat_config.get("extra_body") or {})
        for key, value in chat_config.items():
            if value is None or key == "extra_body":
                continue
            if self._sdk_params is None or key in self._sdk_params:
                kwargs[key] = value
            else:
                extra.setdefault(key, value)
        if extra:
            kwargs["extra_body"] = extra
        return kwargs

    def is_retryable_error(self, exc: BaseException) -> Optional[bool]:
        # The same budget would be exhausted the same way: a retry only pays for another
        # full max_tokens of hidden reasoning. Fail the call so the episode is recorded as
        # an error and re-run after the budget is raised.
        if isinstance(exc, OutputBudgetExceeded):
            return False
        return None

    def _segments_to_content(self, segs: List[Tuple[str, Any]]) -> List[Dict[str, Any]]:
        content: List[Dict[str, Any]] = []
        for kind, val in segs:
            if kind == "text":
                if str(val).strip():
                    content.append({"type": "text", "text": str(val)})
            else:
                content.append({"type": "image_url", "image_url": {"url": pil_to_dataurl_png(val)}})
        return content

    def format_system(self, text: str, images: List[Image.Image]) -> Dict[str, Any]:
        segs = compile_text_images_for_order(text, images)
        return {"role": "system", "content": self._segments_to_content(segs)}

    def format_user_turn(self, text: str, images: List[Image.Image]) -> Dict[str, Any]:
        segs = compile_text_images_for_order(text, images)
        return {"role": "user", "content": self._segments_to_content(segs)}

    async def acompletion(self, messages: List[Dict[str, Any]], **chat_config: Any) -> str:
        resp = await self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            **self._request_kwargs(chat_config),
        )
        # The whole response, verbatim, for the rollout dump (raw_responses.json). Reasoning
        # models return their thinking OUTSIDE ``message.content`` (OpenRouter: ``message.reasoning``
        # / ``reasoning_details``); returning only the content threw it away.
        self.last_raw = _raw_dump(resp)
        raw = self.last_raw if isinstance(self.last_raw, dict) else {}
        if not resp.choices:
            raise ProviderError(self.last_raw, raw.get("error"), f"{self.model} returned no choices")
        choice = resp.choices[0]
        # Raised even when some content came first: it is the provider's failure, and a
        # reply cut off by it is not one the model finished.
        if getattr(choice, "finish_reason", None) == "error":
            raw_choice = (raw.get("choices") or [{}])[0]
            raise ProviderError(self.last_raw, raw_choice.get("error"),
                                f"{self.model} stopped with finish_reason=error")
        content = choice.message.content or ""
        # `finish_reason` carries the only signal that separates "the model declined"
        # from "the model was cut off". A reasoning model spends its output budget on
        # hidden thinking before it writes anything, so a budget that is merely too
        # small comes back as `length` with no content -- indistinguishable, once the
        # reason is dropped, from a refusal or a content filter. A harness then re-asks
        # against the same budget, which cannot help, and scores the episode as a
        # failure. Measured on one reasoning model at high effort, that silently took
        # out 29% of its rows.
        reason = getattr(choice, "finish_reason", None)
        if reason == "length" and not content.strip():
            raise OutputBudgetExceeded(
                self.last_raw,
                f"{self.model} stopped at its output budget with no content "
                f"(finish_reason=length). Raise max_tokens, or "
                f"response_length_per_turn if a harness sets the per-call limit: "
                f"hidden reasoning is billed against it."
            )
        if reason == "length":
            logger.warning(
                "%s was truncated at its output budget after %d characters; "
                "keeping the partial reply.", self.model, len(content))
        return content
