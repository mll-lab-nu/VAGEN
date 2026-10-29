"""The client faces hosted endpoints, so it needs settings sized for them.

httpx keeps 20 connections alive by default, which serialises a run that asked
for hundreds at once, and the SDK retries twice, which is not enough against a
per-minute quota -- a quota is a queue, not a verdict, and giving up on it turns
a served request into a recorded failure.
"""
from __future__ import annotations

from vagen.evaluation.backends.openrouter.openrouter import build_client_openrouter


def _limits(client):
    pool = client._client._transport._pool
    return pool._max_connections, pool._max_keepalive_connections


def test_pool_is_sized_for_the_concurrency():
    client = build_client_openrouter({"api_key": "k"})
    assert _limits(client) == (1024, 1024)


def test_retries_outlast_a_per_minute_quota():
    assert build_client_openrouter({"api_key": "k"}).max_retries == 8


def test_config_overrides_the_defaults():
    client = build_client_openrouter({"api_key": "k", "max_connections": 32, "sdk_max_retries": 1})
    assert _limits(client) == (32, 32)
    assert client.max_retries == 1


def test_adapter_retry_count_does_not_reach_the_sdk():
    """`max_retries` is ThrottledAdapter's; sharing it with the SDK would square it."""
    assert build_client_openrouter({"api_key": "k", "max_retries": 1}).max_retries == 8


def test_env_overrides_the_defaults(monkeypatch):
    monkeypatch.setenv("LLM_MAX_CONNECTIONS", "64")
    monkeypatch.setenv("LLM_MAX_RETRIES", "3")
    client = build_client_openrouter({"api_key": "k"})
    assert _limits(client) == (64, 64)
    assert client.max_retries == 3


def test_base_url_is_still_honoured():
    client = build_client_openrouter({"api_key": "k", "base_url": "https://example.test/v1"})
    assert str(client.base_url).startswith("https://example.test/v1")


def test_the_openrouter_key_goes_only_to_openrouter(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-secret")
    hosted = build_client_openrouter({})
    assert str(hosted.base_url).startswith("https://openrouter.ai/api/v1")
    assert hosted.api_key == "sk-or-secret"
    assert build_client_openrouter({"base_url": "http://127.0.0.1:8311/v1"}).api_key == "EMPTY"
