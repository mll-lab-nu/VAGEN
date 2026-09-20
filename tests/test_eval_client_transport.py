"""Both clients face the same hosted endpoints, so both need the same settings.

httpx keeps 20 connections alive by default, which serialises a run that asked
for hundreds at once, and the SDK retries twice, which is not enough against a
per-minute quota -- a quota is a queue, not a verdict, and giving up on it turns
a served request into a recorded failure.
"""
from __future__ import annotations

import pytest

from vagen.evaluation.backends.openai.openai import build_client_azure, build_client_openai

AZURE = {"azure_endpoint": "https://example.openai.azure.com", "azure_api_key": "k"}


def _limits(client):
    pool = client._client._transport._pool
    return pool._max_connections, pool._max_keepalive_connections


@pytest.mark.parametrize("build,cfg", [
    (build_client_openai, {"api_key": "k"}),
    (build_client_azure, AZURE),
])
def test_pool_is_sized_for_the_concurrency(build, cfg):
    client = build(cfg)
    assert _limits(client) == (1024, 1024)


@pytest.mark.parametrize("build,cfg", [
    (build_client_openai, {"api_key": "k"}),
    (build_client_azure, AZURE),
])
def test_retries_outlast_a_per_minute_quota(build, cfg):
    assert build(cfg).max_retries == 8


@pytest.mark.parametrize("build,cfg", [
    (build_client_openai, {"api_key": "k"}),
    (build_client_azure, AZURE),
])
def test_config_overrides_the_defaults(build, cfg):
    client = build({**cfg, "max_connections": 32, "max_retries": 1})
    assert _limits(client) == (32, 32)
    assert client.max_retries == 1


def test_env_overrides_the_defaults(monkeypatch):
    monkeypatch.setenv("LLM_MAX_CONNECTIONS", "64")
    monkeypatch.setenv("LLM_MAX_RETRIES", "3")
    client = build_client_openai({"api_key": "k"})
    assert _limits(client) == (64, 64)
    assert client.max_retries == 3


def test_base_url_is_still_honoured():
    client = build_client_openai({"api_key": "k", "base_url": "https://example.test/v1"})
    assert str(client.base_url).startswith("https://example.test/v1")
