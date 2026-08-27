"""Which errors get another go.

`ThrottledAdapter` and its two predicates had no test at all, and they decide whether a
transient API failure costs one episode or none. Both bugs below were live.
"""

from __future__ import annotations

import pytest

from vagen.evaluation.backends._common.retry import _is_non_retryable, _is_retryable

CODES = (429, 500, 502, 503, 504)


class _Err(Exception):
    def __init__(self, msg, code=None):
        super().__init__(msg)
        if code is not None:
            self.status_code = code


@pytest.mark.parametrize("code", [500, 501, 502, 503, 504, 507, 520, 522, 524, 529])
def test_every_server_error_is_retried(code):
    """★ A whitelist of "codes we retry" is the wrong shape, and the one that was declared
    -- (429, 500, 502, 503, 504) -- omits 529, which is Anthropic's routine
    `overloaded_error`, and the Cloudflare 52x family. Treating those as permanent makes
    one HTTP call and scores the episode as a task failure."""
    assert _is_retryable(_Err("server side", code), CODES)


def test_a_rate_limit_is_retried_and_the_rest_of_4xx_is_not():
    assert _is_retryable(_Err("slow down", 429), CODES)
    for code in (400, 401, 403, 404, 422):
        assert not _is_retryable(_Err("your fault", code), CODES)


def test_the_status_code_outranks_the_wording():
    """★ The prose was matched for "invalid" / "auth" / "not found" BEFORE the code was
    considered, so a 5xx whose body happened to contain one of those words -- common in
    upstream-proxy wording -- was classified permanent and never retried."""
    assert _is_retryable(_Err("invalid response from upstream", 500), CODES)
    assert _is_retryable(_Err("authentication service unavailable", 503), CODES)
    assert not _is_retryable(_Err("invalid api key", 401), CODES)


def test_an_error_with_no_status_is_retried():
    """A connection reset carries no code. Retrying is the right default: the alternative
    is scoring a network blip as a task failure."""
    assert _is_retryable(_Err("connection reset by peer"), CODES)
    assert _is_retryable(_Err("timed out"), CODES)


def test_wording_still_decides_when_there_is_no_code():
    assert _is_non_retryable(_Err("authentication failed"))
    assert _is_non_retryable(_Err("api key not found"))
    assert not _is_non_retryable(_Err("upstream exploded"))
