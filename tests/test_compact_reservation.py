"""Response-region arithmetic used by harness implementations."""

from vagen.harness import CompactHarness
from vagen.harness._common import BaseHarness


def test_generation_limit_charges_pending_observation_and_reserve():
    assert BaseHarness.generation_limit(1000, 1, 400, pending=100, reserve=200) == 300


def test_generation_limit_reports_no_room_below_floor():
    assert BaseHarness.generation_limit(1000, 64, 900, pending=50, reserve=20) == 0


def test_evaluation_can_disable_response_region_accounting():
    assert BaseHarness.generation_limit(None, 64, 999999, pending=999999) is None


def test_compact_reserves_summary_and_request():
    harness = CompactHarness(summary_budget=100, summary_request_len=37)
    assert harness.reserve == 137


def test_summary_budget_is_forwarded_as_its_own_generation_limit():
    harness = CompactHarness(summary_budget=25)
    assert harness.sampling(harness.summary_budget) == {
        "sampling_params": {"max_new_tokens": 25}
    }
