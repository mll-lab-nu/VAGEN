from vagen.training.agent_loop.verl_client import (
    _backend_sampling_params,
    _collapse_placeholder_runs,
)


def test_sglang_sampling_options_are_translated_and_seeded():
    params = _backend_sampling_params(
        {"stop": ["</answer>"], "include_stop_str_in_output": True},
        backend="sglang",
        request_id="episode-a",
        call_id=2,
        base_seed=42,
        full_determinism=True,
    )

    assert params["no_stop_trim"] is True
    assert "include_stop_str_in_output" not in params
    assert isinstance(params["sampling_seed"], int)


def test_stable_sampling_seed_changes_by_call_not_process():
    kwargs = {
        "params": {},
        "backend": "sglang",
        "request_id": "episode-a",
        "base_seed": 42,
        "full_determinism": True,
    }
    first = _backend_sampling_params(call_id=0, **kwargs)["sampling_seed"]
    repeated = _backend_sampling_params(call_id=0, **kwargs)["sampling_seed"]
    second = _backend_sampling_params(call_id=1, **kwargs)["sampling_seed"]

    assert first == repeated
    assert first != second


def test_vllm_uses_its_seed_parameter():
    params = _backend_sampling_params(
        {},
        backend="vllm",
        request_id="episode-a",
        call_id=0,
        base_seed=42,
        full_determinism=True,
    )

    assert "sampling_seed" not in params
    assert isinstance(params["seed"], int)


def test_sglang_multimodal_prompt_collapses_each_expanded_image_run():
    assert _collapse_placeholder_runs(
        [10, 20, 20, 20, 11, 30, 20, 20, 31],
        {20},
    ) == [10, 20, 11, 30, 20, 31]


def test_placeholder_collapse_keeps_non_placeholder_repetitions():
    assert _collapse_placeholder_runs([1, 1, 2, 2, 3], {2}) == [1, 1, 2, 3]
