import os

from omegaconf import OmegaConf

from vagen.training.main import (
    _configure_backend_determinism,
    _propagate_determinism_env,
)


def _config(*, rollout=False, reward_model=False, seed=42, backend="vllm"):
    return OmegaConf.create(
        {
            "actor_rollout_ref": {
                "rollout": {
                    "full_determinism": rollout,
                    "seed": seed,
                    "name": backend,
                    "engine_kwargs": {},
                }
            },
            "reward": {
                "reward_model": {
                    "enable": reward_model,
                    "rollout": {"full_determinism": reward_model},
                }
            },
        }
    )


def test_rollout_determinism_is_exported_before_ray(monkeypatch):
    for name in ("VERL_FULL_DETERMINISM", "VLLM_BATCH_INVARIANT", "PYTHONHASHSEED"):
        monkeypatch.delenv(name, raising=False)

    _propagate_determinism_env(_config(rollout=True, seed=17))

    assert os.environ["VERL_FULL_DETERMINISM"] == "1"
    assert os.environ["VLLM_BATCH_INVARIANT"] == "1"
    assert os.environ["PYTHONHASHSEED"] == "17"


def test_disabled_determinism_does_not_mutate_environment(monkeypatch):
    for name in ("VERL_FULL_DETERMINISM", "VLLM_BATCH_INVARIANT", "PYTHONHASHSEED"):
        monkeypatch.delenv(name, raising=False)

    _propagate_determinism_env(_config())

    assert "VERL_FULL_DETERMINISM" not in os.environ
    assert "VLLM_BATCH_INVARIANT" not in os.environ
    assert "PYTHONHASHSEED" not in os.environ


def test_sglang_full_determinism_enables_deterministic_engine():
    config = _config(rollout=True, backend="sglang")

    _configure_backend_determinism(config)

    assert config.actor_rollout_ref.rollout.engine_kwargs.sglang.enable_deterministic_inference is True


def test_other_backends_do_not_receive_sglang_options():
    config = _config(rollout=True, backend="vllm")

    _configure_backend_determinism(config)

    assert "sglang" not in config.actor_rollout_ref.rollout.engine_kwargs
