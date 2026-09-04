import importlib

from omegaconf import OmegaConf


def test_preinitialized_ray_passes_runtime_env_to_task_runner(monkeypatch):
    main_module = importlib.import_module("vagen.training.main")
    monkeypatch.setenv("VERL_FULL_DETERMINISM", "0")
    monkeypatch.setenv("VLLM_BATCH_INVARIANT", "0")
    monkeypatch.setenv("PYTHONHASHSEED", "0")
    config = OmegaConf.create(
        {
            "actor_rollout_ref": {
                "rollout": {"full_determinism": True, "seed": 42}
            },
            "reward": {
                "reward_model": {
                    "enable": False,
                    "rollout": {"full_determinism": False},
                }
            },
            "ray_kwargs": {
                "ray_init": {"runtime_env": {"env_vars": {"CUSTOM_ENV": "yes"}}}
            },
            "transfer_queue": {"enable": False},
            "global_profiler": {"tool": None, "steps": None},
        }
    )

    monkeypatch.setattr(main_module.ray, "is_initialized", lambda: True)
    monkeypatch.setattr(
        main_module.ray,
        "init",
        lambda **_kwargs: (_ for _ in ()).throw(AssertionError("ray.init called")),
    )
    monkeypatch.setattr(main_module.ray, "get", lambda value: value)
    monkeypatch.setattr(
        main_module,
        "get_ppo_ray_runtime_env",
        lambda _config: {"env_vars": {"VERL_FULL_DETERMINISM": "1"}},
    )

    class _Run:
        def remote(self, received_config):
            assert received_config is config
            return "done"

    class _TaskRunnerClass:
        run = _Run()

        def __init__(self):
            self.options_kwargs = None

        def options(self, **kwargs):
            self.options_kwargs = kwargs
            return self

        def remote(self):
            return self

    task_runner = _TaskRunnerClass()
    main_module.run_ppo(config, task_runner_class=task_runner)

    assert task_runner.options_kwargs["runtime_env"]["env_vars"] == {
        "VERL_FULL_DETERMINISM": "1",
        "CUSTOM_ENV": "yes",
    }
