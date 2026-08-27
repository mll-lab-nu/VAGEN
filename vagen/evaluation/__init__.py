"""Standalone evaluation orchestration."""


def __getattr__(name):
    if name in {"run_eval_parallel", "NORMAL_FINISH_REASONS"}:
        from vagen.evaluation import runner

        return getattr(runner, name)
    raise AttributeError(name)


__all__ = ["NORMAL_FINISH_REASONS", "run_eval_parallel"]
