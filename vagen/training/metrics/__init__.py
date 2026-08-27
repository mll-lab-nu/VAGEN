"""Training metric facade."""

from vagen.training.metrics._common import METRIC_REGISTRY, register_metric
from vagen.training.metrics.episode import episode_score, episode_turns, reward_variance

__all__ = [
    "METRIC_REGISTRY",
    "episode_score",
    "episode_turns",
    "register_metric",
    "reward_variance",
]
