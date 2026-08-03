"""The concrete trainer VAGEN launches.

``SeparateRayPPOTrainer`` is the base rather than ``RayPPOTrainer`` or
``main_ppo_sync``'s ``PPOTrainer`` because it is the only one of the three that exposes
``_fit_*`` hooks -- and it is also what the async trainers inherit, so the same mixin
carries over to ``OneStepOffRayTrainer`` and ``FullyAsyncTrainer``.
"""

from __future__ import annotations

from verl.experimental.separation.ray_trainer import SeparateRayPPOTrainer

from vagen.trainer.mixin import VagenV0Mixin


class VagenPPOTrainer(VagenV0Mixin, SeparateRayPPOTrainer):
    """verl's separated PPO trainer with VAGEN's hooks mixed in.

    Base order is load-bearing: the mixin has to precede the trainer so its ``_fit_*``
    overrides win the MRO. Reversed, they would never run and nothing would fail --
    ``value_mask`` would just quietly stop being written.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # After super(): reads self.config, which the base sets up.
        self._vagen_init()
