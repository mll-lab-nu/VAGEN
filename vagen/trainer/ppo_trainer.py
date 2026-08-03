"""The concrete trainer VAGEN launches.

``SeparateRayPPOTrainer`` is the base rather than ``RayPPOTrainer`` or
``main_ppo_sync``'s ``PPOTrainer`` because it is the only one of the three that exposes
``_fit_*`` hooks -- and it is also what the async trainers inherit, so the same mixin
carries over to ``OneStepOffRayTrainer`` and ``FullyAsyncTrainer``.
"""

from __future__ import annotations

from verl.experimental.separation.ray_trainer import SeparateRayPPOTrainer
from verl.single_controller.ray.base import RayClassWithInitArgs
from verl.trainer.ppo.ray_trainer import Role

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

    def _create_actor_rollout_classes(self):
        """Place actor and rollout on one pool (hybrid engine).

        The only abstract method on ``SeparateRayPPOTrainer``. It is abstract because
        "separation" is precisely about *where* actor and rollout go, and each async
        strategy answers that differently -- ``OneStepOffRayTrainer`` registers a bare
        ``Role.Actor`` on its own pool. Everything else in the base is already written
        for the colocated case: ``_init_models`` looks up ``Role.ActorRollout``.
        """
        if not self.hybrid_engine:
            raise NotImplementedError(
                "VagenPPOTrainer places actor and rollout together; for separated "
                "placement use one of verl's async trainers as the base instead"
            )
        role = Role.ActorRollout
        if role not in self.role_worker_mapping:
            # `_init_models` indexes all_wg by str(Role.ActorRollout), so a mapping
            # built around ActorRolloutRef ('actor_rollout_ref') would surface as a
            # bare KeyError several steps later.
            raise NotImplementedError(
                f"expected {role} in role_worker_mapping, found {sorted(map(str, self.role_worker_mapping))}; "
                "a fused reference policy is not supported on this path yet"
            )

        resource_pool = self.resource_pool_manager.get_resource_pool(role)
        self.resource_pool_to_cls[resource_pool][str(role)] = RayClassWithInitArgs(
            cls=self.role_worker_mapping[role],
            config=self.config.actor_rollout_ref,
            distillation_config=self.config.get("distillation"),
            role=str(role),
        )

    def _init_async_rollout_manager(self):
        """Stand up the LLM servers and the agent loop manager.

        The base leaves this as ``pass`` -- another placement decision, since a
        separated trainer points its servers at a different pool. Its ``init_workers``
        then reads ``self.llm_server_manager`` to build the checkpoint manager, so a
        colocated subclass has to set that here or fail with a bare AttributeError.

        ``agent_loop_manager_class`` is honoured the same way ``RayPPOTrainer`` does,
        which is how the no-concat runs select ``MultiOutputAgentLoopManager`` without
        an entrypoint fork.
        """
        from verl.utils.import_utils import load_class_from_fqn
        from verl.workers.rollout.llm_server import LLMServerManager

        rollout_cfg = self.config.actor_rollout_ref.rollout
        manager_class_fqn = rollout_cfg.get("agent", {}).get("agent_loop_manager_class")
        if manager_class_fqn:
            AgentLoopManager = load_class_from_fqn(manager_class_fqn, "AgentLoopManager")
        else:
            from verl.experimental.agent_loop import AgentLoopManager

        self.llm_server_manager = LLMServerManager.create(
            config=self.config,
            worker_group=self.actor_rollout_wg,
            rollout_resource_pool=self.resource_pool_manager.get_resource_pool(Role.ActorRollout),
        )

        # Streaming reward computation needs either no reward model, or one on its own
        # pool; VAGEN scores in the env, so the first branch is the usual one.
        enable_agent_reward_loop = not self.use_rm or self.config.reward.reward_model.enable_resource_pool
        reward_loop_worker_handles = (
            self.reward_loop_manager.reward_loop_workers if enable_agent_reward_loop else None
        )

        self.async_rollout_manager = AgentLoopManager.create(
            config=self.config,
            llm_client=self.llm_server_manager.get_client(),
            teacher_client=self.teacher_model_manager.get_client() if self.use_teacher_policy else None,
            reward_loop_worker_handles=reward_loop_worker_handles,
        )
