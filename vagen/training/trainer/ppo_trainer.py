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

from vagen.training.trainer.mixin import VagenV0Mixin


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

    def _get_gen_batch(self, batch):
        """Move the placeholder prompt tensors out of the batch before generation.

        The dataset emits one-element ``input_ids`` / ``attention_mask`` /
        ``position_ids``: an agent loop builds the real prompt from environment
        observations, so there is nothing meaningful to put there, but a DataProto
        still needs tensors to carry a batch size.

        verl used to pop exactly these three for generation and no longer does, which
        leaves the placeholders in ``batch`` while ``gen_batch_output`` brings back the
        real ones -- and ``batch.union(gen_batch_output)`` asserts that shared keys
        hold the same tensor. Popping them restores the intended dataflow: placeholders
        go out with the gen batch (where the loop ignores them) and the generated
        tensors come back unopposed.
        """
        reward_keys = {"data_source", "reward_model", "extra_info", "uid"} & batch.non_tensor_batch.keys()
        placeholders = [k for k in ("input_ids", "attention_mask", "position_ids") if k in batch.batch]

        gen_batch = batch.pop(
            batch_keys=placeholders,
            non_tensor_batch_keys=list(set(batch.non_tensor_batch.keys()) - reward_keys),
        )
        # The agent loop scores in-flight, so it needs the reward-model keys too.
        gen_batch.non_tensor_batch.update(batch.non_tensor_batch)
        return gen_batch

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
