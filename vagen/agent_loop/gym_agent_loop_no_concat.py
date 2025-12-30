# Copyright 2025 Bytedance Ltd.
# Licensed under the Apache License, Version 2.0

import asyncio
import logging
import os
import re
from enum import Enum
from typing import Any, Dict, List, Optional
from uuid import uuid4

from PIL import Image
from .agent_loop_no_concat import AgentLoopBase, AgentLoopOutput, register
from verl.utils.profiler import simple_timer
from verl.utils.rollout_trace import rollout_trace_op
from ..envs.gym_image_env import GymImageEnv
from omegaconf import OmegaConf
import traceback
import importlib
logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))
from .gym_agent_loop import convert_obs_to_content, extract_success, _flatten_text_only_content, _normalize_images

class AgentState(Enum):
    PENDING = "pending"
    GENERATING = "generating"
    INTERACTING = "interacting"
    TERMINATED = "terminated"


class AgentData:
    """Container for all mutable trajectory state."""
    def __init__(
        self,
        metrics: Dict[str, Any],
        request_id: str,
        env: GymImageEnv,
        response_limit: int,
        env_name: str,
        sys_msg: Optional[Dict[str, Any]] = None,
        sys_images: Optional[List[Image.Image]] = None,
        cur_msg: Optional[Dict[str, Any]] = None,
        cur_images: Optional[List[Image.Image]] = None,
        group_idx: int = 0,
        traj_idx: int = 0,
    ):
        self.sys_msg: Optional[Dict[str, Any]] = sys_msg
        self.sys_images: Optional[List[Image.Image]] = sys_images
        
        self.cur_msg: Optional[Dict[str, Any]] = cur_msg
        self.cur_images: Optional[List[Image.Image]] = cur_images
        
        self.metrics = metrics
        self.request_id = request_id
        self.env = env
        self.response_limit = response_limit
        self.env_name = env_name
        self.group_idx = group_idx
        self.traj_idx = traj_idx
        # Token buffers
        self.turn_prompt_ids: Optional[List[int]] = None
        self.turn_response_ids: Optional[List[int]] = None
        self.turn_response_mask: Optional[List[int]] = None
        self.turn_response_logprobs: Optional[List[int]] = None

        # Env stats
        self.env_turns: int = 0


        # Cached assistant text to step env
        self.last_assistant_text: Optional[str] = None
        self.outputs: List[AgentLoopOutput] = []

# -------------------- Gym Agent Loop --------------------

class GymAgentLoop(AgentLoopBase):
    @classmethod
    def init_class(cls, config, tokenizer, processor, **kwargs):
        if cls._class_initialized:
            return
        cls._class_initialized = True
        print("Performing class-level GymAgentLoop initialization")

        cls.tokenizer = tokenizer
        cls.processor = processor
        cls.multi_turn_cfg = config.actor_rollout_ref.rollout.multi_turn
        
        cls.env_registry = {}

        for k, v in config.env_registry.items():
            module_path, class_name = v.rsplit(".", 1)
            module = importlib.import_module(module_path)
            cls.env_registry[k] = getattr(module, class_name)
            
        cls.apply_chat_template_kwargs = config.data.get("apply_chat_template_kwargs", {})
        cls.prompt_length = config.actor_rollout_ref.rollout.prompt_length
        cls.response_length = config.actor_rollout_ref.rollout.response_length
        cls.system_prompt = tokenizer.apply_chat_template(
            [{}], add_generation_prompt=False, tokenize=True, **cls.apply_chat_template_kwargs
        )

    @rollout_trace_op
    async def run(self, sampling_params: Dict[str, Any], **kwargs) -> AgentLoopOutput:
        metrics: Dict[str, Any] = {}
        request_id = uuid4().hex

        # Build env
        env_cls = self.env_registry[kwargs["env_name"]]
        env_config = kwargs["config"]
        seed = kwargs["seed"]
        self.env_max_turns = kwargs.get("max_turns", None)
        env: GymImageEnv = env_cls(env_config=env_config)

        # Bootstrap: reset -> system_prompt (message order: system, then initial user)
        init_obs, info = await env.reset(seed=seed)
        sys_obs = await env.system_prompt()

       
        
        sys_msg={"role": "system", "content": convert_obs_to_content(sys_obs, **kwargs)}
        sys_images=_normalize_images(sys_obs.get("multi_modal_input", {}).get("<image>", []) or [])
        
        cur_msg={"role": "user", "content": convert_obs_to_content(init_obs, **kwargs)}
        cur_images=_normalize_images(init_obs.get("multi_modal_input", {}).get("<image>", []) or [])

        per_turn_response_limit = int(kwargs.get("response_length_per_turn") or self.response_length)
        per_turn_response_limit = min(per_turn_response_limit, self.response_length)
        if per_turn_response_limit <= 0:
            per_turn_response_limit = 1

        agent_data = AgentData(
            sys_msg=sys_msg,
            sys_images=sys_images,
            cur_msg=cur_msg,
            cur_images=cur_images,
            metrics=metrics,
            request_id=request_id,
            env=env,
            response_limit=per_turn_response_limit,
            env_name=kwargs["env_name"],
            group_idx=kwargs["group_idx"],
            traj_idx=kwargs["traj_idx"],
        )

        # State machine: always GENERATE -> INTERACT, and decide termination inside INTERACT
        state = AgentState.PENDING
        while state != AgentState.TERMINATED:
            if state == AgentState.PENDING:
                state = await self._handle_pending_state(agent_data, sampling_params)
            elif state == AgentState.GENERATING:
                state = await self._handle_generating_state(agent_data, sampling_params)
            elif state == AgentState.INTERACTING:
                state = await self._handle_env_state(agent_data, **kwargs)
            else:
                logger.error(f"Invalid state: {state}")
                state = AgentState.TERMINATED

        # Close env after loop
        await env.close()
        return agent_data.outputs

    async def _handle_pending_state(self, agent_data: AgentData, sampling_params: Dict[str, Any]) -> AgentState:
        """Encode initial (system + first user) messages into prompt_ids."""
        image_data = agent_data.sys_images + agent_data.cur_images
        if self.processor is not None:
            raw_prompt = await self.loop.run_in_executor(
                None,
                lambda: self.processor.apply_chat_template(
                    [agent_data.sys_msg, agent_data.cur_msg],
                    add_generation_prompt=True,
                    tokenize=False,
                    **self.apply_chat_template_kwargs,
                ),
            )
            model_inputs = self.processor(text=[raw_prompt], images=image_data, return_tensors="pt")
            agent_data.turn_prompt_ids = model_inputs.pop("input_ids").squeeze(0).tolist()
        else:
            if image_data:
                raise ValueError("Environment returned images but `processor` is None.")
            try:
                agent_data.turn_prompt_ids = await self.loop.run_in_executor(
                    None,
                    lambda: self.tokenizer.apply_chat_template(
                        [agent_data.sys_msg, agent_data.cur_msg],
                        add_generation_prompt=True,
                        tokenize=True,
                        **self.apply_chat_template_kwargs,
                    ),
                )
            except TypeError as e:
                logger.warning(f"TypeError Warning in apply_chat_template in AgentLoop: {e}, switching to flattened text-only content.")
                # Fallback for text-only tokenizer
                flat_messages = [_flatten_text_only_content([agent_data.sys_msg, agent_data.cur_msg])]
                agent_data.turn_prompt_ids = await self.loop.run_in_executor(
                    None,
                    lambda: self.tokenizer.apply_chat_template(
                        flat_messages,
                        add_generation_prompt=True,
                        tokenize=True,
                        **self.apply_chat_template_kwargs,
                    ),
                )
        
        if len(agent_data.turn_prompt_ids)>self.prompt_length:
            logger.warning(f"In env:{agent_data.env_name}, initial prompt length {len(agent_data.turn_prompt_ids)} exceeds prompt_length {self.prompt_length}")
        return AgentState.GENERATING

    
    async def _handle_generating_state(
        self, agent_data: AgentData, sampling_params: Dict[str, Any]
    ) -> AgentState:
        """Generate assistant output and mark generated tokens with mask=1."""
        sampling_params_for_turn = sampling_params.copy()
        max_new_tokens=sampling_params_for_turn.get("max_new_tokens", None) or agent_data.response_limit
        max_new_tokens = min(max_new_tokens, agent_data.response_limit)
        sampling_params_for_turn["max_new_tokens"] = max_new_tokens
        image_data = agent_data.sys_images + agent_data.cur_images

        with simple_timer("generate_sequences", agent_data.metrics):
            output = await self.server_manager.generate(
                request_id = agent_data.request_id,
                prompt_ids = agent_data.turn_prompt_ids,
                sampling_params = sampling_params_for_turn,
                image_data = image_data,
            )


        agent_data.turn_response_ids = output.token_ids
        agent_data.turn_prompt_ids += agent_data.turn_response_ids
        agent_data.turn_response_mask = [1] * len(agent_data.turn_response_ids)
        if output.log_probs:
            agent_data.turn_response_logprobs = output.log_probs

        # Cache assistant text and add assistant message (text-only)
        assistant_message = await self.loop.run_in_executor(
            None, lambda: self.tokenizer.decode(agent_data.turn_response_ids, skip_special_tokens=True)
        )
        agent_data.last_assistant_text = assistant_message
        return AgentState.INTERACTING

    async def _handle_env_state(self, agent_data: AgentData, **kwargs) -> AgentState:
        """
        Step the environment with last assistant action; always collect reward first.
        If terminal (done/success/turn-limit/token-limit), stop WITHOUT appending user suffix,
        so the episode ends on an assistant turn.
        """
        action_str = agent_data.last_assistant_text or ""
        try:
            obs, reward, done, info = await agent_data.env.step(action_str)
            # traceback
        except Exception as exc:
            logger.error(
                "Environment step failed in '%s' with action %r: %s",
                agent_data.env_name,
                action_str,
                exc,
            )
            logger.error("Environment traceback:\n%s", traceback.format_exc())
            obs, reward, done, info = {"obs_str":"Environment Error"}, 0.0, True, {"traj_success": False}

        traj_success = extract_success(info)
        agent_data.env_turns += 1
        last_turn=False
        
        
        
        if done:
            last_turn = True

        if self.env_max_turns is not None and agent_data.env_turns >= int(self.env_max_turns):
            last_turn = True

        
        if len(agent_data.turn_response_mask) >= self.response_length:
            last_turn = True

        turn_images=agent_data.sys_images+agent_data.cur_images
        output = AgentLoopOutput(
            prompt_ids=agent_data.turn_prompt_ids[-self.prompt_length:],
            response_ids=agent_data.turn_response_ids[: self.response_length],
            response_mask=agent_data.turn_response_mask[: self.response_length],
            multi_modal_data={"image":  turn_images} if turn_images else {},
            response_logprobs=(
                agent_data.turn_response_logprobs[: self.response_length] if agent_data.turn_response_logprobs else None
            ),
            reward_score=float(reward),
            num_turns=1,
            metrics=agent_data.metrics,
            extra_fields={"reward_extra_info": {
                "traj_success": traj_success,
                },
                "image": turn_images,
                "last_turn": last_turn,
                "group_idx": agent_data.group_idx,
                "traj_idx": agent_data.traj_idx,
                "turn_idx": agent_data.env_turns,
                          
            },
        )
        agent_data.outputs.append(output)
        
        # update cur msg and images
        cur_msg={"role": "user", "content": convert_obs_to_content(obs, **kwargs)}
        cur_images=_normalize_images(obs.get("multi_modal_input", {}).get("<image>", []) or [])
        agent_data.cur_msg = cur_msg
        agent_data.cur_images = cur_images
        if last_turn:
            return AgentState.TERMINATED

        return AgentState.PENDING
