"""Webarena-specific agent loop subclass.

Overrides `GymAgentLoop._handle_generating_state` to compress historical
user messages (HTML obs → placeholder) before each LLM call, mirroring
WebAgent-R1's `WebRLChatPromptConstructor`. Without this, multi-turn
training accumulates full HTML each turn and hits the 32K context limit
within ~4-5 turns on heavy shopping/gitlab pages.

Wired in via `vagen/envs/webarena/configs/agent.yaml`, which webarena
training scripts (`examples/train/webarena/*.sh`) pass as
`agent_loop_config_path`. The YAML keeps the existing `gym_agent` name
(matching the hardcoded value in `vagen/gym_agent_dataset.py`) so the
dataset side needs no change.
"""

from typing import Any, Dict

from verl.utils.profiler import simple_timer

from vagen.agent_loop.gym_agent_loop import (
    AgentData,
    AgentState,
    GymAgentLoop,
    _flatten_text_only_content,
    logger,
)

from .utils.prompt import compress_history


class WebArenaGymAgentLoop(GymAgentLoop):
    """Drop-in replacement for `GymAgentLoop` for webarena training.

    Identical behavior except: in text-only multi-turn flows with 2+
    user turns, re-tokenizes `prompt_ids` from the compressed message
    history for the current generate() call. The accumulator
    `agent_data.prompt_ids` keeps the uncompressed sequence (used by
    verl for gradient updates).
    """

    async def _handle_generating_state(
        self, agent_data: AgentData, sampling_params: Dict[str, Any]
    ) -> AgentState:
        sampling_params_for_turn = sampling_params.copy()
        max_new_tokens = sampling_params_for_turn.get("max_new_tokens", None) or agent_data.response_limit
        max_new_tokens = min(max_new_tokens, agent_data.response_limit)
        sampling_params_for_turn["max_new_tokens"] = max_new_tokens

        # stop_token_ids fix (same as parent's behavior after the verl-level
        # bug fix lands; kept here so this subclass works whether the base
        # has it or not).
        if "stop_token_ids" not in sampling_params_for_turn:
            eos_id = getattr(self.tokenizer, "eos_token_id", None)
            if eos_id is not None:
                stop_ids = [eos_id] if isinstance(eos_id, int) else list(eos_id)
                sampling_params_for_turn["stop_token_ids"] = stop_ids

        # === webarena-specific: re-tokenize from compressed messages ===
        # Only kicks in for text-only flows once we have 2+ user turns
        # (i.e., turn 1 onward).
        prompt_ids_for_call = agent_data.prompt_ids
        if (
            self.processor is None
            and sum(1 for m in agent_data.messages if m.get("role") == "user") >= 2
        ):
            compressed = compress_history(agent_data.messages)
            flat = [_flatten_text_only_content(m) for m in compressed]
            prompt_ids_for_call = await self.loop.run_in_executor(
                None,
                lambda: self.tokenizer.apply_chat_template(
                    flat,
                    add_generation_prompt=True,
                    tokenize=True,
                    return_dict=False,
                    **self.apply_chat_template_kwargs,
                ),
            )

        with simple_timer("generate_sequences", agent_data.metrics):
            output = await self.server_manager.generate(
                request_id=agent_data.request_id,
                prompt_ids=prompt_ids_for_call,
                sampling_params=sampling_params_for_turn,
                image_data=agent_data.image_data,
            )

        agent_data.response_ids = output.token_ids
        if len(output.token_ids) > agent_data.response_limit:
            logger.warning(
                f"In env:{agent_data.env_name}, generated response length "
                f"{len(output.token_ids)} exceeds per-turn response_limit "
                f"{agent_data.response_limit}"
            )
        agent_data.prompt_ids += agent_data.response_ids
        agent_data.response_mask += [1] * len(agent_data.response_ids)
        if output.log_probs:
            agent_data.response_logprobs += output.log_probs

        assistant_message = await self.loop.run_in_executor(
            None,
            lambda: self.tokenizer.decode(agent_data.response_ids, skip_special_tokens=True),
        )
        agent_data.last_assistant_text = assistant_message
        agent_data.messages.append({"role": "assistant", "content": assistant_message})
        return AgentState.INTERACTING
