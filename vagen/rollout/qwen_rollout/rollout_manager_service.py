
from typing import List, Union, Optional, Dict
import copy
from collections import defaultdict
import torch
import numpy as np
from transformers import PreTrainedTokenizer, ProcessorMixin
from dataclasses import dataclass, field
import PIL
import re

from verl import DataProto
from verl.utils.model import compute_position_id_with_mask
import verl.utils.torch_functional as verl_F
from verl.utils.dataset.rl_dataset import process_image, collate_fn
import vagen.env
from vagen.env import REGISTERED_ENV
from vagen.rollout.multimodal_utils import get_multimodal_handler, detect_model_type
from vagen.server.client import BatchEnvClient
from vagen.rollout.utils.mask_utils import compute_loss_mask
class QwenVLRolloutManagerService():
    def __init__(self,
                 actor_rollout_wg,
                 config,
                 tokenizer: PreTrainedTokenizer,
                 processor: Optional[ProcessorMixin] = None,
                 split="train",
                 ):
        self.split=split
        self.tokenizer = tokenizer
        self.processor = processor
        self.config = config
        self.actor_rollout_wg = actor_rollout_wg
        self.recorder= None # defaultdict(list) env_id:record
        self.envs = None # dict env_id:env_config_instance
        self.system_prompts = None # dict env_id:str
        self.env_states = None # dict
        self.batch_idx_to_env_id = None # dict
        self.env_client = BatchEnvClient(base_url=self.config.base_url,timeout=self.config.timeout,max_workers=self.config.max_workers)
        
        # Detect model type and get appropriate handler
        self.multimodal_handler = get_multimodal_handler(detect_model_type(processor))


    @torch.no_grad()
    def _handle_special_tokens(self, llm_raw_response: str, prep_for_loss_mask: bool) -> str:
        """
        1. Filter out special tokens: <image> and special tokens marking environment observation in the llm generated response
        2. prep_for_loss_mask: if true, add special tokens to the beginning and end of the response if compute_loss_mask is True
        """
        llm_raw_response = llm_raw_response.replace('<image>', '')
        if prep_for_loss_mask:
            # filtering special tokens for llm_raw_response, then adding them to the beginning and end of the response for loss mask computation
            sptk_b = self.config.special_token_for_loss_mask[0]
            sptk_e = self.config.special_token_for_loss_mask[1]
            llm_raw_response = llm_raw_response.replace(sptk_b, '')
            llm_raw_response = llm_raw_response.replace(sptk_e, '')
            llm_raw_response = sptk_b + llm_raw_response + sptk_e
        return llm_raw_response
    
    @torch.no_grad()
    def _handle_multi_modal_data(
            self, 
            prompt_template: str, 
            row_dict: Dict,
            image_data: List[PIL.Image.Image],
            do_embedding: bool = True,
        ) -> str:
        """Handle multi-modal data in the prompt template using model-specific handlers.
        
        - For do_embedding=False(vllm), replace <image> with model-specific tokens -> raw_prompt
        - For do_embedding=True, replace <image> with model-specific embedded tokens -> prompt_template
        """
        if self.multimodal_handler is None:
            raise ValueError("No multimodal handler available. Processor might be None.")
        
        return self.multimodal_handler(
            prompt_template=prompt_template,
            row_dict=row_dict, 
            image_data=image_data,
            processor=self.processor,
            do_embedding=do_embedding
        )
    
    @torch.no_grad()
    def _compute_loss_mask(self, input_ids, attention_mask):
        # Get token IDs for special tokens and pad token
        sptk_b = self.tokenizer.convert_tokens_to_ids(self.config.special_token_for_loss_mask[0])
        sptk_e = self.tokenizer.convert_tokens_to_ids(self.config.special_token_for_loss_mask[1])
        pad_token_id = self.tokenizer.pad_token_id

        return compute_loss_mask(input_ids, attention_mask, sptk_b, sptk_e, pad_token_id)
    
    @torch.no_grad()
    def reset(self, mini_batch:DataProto):
        """
        Reset environments based on provided configurations, reusing environments when possible.
        - For env with same config and env_name, reuse the same environment (reset)
        - For env with different config or env_name, close the old environment and create a new one
        - Reset the recorder
        
        Args:
            env_configs: List of environment configurations containing env_name, config, and seed
        
        Returns:
            Initial observations and info from all environments
        """
        # Step 1: Sort environments into buckets by env_name and config
        # Try to reuse environemnts with the same config and env_name
        env_configs = [
                mini_batch.non_tensor_batch['extra_info'][i]
                for i in range(len(mini_batch))
            ]
        env_buckets = defaultdict(set)
        
        if self.envs is None:
            self.envs = {} # This is now id:config_instance
            
        for env_id, env_config_instance in self.envs.items():
            env_config_id = env_config_instance.config_id()
            bucket_key = env_config_id
            env_buckets[bucket_key].add(env_id)
        
        # Step1. collect envs which need to be reset and new env configs
        ids2seeds_reset = {}
        configs_to_create=[]
        for i, cfg in enumerate(env_configs):
            # Create bucket key
            config_instance= REGISTERED_ENV[cfg["env_name"]]["config_cls"](**cfg["env_config"])
            env_config_id = config_instance.config_id()
            bucket_key = env_config_id
            
            # Check if we have an available environment with the same config
            if bucket_key in env_buckets and env_buckets[bucket_key]:
                old_env_id = env_buckets[bucket_key].pop()
                ids2seeds_reset[old_env_id] = cfg["seed"]
            else:
                # don't initialize the environment here, close unused environments first
                configs_to_create.append(cfg)
        
        # Step 2: Collect ids which need to be closed
        ids_to_close=[]
        # Close unused environments
        for bucket_key, env_ids in env_buckets.items():
            for env_id in env_ids:
                ids_to_close.append(env_id)
                self.envs.pop(env_id)

        # Step 3: Close unused environments
        #print(f"[DEBUG] ids_to_close: {ids_to_close}")
        self.env_client.close_batch(ids_to_close)
        # Step 4: Create new environments
        ids2configs_create = {}
        id=0
        for cfg in configs_to_create:
            id+=1
            while self.split+str(id) in self.envs:
                id+=1
            id_str = self.split+str(id)
            ids2configs_create[id_str] = cfg
            ids2seeds_reset[id_str] = cfg["seed"]
            self.envs[id_str] = REGISTERED_ENV[cfg["env_name"]]["config_cls"](**cfg["env_config"])
        #print(f"[DEBUG] ids2configs_create: {ids2configs_create}")
        self.env_client.create_environments_batch(ids2configs_create)
        # Step 5: Reset environments
        #print(f"[DEBUG] ids2seeds_reset: {ids2seeds_reset}")
        reset_results=self.env_client.reset_batch(ids2seeds_reset)
        
        
        if self.recorder is not None:
            del self.recorder
        self.recorder = defaultdict(list)
        initial_obs = {}
        initial_info = {}
        
        
        for env_id, rst in reset_results.items():
            obs, info = rst
            initial_obs[env_id] = obs
            initial_info[env_id] = info
            self.record(
                env_id, 
                obs=obs, 
                reward=0, 
                done=False, 
                info=info
            )
        
        self.env_states = {env_id: {'step': 0, 'done': False,'metrics':{"turn_metrics":defaultdict(list),"traj_metrics":{}}} for env_id in self.envs}
        self.system_prompts=self.env_client.get_system_prompts_batch(list(self.envs.keys()))
        return initial_obs, initial_info
    
    @torch.no_grad()
    def record(self, env_id, obs, reward, done, info):
        """
        Record each step's obs, info, done, reward,
        Please include "llm_raw_response" in info # it will be decoded by rollout manager and pass to env, then should pass back
        """
        # Create a record entry for this step
        assert obs is not None, "obs cannot be None"
        assert info is not None, "info cannot be None"
        assert isinstance(reward, (int, float)), "reward must be a number"
        assert isinstance(done, bool), "done must be a boolean"
        record_entry = {
            'env_id': env_id,
            'done': done,
            'reward': reward,
            'info': info,
            'obs_str': obs['obs_str'],
        }
        image_placeholder = self.envs[env_id].get('image_placeholder', "<image>")
        if 'multi_modal_data' in obs:
            if image_placeholder in obs['multi_modal_data']:
                record_entry['image_data'] = [process_image(image) for image in obs['multi_modal_data'][image_placeholder]]
        self.recorder[env_id].append(record_entry)

    @torch.no_grad()
    def _single_recording_to_prompt(self,
                            recording: List[Dict], 
                            step: int, 
                            window_size: int = None,
                            is_final: bool = False,
                            prep_for_loss_mask: bool = False,
        ):
        """
        Given a recording, generate the prompt for MLLM
        Chat: Sys -> |InitUser| -> |Assistant, User| -> |Assistant, User| -> ... -> |Assistant, User Final|

        Args:
            recording: List of dictionaries containing recorded environment interactions
            step: Current step to generate prompt for
            window_size: Number of past steps to include in the context
            is_final: Whether the prompt is for the final step 
                - if True, the end of the chat is from the last assistant's response
            prep_for_loss_mask: whether to use special token to wrap llm response
            
        Returns:
            dict: prompt_with_chat_template : str, image_data: list of images, reward: list of reward
        """
        
        assert step >= 0
        start_step = max(0, step - window_size) if window_size is not None else 0
        end_step = step
        assert len(recording) >= end_step + 1, 'History length is not enough'
        history = recording[start_step: end_step + 1]
        rewards=[]
        chat = []
        
        env_id = history[0]['env_id']
        chat.append({"role": "system", "content": self.system_prompts[env_id]})

        image_data=[]
        for i, record in enumerate(history):
            if i>0:
                llm_raw_response = record['info']['llm_raw_response']
                filtered_llm_raw_response = self._handle_special_tokens(llm_raw_response, prep_for_loss_mask=prep_for_loss_mask)
                chat.append({"role": "assistant", "content": filtered_llm_raw_response})
                rewards.append(record['reward'])
            if i<len(history)-1 or not is_final:
                chat.append({"role": "user", "content": record['obs_str']})
                if 'image_data' in record:
                    for img in record['image_data']:
                        image_data.append(img)
            
        prompt_with_chat_template = self.tokenizer.apply_chat_template(chat, add_generation_prompt=(not is_final), tokenize=False)
        if is_final: # NOTE hard coded
            assert prompt_with_chat_template[-1] == '\n', f"The last token should be new line token, got {prompt_with_chat_template[-1]}"
            prompt_with_chat_template = prompt_with_chat_template[:-1] # remove the last in token
        # switch box_end and im_end so that the model can learn to generate <|im_end|>
        prompt_with_chat_template = prompt_with_chat_template.replace(
            f'{self.config.special_token_for_loss_mask[1]}{self.tokenizer.eos_token}',
            f'{self.tokenizer.eos_token}{self.config.special_token_for_loss_mask[1]}')
        return {
            "prompt": prompt_with_chat_template,
            "image_data": image_data,
            "rewards": rewards,
        }
    
    @torch.no_grad()
    def _generate_input_for_rollout(
            self, 
            recording: List[Dict], 
            step: int, 
            window_size: int = None,
        ):
        """
        Given a recording, generate the input for MLLM
        
        Args:
            recording: List of dictionaries containing recorded environment interactions
            step: Current step to generate input for
            window_size: Number of past steps to include in the context
        
        Returns:
            Dictionary containing properly formatted inputs for the MLLM
            - prompts: task instruction
            - responses: responses generated from prompts
            - input_ids, attention_mask, position_ids: prompts and responses generated from prompts
            - position_ids: 
                - position_ids for prompts: rope
                - rest postion_ids: refer to vllm_rollout_spmd.py to check how to compute
        """
        rst=self._single_recording_to_prompt(recording, step, window_size, is_final=False, prep_for_loss_mask=False)
        prompt_with_chat_template=rst['prompt']
        image_data=rst['image_data']        
        has_images = len(image_data) > 0        

        row_dict = {}
        if has_images:  # expand image token
            prompt_with_chat_template, row_dict, _, raw_prompt = self._handle_multi_modal_data(
                prompt_with_chat_template, row_dict, image_data, do_embedding=False)
        else:
            raw_prompt = prompt_with_chat_template

        # use random input_ids and attention_mask for vllm only takes raw_prompt_ids as input when generating sequences
        # TODO check if this is correct
        row_dict['raw_prompt_ids'] = self.tokenizer.encode(raw_prompt, add_special_tokens=False)
        row_dict['input_ids'] = torch.tensor([0], dtype=torch.long)
        row_dict['attention_mask'] = torch.tensor([0], dtype=torch.long)
        row_dict['position_ids'] = torch.tensor([0], dtype=torch.long)

        # add index for each prompt
        index = row_dict.get("extra_info", {}).get("index", 0)
        row_dict["index"] = index

        return row_dict


    @torch.no_grad()
    def _generate_input_for_update(
            self, 
            recording: List[Dict], 
            step: int, 
            window_size: int = None,
        ):
        """
        Given a recording, generate the final input for MLLM
        
        Args:
            recording: List of dictionaries containing recorded environment interactions
            step: Current step to generate input for
            window_size: Number of past steps to include in the context
        
        Returns:
            Dictionary containing properly formatted inputs for the MLLM
            - prompts: task instruction
            - responses: responses generated from prompts
            - input_ids, attention_mask, position_ids: prompts and responses generated from prompts
            - position_ids: 
                - position_ids for prompts: rope
                - rest postion_ids: refer to vllm_rollout_spmd.py to check how to compute

        """



        # handle prompt, prompt=pad_token since we now have everything in response and compute a loss mask for them
        prompt_with_chat_template=self.tokenizer.pad_token 
        
        # handle response
        response_rst=self._single_recording_to_prompt(recording, step, window_size, is_final=True, prep_for_loss_mask=True)
        response_with_chat_template=response_rst['prompt']
        image_data=response_rst['image_data']
        rewards=response_rst['rewards']
       
        has_images = len(image_data) > 0
        row_dict = {}
        if has_images:  # expand image token
            response_with_chat_template, row_dict, image_grid_thw, _ = self._handle_multi_modal_data(
                response_with_chat_template, row_dict, image_data, do_embedding=True)

        
        input_ids_response, attention_mask_response = verl_F.tokenize_and_postprocess_data(prompt=response_with_chat_template,
                                                                         tokenizer=self.tokenizer,
                                                                         max_length=self.config.max_trajectory_length-1, # -1 for the prompt padding token
                                                                         pad_token_id=self.tokenizer.pad_token_id,
                                                                         left_pad=False,
                                                                         truncation=self.config.truncation)
        input_ids_prompt, attention_mask_prompt = verl_F.tokenize_and_postprocess_data(prompt=prompt_with_chat_template,
                                                                         tokenizer=self.tokenizer,
                                                                         max_length=1,
                                                                         pad_token_id=self.tokenizer.pad_token_id,
                                                                         left_pad=True,
                                                                         truncation=self.config.truncation)
        attention_mask_prompt=torch.zeros_like(input_ids_prompt) # All prompt will be masked
        
        
        input_ids_response, attention_mask_response, loss_mask_response,end_of_response_position_mask_response = self._compute_loss_mask(input_ids_response, attention_mask_response)
        
        input_ids_prompt=input_ids_prompt[0]
        attention_mask_prompt=attention_mask_prompt[0]
        loss_mask_prompt = torch.zeros_like(attention_mask_prompt)
        end_of_response_position_mask_prompt = torch.zeros_like(attention_mask_prompt)
        
        input_ids_response=input_ids_response[0]
        attention_mask_response=attention_mask_response[0]
        loss_mask_response=loss_mask_response[0]
        end_of_response_position_mask_response=end_of_response_position_mask_response[0]
        
    
        
        loss_mask = torch.cat([loss_mask_prompt, loss_mask_response], dim=-1)
        end_of_response_position_mask = torch.cat([end_of_response_position_mask_prompt, end_of_response_position_mask_response], dim=-1)
        input_ids = torch.cat([input_ids_prompt, input_ids_response], dim=-1)
        attention_mask = torch.cat([attention_mask_prompt, attention_mask_response], dim=-1)

        
        
        position_ids_prompt = compute_position_id_with_mask(attention_mask_prompt)
        # if self.image_key in row_dict:
        if has_images and image_grid_thw is not None:
            # Qwen model with image_grid_thw
            from verl.models.transformers.qwen2_vl import get_rope_index
            position_ids_response = get_rope_index(
                self.processor,
                image_grid_thw=image_grid_thw,
                input_ids=input_ids_response,
                attention_mask=attention_mask_response,
            )  # (3, seq_len)
            position_ids_prompt=position_ids_prompt.view(1, -1).expand(3, -1)
        else:
            # InternVL model or no images - use simple position computation
            response_length = input_ids_response.shape[0]
            delta_position_id = torch.arange(1, response_length + 1, device=position_ids_prompt.device)
            position_ids_response = position_ids_prompt[-1:] + delta_position_id
        
        if self.config.use_multi_turn_reward:
            reward_positions = torch.nonzero(end_of_response_position_mask).squeeze(-1)
            multi_turn_token_level_rewards = torch.zeros_like(end_of_response_position_mask, dtype=torch.float)
            assert len(reward_positions) == len(rewards), "Number of rewards does not match number of reward positions"
            for idx,reward in enumerate(rewards):
                multi_turn_token_level_rewards[reward_positions[idx]] = reward
            row_dict["multi_turn_token_level_rewards"] = multi_turn_token_level_rewards # (seq_len,) 
            row_dict["end_of_response_position_mask"] = end_of_response_position_mask
        if self.config.use_loss_mask:
            row_dict['loss_mask'] = loss_mask
        if self.config.use_gae_mask:
            row_dict['gae_mask'] = loss_mask
        row_dict["end_of_response_position_mask"] = end_of_response_position_mask # 
        position_ids = torch.cat([position_ids_prompt, position_ids_response], dim=-1)
        row_dict['prompts'] = input_ids_prompt
        row_dict['responses'] = input_ids_response
        row_dict['input_ids'] = input_ids
        row_dict['attention_mask'] = attention_mask
        row_dict['position_ids'] = position_ids
        index = row_dict.get("extra_info", {}).get("index", 0)
        row_dict["index"] = index
        row_dict["step_reward_sum"]= sum(rewards)
        
        return row_dict

    @torch.no_grad()
    def generate_batch_for_rollout(self, step, window_size):
        """
        Generate a batch of data for the current step
        
        Args:
            step: Current step to generate input for
            window_size: Number of past steps to include in the context
        
        Returns:
            Dictionary containing properly formatted inputs for the MLLM
            - None if no data is available (all environments are done)
        """
        batch = []
        self.batch_idx_to_env_id = {}
        batch_idx = 0
        for env_id in self.envs.keys():
            if self.env_states[env_id]['done']:
                continue

            batch.append(self._generate_input_for_rollout(self.recorder[env_id], step, window_size))
            self.batch_idx_to_env_id[batch_idx] = env_id
            batch_idx += 1
        if not batch:
            return None
        if len(batch) % self.config.n_gpus_per_node != 0:
            # Pad the batch to make it divisible by n_gpus_per_node
            while len(batch) % self.config.n_gpus_per_node != 0:
                # do we need to use copy or not here?
                batch.append(batch[-1].copy())
        return collate_fn(batch)
    
    @torch.no_grad()
    def rollout_loop(self):
        """
        Step the environment and record the results
        
        Returns:
            Dictionary containing the results of the step
        """
        for step in range(self.config.max_turns):
            input_batch_dict = self.generate_batch_for_rollout(step, self.config.window_size)
            if input_batch_dict is None:
                break
            input_batch = DataProto.from_single_dict(input_batch_dict)
            if 'multi_modal_data' in input_batch.non_tensor_batch.keys():
                gen_batch = input_batch.pop(
                    batch_keys=['input_ids', 'attention_mask', 'position_ids'],
                    non_tensor_batch_keys=['raw_prompt_ids', 'multi_modal_data'],
                )
            else:
                gen_batch = input_batch.pop(
                    batch_keys=['input_ids', 'attention_mask', 'position_ids'],
                    non_tensor_batch_keys=['raw_prompt_ids'],
                )

            # transform raw_prompt_ids to list instead of numpy array
            # The reason is that when constructing raw_prompt_ids, if the all the list share the same length
            # Numpy array will automatically transfer list to numpy array.
            raw_prompt_ids = gen_batch.non_tensor_batch['raw_prompt_ids']
            raw_prompt_ids_array = np.ndarray(shape=(len(raw_prompt_ids),), dtype=object)
            for i in range(len(raw_prompt_ids)):
                if isinstance(raw_prompt_ids[i],list):
                    raw_prompt_ids_array[i] = raw_prompt_ids[i]
                else:
                    raw_prompt_ids_array[i] = raw_prompt_ids[i].tolist()
            gen_batch.non_tensor_batch['raw_prompt_ids'] = raw_prompt_ids_array
            
            output_batch = self.actor_rollout_wg.generate_sequences(gen_batch)
            
            
            
            responses_str = self.tokenizer.batch_decode(
                output_batch.batch['responses'], 
                skip_special_tokens=True
            ) # seems here will remove special token like "<|im_end|>"
            
            ids2actions = {}
            for batch_idx, env_id in self.batch_idx_to_env_id.items(): 
                ids2actions[env_id] = responses_str[batch_idx]
            
            step_results = self.env_client.step_batch(ids2actions)
            for env_id, rst in step_results.items():
                obs, reward, done, info = rst
                self.env_states[env_id]['step'] += 1
                self.env_states[env_id]['done'] = done
                self.env_states[env_id]['metrics']['traj_metrics'] = info['metrics'].get('traj_metrics', {})
                for k,v in info['metrics']['turn_metrics'].items():
                    self.env_states[env_id]['metrics']['turn_metrics'][k].append(v)
                
                self.record(env_id, obs, reward, done, info)
        
    @torch.no_grad()
    def generate_batch_for_update(self) -> DataProto:
        """
        Get the final trajectory of all environments

        Returns:
            batch (DataProto): batch of final trajectory of all environments
        """
        batch_list = []
        reward_rst=self.env_client.compute_reward_batch(list(self.envs.keys()))
        for env_id in self.envs.keys():
            row_dict = self._generate_input_for_update(
                recording=self.recorder[env_id],
                step=self.env_states[env_id]['step'],
                window_size=None,
            )
            step_reward_sum= row_dict['step_reward_sum']
    
            row_dict['reward_model'] = {"style": "given", "ground_truth": {"reward": reward_rst[env_id]+step_reward_sum}}
            if self.config.use_multi_turn_reward:
                end_of_response_position_mask = row_dict['end_of_response_position_mask']
                reward_positions = torch.nonzero(end_of_response_position_mask).squeeze(-1)
                last_reward_index = reward_positions[-1]
                row_dict['multi_turn_token_level_rewards'][last_reward_index] += reward_rst[env_id]
            batch_list.append(row_dict)
        batch_dict = collate_fn(batch_list)
        batch = DataProto.from_single_dict(batch_dict)
        return batch
    
    @torch.no_grad()
    def recording_to_log(self):
        """
        Get the recording of all environments
        
        Returns:
            Dictionary containing the recording of all environments
        """
        env_info = []
        reward_rst=self.env_client.compute_reward_batch(list(self.envs.keys()))
        for env_id, record in self.recorder.items():
            config_id = self.envs[env_id].config_id()
            step= self.env_states[env_id]['step']
            output_rst = self._single_recording_to_prompt(record, self.env_states[env_id]['step'], window_size=None, is_final=False)
            image= output_rst['image_data']
            done = self.env_states[env_id]['done']
            score = reward_rst[env_id]+ sum(output_rst['rewards'])
            
            
            metrics={
                "score": score,
                "done": done,
                "step": step,
            }
            
            turn_metrics={
                k: sum(v)/step if step != 0 else 0 for k, v in self.env_states[env_id]['metrics']['turn_metrics'].items()
            }
            traj_metrics=self.env_states[env_id]['metrics']['traj_metrics']
            metrics.update(turn_metrics)
            metrics.update(traj_metrics)
            env_info.append({
                "env_id": env_id,
                "config_id": config_id,
                "output_str": output_rst['prompt'],
                "image_data": image,
                "metrics": metrics,
            })
        return env_info
            
            