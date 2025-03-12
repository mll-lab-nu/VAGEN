
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

from vagen.env.register import REGISTERED_ENVS
from vagen.env.base import EnvConfig

@dataclass
class QwenVLRolloutConifg:
    window_size: int = 5
    max_prompt_length: int = 2048 # 1024 
    max_response_length: int = 2048 # 1024 
    max_turns: int = 5
    n_gpu_per_node: int = 1 # used for multigpu batch balancing
    sptk_for_loss_mask: List[str] = field(default_factory=lambda: ['<|box_start|>', '<|box_end|>'])
    
class QwenVLRolloutManger():
    def __init__(self,
                 actor_rollout_wg,
                 tokenizer: PreTrainedTokenizer,
                 config: QwenVLRolloutConifg,
                 processor: Optional[ProcessorMixin] = None,
                 verbose: bool = False,
                 truncation='error',
                 ):
        self.tokenizer = tokenizer
        self.processor = processor
        self.config = config
        self.actor_rollout_wg = actor_rollout_wg
        self.verbose = verbose
        self.truncation = truncation
        self.recorder= None # defaultdict(list)
        self.envs = None # dict
        self.env_states = None # dict
        self.batch_idx_to_env_id = None # dict

    def _handle_special_tokens(self, llm_raw_response: str, compute_loss_mask: bool) -> str:
        """
        1. Filter out special tokens: <image> and special tokens marking environment observation in the llm generated response
        2. Add special tokens to the beginning and end of the response if compute_loss_mask is True
        """
        llm_raw_response = re.sub(r'<image>', '', llm_raw_response)
        if compute_loss_mask:
            # filtering special tokens for llm_raw_response, then adding them to the beginning and end of the response
            sptk_b = self.config.sptk_for_loss_mask[0]
            sptk_e = self.config.sptk_for_loss_mask[1]
            llm_raw_response = re.sub(sptk_e, '', llm_raw_response)
            llm_raw_response = re.sub(sptk_b, '', llm_raw_response)
            llm_raw_response = sptk_b + llm_raw_response + sptk_e
        return llm_raw_response
    
    def _handle_multi_modal_data(
            self, 
            prompt_template: str, 
            row_dict: Dict,
            image_data: List[PIL.Image.Image],
            do_embedding: bool = True,
        ) -> str:
        """Handle multi-modal data in the prompt template

        - For do_embedding=False(vllm), replace <image> with <|vision_start|><|image_pad|><|vision_end|> -> raw_prompt
        - For do_embedding=True, replace <image> with <|vision_start|>{image_token}<|vision_end|> -> prompt_template
            - where {image_token} is the length of image embedding
        """
        raw_prompt = prompt_template.replace('<image>', '<|vision_start|><|image_pad|><|vision_end|>')
        row_dict['multi_modal_data'] = {'image': image_data}
        image_grid_thw = None
        if do_embedding:
            image_inputs = self.processor.image_processor(image_data, return_tensors='pt')
            image_grid_thw = image_inputs['image_grid_thw']
            row_dict['multi_modal_inputs'] = {key: val for key, val in image_inputs.items()}
        if image_grid_thw is not None:
            merge_length = self.processor.image_processor.merge_size**2
            index = 0
            while '<image>' in prompt_template:
                prompt_template = prompt_template.replace(
                    '<image>',
                    '<|vision_start|>' + '<|placeholder|>' * (image_grid_thw[index].prod() // merge_length) +
                    '<|vision_end|>',
                    1,
                )
                index += 1

            prompt_template = prompt_template.replace('<|placeholder|>',
                                                        self.processor.image_token)
        
        return prompt_template, row_dict, image_grid_thw, raw_prompt
    
    def _compute_loss_mask(self, input_ids, attention_mask):
        """
        Compute loss mask for the input ids and attention mask

        Args:
            input_ids: (batch_size, seq_len)
            attention_mask: (batch_size, seq_len)
    
        Returns:
            input_ids: (batch_size, seq_len)
            attention_mask: (batch_size, seq_len)
            loss_mask: (batch_size, seq_len)
        
        - There will be different stratgy to handel special tokens in the list
        - 1. remove them, in this case we need to fill the hole by adding pad in the right and shift the sequence left
        - 2. keep them, attention mask will be 0 for them
        - 3. Replace them with pad token
        """
        # let's use 2nd for now
        
        # sptk_b = self.tokenizer.convert_tokens_to_ids(self.config.sptk_for_loss_mask[0])
        # sptk_e = self.tokenizer.convert_tokens_to_ids(self.config.sptk_for_loss_mask[1])
        # loss_mask = torch.ones_like(input_ids) # 0 for no loss, 1 for loss
        # sptk_s_indices = (input_ids == sptk_b).nonzero().flatten()
        # sptk_e_indices = (input_ids[0] == sptk_e).nonzero().flatten()
        # attention_mask[sptk_s_indices] = 0
        # attention_mask[sptk_e_indices] = 0
        # for s,e in zip(sptk_s_indices,sptk_e_indices):
        #     loss_mask[s + 1:e] = 1
        # return input_ids, attention_mask, loss_mask
        
        """
        Let's use the 3rd strategy for now
        Compute loss mask for the input ids and attention mask by:
        1. Removing special tokens
        2. Adding padding on the right
        3. Shifting the sequence left
        """
        
        # Get token IDs for special tokens and pad token
        sptk_b = self.tokenizer.convert_tokens_to_ids(self.config.sptk_for_loss_mask[0])
        sptk_e = self.tokenizer.convert_tokens_to_ids(self.config.sptk_for_loss_mask[1])
        pad_token_id = self.tokenizer.pad_token_id
        
        batch_size, seq_len = input_ids.shape
        
        # Initialize output tensors with same shape as inputs
        new_input_ids = input_ids.clone()
        new_attention_mask = attention_mask.clone()
        loss_mask = torch.zeros_like(input_ids)
        
        # Process each example in the batch
        for b in range(batch_size):
            # Count right padding tokens using attention mask
            right_pad_tokens = (attention_mask[b] == 0).cumprod(dim=0).sum().item()
            
            # Assert that initial padding tokens have attention mask of 0
            assert torch.all(attention_mask[b, -right_pad_tokens:] == 0), "right padding tokens must have attention mask of 0"
            
            # Find special token indices
            sptk_b_indices = (input_ids[b] == sptk_b).nonzero().flatten()
            sptk_e_indices = (input_ids[b] == sptk_e).nonzero().flatten()
            
            # Create a mask for tokens that should compute loss
            hole_pos=[] # initialize holes position list with last padding token position
            for start_pos, end_pos in zip(sptk_b_indices, sptk_e_indices):
                loss_mask[b][start_pos+1:end_pos] = 1
                hole_pos.append(start_pos.item())
                hole_pos.append(end_pos.item())
            hole_pos.append(seq_len-right_pad_tokens)
            assert new_input_ids[b][seq_len-right_pad_tokens]==pad_token_id
            
            # shift right to fill the wholes
            holes_to_fill=1
            for i in range(0,len(hole_pos)-1):
                start_pos = hole_pos[i]
                end_pos = hole_pos[i+1]
                loss_mask[b][start_pos+1-holes_to_fill:end_pos-holes_to_fill]=loss_mask[b][start_pos+1:end_pos]
                new_input_ids[b][start_pos+1-holes_to_fill:end_pos-holes_to_fill]=new_input_ids[b][start_pos+1:end_pos]
                new_attention_mask[b][start_pos+1-holes_to_fill:end_pos-holes_to_fill]=new_attention_mask[b][start_pos+1:end_pos]
                holes_to_fill+=1

            valid_tokens = seq_len-right_pad_tokens-len(hole_pos)+1 # the number of non-special tokens and non-padding tokens
            loss_mask[b][valid_tokens:]=0
            new_input_ids[b][valid_tokens:]=pad_token_id
            new_attention_mask[b][valid_tokens:]=0
            
        return new_input_ids, new_attention_mask, loss_mask
    
        
    def reset(self, env_configs: List[EnvConfig]):
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
        
        env_buckets = defaultdict(set)
        new_envs = {}
        
        if self.envs is None:
            self.envs = {}
            
        for env_id, env in self.envs.items():
            env_name = env.name_repr()
            env_config = env.config_repr(env.env_config)
            bucket_key = f"{env_name}:{env_config}"
            env_buckets[bucket_key].add(env_id)
        
        for i, cfg in enumerate(env_configs):
            env_id = i
            env_name = cfg.env_name
            env_config = cfg.env_config
            seed = cfg.seed
            
            # Create bucket key
            config_key = REGISTERED_ENVS[env_name].config_repr(env_config)
            bucket_key = f"{env_name}:{config_key}"
            
            # Check if we have an available environment with the same config
            if bucket_key in env_buckets and env_buckets[bucket_key]:
                old_env_id = env_buckets[bucket_key].pop()
                new_envs[env_id] = {
                    "env":self.envs[old_env_id],
                    "seed":seed,
                    "env_config":env_config,
                }
            else:
                # don't initialize the environment here, close unused environments first
                new_envs[env_id] = {
                    "env_class":REGISTERED_ENVS[env_name],
                    "seed":seed,
                    "env_config":env_config,
                }
        
        # Close unused environments
        for bucket_key, env_ids in env_buckets.items():
            for env_id in env_ids:
                self.envs[env_id].close()
                del self.envs[env_id]

        
        # Step 2: Reset environments and collect observations/info
        
        if self.recorder is not None:
            del self.recorder
        self.recorder = defaultdict(list)
        initial_obs = {}
        initial_info = {}
        for env_id, env_info in new_envs.items():
            if "env" in env_info:
                self.envs[env_id] = env_info["env"]
            else:
                assert "env_class" in env_info
                self.envs[env_id] = env_info["env_class"](**env_info["env_config"])
            obs, info = self.envs[env_id].reset(env_info["seed"])
            initial_obs[env_id] = obs
            initial_info[env_id] = info
            self.record(
                env_id, 
                obs=obs, 
                reward=0, 
                done=False, 
                info=info
            )
        
        self.env_states = {env_id: {'step': 0, 'done': False} for env_id in self.envs}
        
        return initial_obs, initial_info
    
    
    def record(self, env_id, obs, reward, done, info):
        """
        Record each step's obs, info, done, reward,
        Please include "llm_raw_response" in info # it will be decoded by rollout manager and pass to env, then should pass back
        """
        # Create a record entry for this step
        record_entry = {
            'env_id': env_id,
            'done': done,
            'reward': reward,
            'info': copy.deepcopy(info) if info is not None else None
        }
        
        # Process observations if provided
        if obs is not None:
            template = obs['text_template']
            mm_observation = obs['multi_modal_data']
            
            # Get all placeholders from mm_observation keys
            placeholders = list(mm_observation.keys())

            if placeholders and self.processor is not None:
                record_entry["image_data"] = []

            # Create a list of all placeholder positions with their corresponding placeholder
            all_positions = []
            for placeholder in placeholders:
                positions = [m.start() for m in re.finditer(placeholder, template)]
                for pos in positions:
                    all_positions.append((pos, placeholder))

            # Sort positions in ascending order
            all_positions.sort(key=lambda x: x[0])  # This sorts based on the first element of each tuple (position)

            # Now process in order of occurrence
            while all_positions:
                position, placeholder = all_positions.pop(0)
                template = template[:position] + '<image>' + template[position + len(placeholder):]
                if "image_data" in record_entry:
                    processed_image = process_image(mm_observation[placeholder])
                    record_entry["image_data"].append(processed_image)
            
            record_entry["text_template"] = template
            
        # if observation is not None:
        #     template = observation.observation_template
        #     mm_observation = observation.multi_modal_observation

        #     mllm_keys = re.findall(r'<image([^>]+)>', template)
        #     # Process multimodal inputs if present in observation
        #     if mllm_keys and self.processor is not None:
        #         record_entry["image_data"] = [process_image(mm_observation[key]) for key in mllm_keys]
        #         #record_entry["image_inputs"] = self.processor.image_processor(record_entry["image_data"], return_tensors='pt')
        #         record_entry["text_template"] = re.sub(r'<image([^>]+)>', '<image>', template)

        self.recorder[env_id].append(record_entry)


    def _generate_input_item(
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
        assert step >= 0
        
        start_step = max(0, step - window_size) if window_size is not None else 0
        end_step = step
        assert len(recording) >= max(1, end_step + 1 - start_step), 'History length is not enough'
        history = recording[start_step: end_step + 1]

        chat = []
        first_record = history[0]
        env_id = first_record['env_id']
        chat.append({"role": "system", "content": self.envs[env_id].get_task_instruction()})
        if 'text_template' in first_record:
            chat.append({"role": "user", "content": first_record['text_template']})
    
        for i, record in enumerate(history[1:], 1):
            chat.append({"role": "user", "content": record['text_template']})
            if i < len(history) - 1:
                assert 'llm_raw_response' in record['info']
                llm_raw_response = record['info']['llm_raw_response']
                filtered_llm_raw_response = self._remove_special_tokens(llm_raw_response, compute_loss_mask=False)
                chat.append({"role": "assistant", "content": filtered_llm_raw_response})

        image_data = [image for record in history if 'image_data' in record for image in record["image_data"]]
        has_images = len(image_data) > 0        
        prompt_with_chat_template = self.tokenizer.apply_chat_template(chat, add_generation_prompt=True, tokenize=False) + '<think>'

        row_dict = {}
        if has_images:  # expand image token
            prompt_with_chat_template, row_dict, _, raw_prompt = self._handle_multi_modal_data(
                prompt_with_chat_template, row_dict, image_data, do_embedding=False)
        else:
            raw_prompt = prompt_with_chat_template

        # use random input_ids and attention_mask for vllm only takes raw_prompt_ids as input when generating sequences
        # TODO check if this is correct
        row_dict['input_ids'] = torch.randint(0, 10000, (1024,))
        row_dict['attention_mask'] = torch.randint(0, 1, (1024,))
        row_dict['position_ids'] = torch.randint(0, 1024, (1024,))
        row_dict['raw_prompt_ids'] = self.tokenizer.encode(raw_prompt, add_special_tokens=False)

        # encode prompts without chat template
        # if self.return_raw_chat:
        #     row_dict['raw_prompt'] = chat.tolist()

        # add index for each prompt
        index = row_dict.get("extra_info", {}).get("index", 0)
        row_dict["index"] = index

        return row_dict



    def _generate_input_final_item(
            self, 
            recording: List[Dict], 
            step: int, 
            window_size: int = None,
            compute_loss_mask: bool = False,
        ):
        """
        Given a recording, generate the final input for MLLM
        
        Args:
            recording: List of dictionaries containing recorded environment interactions
            step: Current step to generate input for
            window_size: Number of past steps to include in the context
            compute_loss_mask: Whether to compute loss mask (loss mask: 1 for assistant response, 0 for user response)
        
        Returns:
            Dictionary containing properly formatted inputs for the MLLM
            - prompts: task instruction
            - responses: responses generated from prompts
            - input_ids, attention_mask, position_ids: prompts and responses generated from prompts
            - position_ids: 
                - position_ids for prompts: rope
                - rest postion_ids: refer to vllm_rollout_spmd.py to check how to compute

        """
        assert step >= 0
        
        start_step = max(0, step - window_size) if window_size is not None else 0
        end_step = step
        assert len(recording) >= max(1, end_step + 1 - start_step), 'History length is not enough'
        history = recording[start_step: end_step + 1]


        init_chat = []
        first_record = history[0]
        env_id = first_record['env_id']
        init_chat.append({"role": "system", "content": self.envs[env_id].get_task_instruction()})
        if 'text_template' in first_record:
            init_chat.append({"role": "user", "content": first_record['text_template']})
        prompt_with_chat_template = self.tokenizer.apply_chat_template(init_chat, add_generation_prompt=True, tokenize=False) + '<think>'
        image_data = [first_record['image_data']] if 'image_data' in first_record else []
        has_images = len(image_data) > 0
        row_dict = {}
        if has_images:  # expand image token
            prompt_with_chat_template, row_dict, image_grid_thw, _ = self._handle_multi_modal_data(
                prompt_with_chat_template, row_dict, image_data, do_embedding=True)



        response_chat = [] # currently use chat format for response
        for i, record in enumerate(history[1:], 1):
            response_chat.append({"role": "user", "content": record['text_template']})
            assert 'llm_raw_response' in record['info']
            llm_raw_response = record['info']['llm_raw_response']
            filtered_llm_raw_response = self._handle_special_tokens(llm_raw_response, compute_loss_mask=compute_loss_mask)
            response_chat.append({"role": "assistant", "content": filtered_llm_raw_response})
        response_with_chat_template = self.tokenizer.apply_chat_template(response_chat, add_generation_prompt=False, tokenize=False)


        input_ids_prompt, attention_mask_prompt = verl_F.tokenize_and_postprocess_data(prompt=prompt_with_chat_template,
                                                                         tokenizer=self.tokenizer,
                                                                         max_length=self.config.max_prompt_length,
                                                                         pad_token_id=self.tokenizer.pad_token_id,
                                                                         left_pad=True,
                                                                         truncation=self.truncation)

        input_ids_response, attention_mask_response = verl_F.tokenize_and_postprocess_data(prompt=response_with_chat_template,
                                                                         tokenizer=self.tokenizer,
                                                                         max_length=self.config.max_response_length,
                                                                         pad_token_id=self.tokenizer.pad_token_id,
                                                                         left_pad=False,
                                                                         truncation=self.truncation)
        
        input_ids_prompt = input_ids_prompt[0]
        input_ids_response = input_ids_response[0]
        attention_mask_prompt = attention_mask_prompt[0]
        attention_mask_response = attention_mask_response[0]

        if compute_loss_mask:
            input_ids_response, attention_mask_response, loss_mask_response = self._compute_loss_mask(input_ids_response, attention_mask_response)
            loss_mask_prompt = torch.zeros_like(attention_mask_prompt)
            loss_mask = torch.cat([loss_mask_prompt, loss_mask_response], dim=-1)
        input_ids = torch.cat([input_ids_prompt, input_ids_response], dim=-1)
        attention_mask = torch.cat([attention_mask_prompt, attention_mask_response], dim=-1)

        # if self.image_key in row_dict:
        if has_images:
            from verl.models.transformers.qwen2_vl import get_rope_index

            position_ids_prompt = get_rope_index(
                self.processor,
                image_grid_thw=image_grid_thw,
                input_ids=input_ids_prompt,
                attention_mask=attention_mask_prompt,
            )  # (3, seq_len)
        else:
            position_ids_prompt = compute_position_id_with_mask(attention_mask_prompt)

        """
        Following codes are adapted from vllm_rollout_spmd.py
        TODO check if this is correct
        """
        position_ids_prompt = position_ids_prompt
        
        response_length = input_ids_response.shape[0]
        
        # Create position ids for response
        delta_position_id = torch.arange(1, response_length + 1, device=position_ids_prompt.device)
        delta_position_id = delta_position_id
        
        if position_ids_prompt.dim() == 2:  # qwen2vl mrope
            delta_position_id = delta_position_id.view(1, -1).expand(3, -1)
        
        # Concatenate prompt and response position ids
        position_ids_response = position_ids_prompt[:, -1:] + delta_position_id
        position_ids = torch.cat([position_ids_prompt, position_ids_response], dim=-1)

        row_dict['prompts'] = input_ids_prompt
        row_dict['responses'] = input_ids_response
        row_dict['input_ids'] = input_ids
        row_dict['attention_mask'] = attention_mask
        row_dict['position_ids'] = position_ids
        if compute_loss_mask:
            row_dict['loss_mask'] = loss_mask
        # NOTE there is no raw_prompt_ids in the final item

        # encode prompts without chat template
        # if self.return_raw_chat:
        #     row_dict['raw_prompt'] = chat.tolist()

        # add index for each prompt
        index = row_dict.get("extra_info", {}).get("index", 0)
        row_dict["index"] = index

        return row_dict


    def gen_batch(self, step, window_size):
        """
        Generate a batch of data for the current step
        
        Args:
            step: Current step to generate input for
            window_size: Number of past steps to include in the context
        
        Returns:
            Dictionary containing properly formatted inputs for the MLLM›
        """
        batch = []
        self.batch_idx_to_env_id = {}
        batch_idx = 0
        for env_id in self.envs.keys():
            if self.env_states[env_id]['done']:
                continue
            # batch.append(self.__getitem__(self.recorder[env_id], step, window_size))
            batch.append(self._generate_input_item(self.recorder[env_id], step, window_size))
            self.batch_idx_to_env_id[batch_idx] = env_id
            batch_idx += 1
        if len(batch) % self.config.n_gpu_per_node != 0:
            # Pad the batch to make it divisible by n_gpu_per_node
            while len(batch) % self.config.n_gpu_per_node != 0:
                batch.append(batch[-1])
        return collate_fn(batch)
    
    
    
    def rollout_loop(self):
        """
        Step the environment and record the results
        
        Returns:
            Dictionary containing the results of the step
        """
        for step in range(self.config.max_turns):
            input_batch_dict = self.gen_batch(step, self.config.window_size)
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
                raw_prompt_ids_array[i] = raw_prompt_ids[i].tolist()
            gen_batch.non_tensor_batch['raw_prompt_ids'] = raw_prompt_ids_array
            
            output_batch = self.actor_rollout_wg.generate_sequences(gen_batch)
            responses_str = self.tokenizer.batch_decode(
                output_batch.batch['responses'], 
                skip_special_tokens=True
            )
            
            for batch_idx, env_id in self.batch_idx_to_env_id.items(): # TODO whether multiple actions in one rollout are considered here
                obs, reward, done, info = self.envs[env_id].step(responses_str[batch_idx])
                self.env_states[env_id]['step'] += 1
                self.env_states[env_id]['done'] = done
                self.record(env_id, obs, reward, done, info)
        
        
        
    def get_final_trajectory(self) -> DataProto:
        """
        Get the final trajectory of all environments

        Returns:
            Dictionary containing the final trajectory of all environments
            - prompts: task instruction
            - responses: responses generated from prompts
            - input_ids, attention_mask, position_ids: prompts and responses generated from prompts
            - position_ids: 
                - position_ids for prompts: rope
                - rest postion_ids: refer to vllm_rollout_spmd.py to check how to compute
        """
        batch_list = []
        for env_id in self.envs.keys():
            # row_dict = self.__getitem__(
            #     self.recorder[env_id],
            #     self.env_states[env_id]['step'],
            #     self.config.window_size,
            #     compute_loss_mask=True,
            #     final=True,
            # )
            row_dict = self._generate_input_final_item(
                recording=self.recorder[env_id],
                step=self.env_states[env_id]['step'],
                window_size=self.config.window_size,
                compute_loss_mask=True,
            )
            row_dict['reward_model'] = {"style": "given", "ground_truth": {"reward": self.envs[env_id].get_traj_reward()}}
            batch_list.append(row_dict)
        batch_dict = collate_fn(batch_list)
        batch = DataProto.from_single_dict(batch_dict)
        return batch