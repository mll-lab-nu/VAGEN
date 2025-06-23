# Process Reward for Grounding and World Modeling
#
# This file helps to give process rewards for environments that support "grounding", 
# "worldmodeling", or "grounding_worldmodeling" response formats.
#
# Process rewards are calculated in two ways:
# 1. Observation process reward: Evaluates how well the agent grounds its observations
# 2. Prediction process reward: Evaluates the accuracy of the agent's world model predictions
#
# Requirements:
# - The environment must implement get_env_state() method that returns a text-based 
#   description of the current environment state
# - This state description is used as ground truth for calculating process rewards
from typing import List, Dict, Any
import asyncio
import time
from vagen.server.llm_as_judge import run_llm_judge
from vagen.server.llm_as_judge_sokoban_frozenlake import run_llm_judge as run_llm_judge_new

def env_state_reward_wrapper(step_func):
    def wrapped_step(self, action_str):
        if hasattr(self, 'config') and self.config.get("use_state_reward", False):
            
            prompt_format = self.config.get("prompt_format", None)
            if prompt_format is None:
                raise ValueError("Prompt format is not specified in the config.")
            assert ("grounding" in prompt_format or "worldmodeling" in prompt_format)
        
            pre_state = self.get_env_state()
            obs, reward, done, info = step_func(self, action_str)
            post_state = self.get_env_state()
            
            if "metrics" not in info:
                info["metrics"] = {"turn_metrics": {}, "traj_metrics": {}}
            if "turn_metrics" not in info["metrics"]:
                info["metrics"]["turn_metrics"] = {}
                
            if info.get("is_format_rewarded", False): # if no format reward, no need to calculate state reward, skipping
                info["use_state_reward"] = True
                if "observation_content" in info and info["observation_content"]:
                    info["observation_state"] = pre_state
                if "prediction_content" in info and info["prediction_content"]:
                    info["prediction_state"] = post_state
            else:
                info["use_state_reward"] = False
                if "observation_content" in info and info["observation_content"]:
                    info["metrics"]["turn_metrics"]["grounding_reward"] = 0.0
                if "prediction_content" in info and info["prediction_content"]:
                    info["metrics"]["turn_metrics"]["worldmodeling_reward"] = 0.0
            return obs, reward, done, info
        else:
            return step_func(self, action_str)
    return wrapped_step

def service_state_reward_wrapper(step_batch_func):
    def wrapped_step_batch(self, ids2actions):
        # Call the original step_batch function
        step_batch_results = step_batch_func(self, ids2actions)
        if not self.config.get("use_state_reward", False):
            print("[DEUBG] State reward wrapper closed")
            return step_batch_results
        print("[DEUBG] State reward wrapper enabled")
        input_to_llm = []
        for id, result in step_batch_results.items():
            obs, reward, done, info = result
            env_name = self.env_configs[id].get("env_name", "default_env")
            if info.get("use_state_reward", False):
                if info.get("observation_content", None) and info.get("observation_state", None):
                    input_to_llm.append({
                        "id": id,
                        "content": info["observation_content"],
                        "state": info["observation_state"],
                        "type": "grounding",
                        "env_name": env_name,
                    })
                if info.get("prediction_content", None) and info.get("prediction_state", None):
                    input_to_llm.append({
                        "id": id,
                        "content": info["prediction_content"],
                        "state": info["prediction_state"],
                        "type": "worldmodeling",
                        "env_name": env_name,
                    })
                    
        if len(input_to_llm) > 0:
            # Use synchronous batch processing
            results = run_llm_judge(input_to_llm)
        else:
            return step_batch_results
        
        new_step_batch_results = {id: list(result) for id, result in step_batch_results.items()}
        
        for item, result in zip(input_to_llm, results):
            id = item["id"]
            env_config = self.env_configs[id]
            score= result["score"]
            if item["type"] == "grounding":
                new_step_batch_results[id][3]["metrics"]["turn_metrics"]["grounding_reward"] = score * env_config.get("grounding_reward_weight", 0.5)
                new_step_batch_results[id][1] += score * env_config.get("grounding_reward_weight", 0.5)
            elif item["type"] == "worldmodeling":
                new_step_batch_results[id][3]["metrics"]["turn_metrics"]["worldmodeling_reward"] = score * env_config.get("worldmodeling_reward_weight", 0.5)
                new_step_batch_results[id][1] += score * env_config.get("worldmodeling_reward_weight", 0.5)
        
        return {id: tuple(result) for id, result in new_step_batch_results.items()}
                
    return wrapped_step_batch


def service_state_reward_wrapper_v2(step_batch_func):
    def wrapped_step_batch(self, ids2actions):
        # Call the original step_batch function
        step_batch_results = step_batch_func(self, ids2actions)
        if not self.config.get("use_state_reward", False):
            print("[DEUBG] State reward wrapper closed")
            return step_batch_results
        print("[DEUBG] State reward wrapper enabled")
        input_to_llm = []
        for id, result in step_batch_results.items():
            obs, reward, done, info = result
            env_name = self.env_configs[id].get("env_name", "default_env")
            if info.get("use_state_reward", False):
                if info.get("observation_content", None) and info.get("observation_state", None):
                    prompt=self.gen_visual_reasoning_prompt(content=info["observation_content"],state=info["observation_state"],type="grounding",env_name=env_name)
                    input_to_llm.append({
                        "id": id,
                        "content": info["observation_content"],
                        "state": info["observation_state"],
                        "type": "grounding",
                        "env_name": env_name,
                        "prompt":prompt
                    })
                if info.get("prediction_content", None) and info.get("prediction_state", None):
                    prompt=self.gen_visual_reasoning_prompt(content=info["prediction_content"],state=info["prediction_state"],type="worldmodeling",env_name=env_name)
                    input_to_llm.append({
                        "id": id,
                        "content": info["prediction_content"],
                        "state": info["prediction_state"],
                        "type": "worldmodeling",
                        "env_name": env_name,
                        "prompt":prompt
                    })
                    
        if len(input_to_llm) > 0:
            # Use synchronous batch processing
            results = run_llm_judge_new(input_to_llm) # a dict containing a set of metrics
        else:
            return step_batch_results
        
        new_step_batch_results = {id: list(result) for id, result in step_batch_results.items()}
        
        for item, result in zip(input_to_llm, results):
            id = item["id"]
            state=item["state"]
            env_config = self.env_configs[id]
            response= result["parsed_response"]
            kwargs={
                "response": response,
                "state": state,
            }
            score=self.calculate_visual_reasoning_reward(**kwargs)
            if item["type"] == "grounding":
                new_step_batch_results[id][3]["metrics"]["turn_metrics"]["grounding_reward"] = score * env_config.get("grounding_reward_weight", 0.5)
                new_step_batch_results[id][1] += score * env_config.get("grounding_reward_weight", 0.5)
            elif item["type"] == "worldmodeling":
                new_step_batch_results[id][3]["metrics"]["turn_metrics"]["worldmodeling_reward"] = score * env_config.get("worldmodeling_reward_weight", 0.5)
                new_step_batch_results[id][1] += score * env_config.get("worldmodeling_reward_weight", 0.5)
        
        return {id: tuple(result) for id, result in new_step_batch_results.items()}
                
    return wrapped_step_batch


def service_state_reward_wrapper_v3(step_batch_func):
    def wrapped_step_batch(self, ids2actions):
        # Call the original step_batch function
        step_batch_results = step_batch_func(self, ids2actions)
        if not self.config.get("use_state_reward", False):
            print("[DEUBG] State reward wrapper closed")
            return step_batch_results
        print("[DEUBG] State reward wrapper enabled")
        input_to_llm = []
        for id, result in step_batch_results.items():
            obs, reward, done, info = result
            env_name = self.env_configs[id].get("env_name", "default_env")
            if info.get("use_state_reward", False):
                if info.get("observation_content", None) and info.get("observation_state", None):
                    prompt=self.gen_visual_reasoning_prompt(content=info["observation_content"],state=info["observation_state"],type="grounding",env_name=env_name)
                    input_to_llm.append({
                        "id": id,
                        "content": info["observation_content"],
                        "state": info["observation_state"],
                        "type": "grounding",
                        "env_name": env_name,
                        "prompt":prompt
                    })
                if info.get("prediction_content", None) and info.get("prediction_state", None):
                    prompt=self.gen_visual_reasoning_prompt(content=info["prediction_content"],state=info["prediction_state"],type="worldmodeling",env_name=env_name)
                    input_to_llm.append({
                        "id": id,
                        "content": info["prediction_content"],
                        "state": info["prediction_state"],
                        "type": "worldmodeling",
                        "env_name": env_name,
                        "prompt":prompt
                    })
                    
        if len(input_to_llm) > 0:
            # Use synchronous batch processing
            results = run_llm_judge_new(input_to_llm) # a dict containing a set of metrics
        else:
            return step_batch_results
        
        new_step_batch_results = {id: list(result) for id, result in step_batch_results.items()}
        grounding_contents= []
        worldmodeling_contents = []
        for item in input_to_llm:
            if item["type"] == "grounding":
                grounding_contents.append(item["content"])
            elif item["type"] == "worldmodeling":
                worldmodeling_contents.append(item["content"])
        self.top_strings_tracker_grounding.add_strings(grounding_contents)
        self.top_strings_tracker_worldmodeling.add_strings(worldmodeling_contents)
        self.top_strings_tracker_grounding.trim_to_m()
        self.top_strings_tracker_worldmodeling.trim_to_m()
        for item, result in zip(input_to_llm, results):
            id = item["id"]
            state=item["state"]
            content= item["content"]
            r_type=item["type"]
            env_name = item["env_name"]
            prompt=item["prompt"]
            env_config = self.env_configs[id]
            response= result["parsed_response"]
            kwargs={
                "response": response,
                "state": state,
                "content": content,
                "r_type": r_type,
                "env_name": env_name,
                "prompt": prompt
            }
            score=self.calculate_visual_reasoning_reward(**kwargs)
            if item["type"] == "grounding":
                new_step_batch_results[id][3]["metrics"]["turn_metrics"]["grounding_reward"] = score * env_config.get("grounding_reward_weight", 0.5)
                new_step_batch_results[id][1] += score * env_config.get("grounding_reward_weight", 0.5)
            elif item["type"] == "worldmodeling":
                new_step_batch_results[id][3]["metrics"]["turn_metrics"]["worldmodeling_reward"] = score * env_config.get("worldmodeling_reward_weight", 0.5)
                new_step_batch_results[id][1] += score * env_config.get("worldmodeling_reward_weight", 0.5)
        
        return {id: tuple(result) for id, result in new_step_batch_results.items()}
                
    return wrapped_step_batch