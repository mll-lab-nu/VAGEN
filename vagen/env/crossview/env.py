from vagen.env.base.base_env import BaseEnv
from vagen.env.base.base_env_config import BaseEnvConfig
from typing import Dict, List, Tuple, Any, Optional
import json
import os
import random
import re
from PIL import Image
from dataclasses import dataclass, field
from .env_config import CrossViewEnvConfig
from vagen.env.utils.context_utils import parse_llm_raw_response
from .utils import ANSWER_EXTRACTION_MAP, FORMAT_CHECK_MAP

class CrossViewEnv(BaseEnv):
    def __init__(self, config: CrossViewEnvConfig):
        self.config = config
        self.script_dir = os.path.dirname(os.path.abspath(__file__))
        self.image_dir = os.path.join(self.script_dir,config.image_path)
        self.type=config.type
        self.split = config.split
        train_data_path=f"crossviewQA_train_{self.type}.jsonl"
        test_data_path=f"crossviewQA_tinybench_{self.type}.jsonl"
        self.data_path=os.path.join(self.script_dir,"MindCube_RL_Data",train_data_path if self.split=="train" else test_data_path)
        with open(self.data_path, "r") as f:
            self.dataset = [json.loads(line) for line in f]
        self.current_data = None
        self.current_seed = None
        self.done = False
        self.total_reward = 0
        self.answer_extraction = ANSWER_EXTRACTION_MAP[self.type]
        self.format_checking = FORMAT_CHECK_MAP[self.type]

        
    def reset(self, seed=None) -> Tuple[Dict, Dict]:
        """Reset environment with new seed"""
        if seed is not None:
            self.current_seed = seed
            random.seed(seed)
        
        self.done = False
        self.total_reward = 0
        
        # Select a random data point
        idx = self.current_seed % len(self.dataset) if self.current_seed is not None else random.randint(0, len(self.dataset) - 1)
        self.current_data = self.dataset[idx]
        
        # Create observation
        obs = self._create_observation()
        info = {
            "ground_truth": self.current_data["gt_answer"],
            "question_id": self.current_data["id"],
        }
        
        return obs,info
    
    def _create_observation(self) -> Dict:
        """Create observation with question and images"""
        # Get question from conversation
        # Load images
        images = []
        for path in self.current_data["images"]:
            # Handle path that starts with other_all_image/
            
            full_path = os.path.join(self.image_dir, path)
            img = Image.open(full_path)
           
            img = img.resize(self.config.image_size, Image.LANCZOS)
            images.append(img)
           
        
        # Create observation string with image placeholders
        obs_str = self.current_data["question_str"]
        
        return {
            'obs_str': obs_str,
            'multi_modal_data': {
                self.config.image_placeholder: images
            }
        }
    
    def _reward_calculation(self, format_checking_result: bool, success: bool):
        if self.config.reward_type == 'format-only':
            return 1.0 if format_checking_result else 0.0
        elif self.config.reward_type == 'answer-only':
            return 1.0 if success else 0.0
        elif self.config.reward_type == 'format-answer-same':
            r = 0.0
            if format_checking_result:
                r+=1.0
            if success:
                r+=1.0
            return r

        elif self.config.reward_type == 'base':
            r = 0.0
            if format_checking_result:
                r+=1.0
            if success:
                r+=5.0
            return r
        
    
    def step(self, llm_raw_response) -> Tuple[Dict, float, bool, Dict]:
        """Process the LLM's response and compute reward"""

    
        format_checking_result = self.format_checking(llm_raw_response)[0]
        parsed_answer=self.answer_extraction(llm_raw_response)
        gt_answer = self.current_data["gt_answer"]
        if parsed_answer is None:
            format_checking_result = False
        success = format_checking_result and parsed_answer.lower()==gt_answer.lower()
        reward = self._reward_calculation(format_checking_result, success)
        info = {
            "metrics":{ 
                "turn_metrics": {
                "action_is_effective": format_checking_result,
                "action_is_valid": format_checking_result,
            },
                "traj_metrics": {
                    "success":  success,
                }
            },
            "llm_raw_response": llm_raw_response,
        }
        self.done=success
        obs=self._create_observation()
        return obs, reward, True, info
    
    def close(self):
        """Close the environment"""
        pass
    
    def system_prompt(self) -> str:
        return """You are an AI assistant that answers visual questions based on images."""


if __name__ == "__main__":
    # Create config
    config = CrossViewEnvConfig()
    
    # Create environment
    env = CrossViewEnv(config)
    
    print("System prompt:")
    print(env.system_prompt())
    print("\n" + "-"*50 + "\n")
    
    
    i = 0
    while True:
        # Get user input
        # Reset environment and get first observation
        obs, info = env.reset(seed=i)
        print("Question:")
        print(obs["obs_str"])
        print("\nGround truth:", info["ground_truth"])
        if config.image_placeholder in obs["multi_modal_data"] and obs["multi_modal_data"][config.image_placeholder]:
            os.makedirs("./test_crossview", exist_ok=True)
            for j, img in enumerate(obs["multi_modal_data"][config.image_placeholder]):
                img.save(f"./test_crossview/crossview_{i}_{j}.png")
        print(f"\nSaved {len(obs['multi_modal_data'][config.image_placeholder])} images to ./test_crossview/")
        answer = input("\nEnter your answer (or just press Enter to use the default format): ")
        
        # If user just pressed Enter, use a default think/answer format
        if not answer:
            llm_response = "<think>Analyzing the two views...</think><answer>B</answer>"
        # If answer doesn't have the think/answer format, add it
        elif "<think>" not in answer:
            llm_response = f"<think>Analyzing the two views...</think><answer>{answer}</answer>"
        else:
            llm_response = answer
        
        # Step the environment
        obs, reward, done, info = env.step(llm_response)
        
        # Display results
        print("\nAction Result:")
        print(f"info: {info}")
        print(f"Reward: {reward}")
        print(f"Total Reward: {env.compute_reward()}")
        i+=1
    
  