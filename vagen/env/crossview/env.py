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


class CrossViewEnv(BaseEnv):
    def __init__(self, config: CrossViewEnvConfig):
        self.config = config
        self.script_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),"CrossViewQA")
        self.split = config.split
        self.image_dir = os.path.join(self.script_dir, config.image_dir)
        
        # Load dataset
        if self.split == "train":
            self.data_path=os.path.join(self.script_dir, config.train_data_path)
            with open(self.data_path, "r") as f:
                self.dataset = json.load(f)   
        elif self.split == "test":
            self.data_path=os.path.join(self.script_dir, config.test_data_path)
           # this is jsonl file
            with open(self.data_path, "r") as f:
                self.dataset = [json.loads(line) for line in f]
            self.dataset=self._convert_data_format(self.dataset)
        self.current_data = None
        self.current_seed = None
        self.done = False
        self.total_reward = 0
    
    def _convert_data_format(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Convert the JSONL format to the required JSON structure.
        
        Args:
            data: List of dictionaries from the JSONL file
            
        Returns:
            List of dictionaries in the required format with id, conversation, and images
        """
        converted_data = []
        
        for item in data:
            # Create the conversation structure
            conversation = [
                {
                    "role": "user",
                    "content": item["question"]
                },
                {
                    "role": "assistant",
                    "content": item["gt_answer"]
                }
            ]
            
            # Create the converted item
            converted_item = {
                "id": item["id"],
                "conversation": conversation,
                "images": item["images"]
            }
            
            converted_data.append(converted_item)
        
        return converted_data
        
        
        
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
            "ground_truth": self.current_data["conversation"][1]["content"],
            "question_id": self.current_data["id"],
        }
        
        return obs,info
    
    def _create_observation(self) -> Dict:
        """Create observation with question and images"""
        # Get question from conversation
        question = self.current_data["conversation"][0]["content"]
        
        # Load images
        images = []
        for path in self.current_data["images"]:
            # Handle path that starts with other_all_image/
            
            full_path = os.path.join(self.image_dir, path)
            img = Image.open(full_path)
           
            img = img.resize(self.config.image_size, Image.LANCZOS)
            images.append(img)
           
        
        # Create observation string with image placeholders
        image_placeholders = " ".join([self.config.image_placeholder] * len(images))
        obs_str = f"""Question: {question}
{image_placeholders}
Please look at the images and answer the question. 
Your answer should be in the format of <think>...</think><answer>...</answer>. 
Please give your thought first then answer.
e.g. <think>I can see there're multiple images with different view. I can see from the second view the object is on the target's left.I think the correct answer is A</think><answer>A</answer>"""
        
        return {
            'obs_str': obs_str,
            'multi_modal_data': {
                self.config.image_placeholder: images
            }
        }
    
    def step(self, llm_raw_response) -> Tuple[Dict, float, bool, Dict]:
        """Process the LLM's response and compute reward"""
 
        # Parse the response
        parsed_response = parse_llm_raw_response(
            llm_raw_response,
            special_token_list=self.config.special_token_list,
            action_sep=self.config.action_sep
        )
        
        # Get action content and ground truth
        action_content = parsed_response["action_content"].strip()
        ground_truth = self.current_data["conversation"][1]["content"].strip()
        
        # Simple exact match (case-insensitive)
        action_is_valid = action_content != ""
        success = action_is_valid and action_content.strip().lower()[0] == ground_truth.strip().lower()[0]
        action_is_effective = action_is_valid
        
        # Compute reward - base reward + format reward if applicable
        reward = 5.0 if success else 0.0
        if parsed_response["format_correct"] and action_is_valid:
            reward += self.config.format_reward
        
        self.total_reward += reward
        
        # Set done to True (single-step environment)
        self.done = True
        
        # Return observation, reward, done, info
        obs = self._create_observation()
        
        info = {
            "metrics":{ 
                "turn_metrics": {
                "action_is_effective": action_is_effective,
                "action_is_valid": action_is_valid,
            },
                "traj_metrics": {
                    "success": success,  # Will be set to True if agent reaches goal
                }
            },
            "llm_raw_response": llm_raw_response,
            "llm_response": parsed_response["llm_response"],
            "think_content": parsed_response["think_content"],
            "action_content": action_content,
            "actions": parsed_response["actions"],
            "ground_truth": ground_truth,
        }
        
        return obs, reward, self.done, info
    
    def close(self):
        """Close the environment"""
        pass
    
    def system_prompt(self) -> str:
        return """You are an AI assistant that answers visual questions based on images.
Given images and a question, first give your thought then answer.
Your answer should be in the format of <think>...</think><answer>...</answer>.
e.g. <think>I can see there're multiple images with different view. I can see from the second view the object is on the target's left.I think the correct answer is A</think><answer>A</answer>"""


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
    
  