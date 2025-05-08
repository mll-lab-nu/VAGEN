from typing import List, Dict, Any
import asyncio
import re
import json
import os
import time
import yaml
import hydra
from omegaconf import DictConfig, OmegaConf
from datetime import datetime
from together import AsyncTogether

CONFIG_NAME = "llm_judge"
CONFIG_PATH = "./"

PROMPT_TEMPLATE = """
Compare the natural language description with the state information dictionary.
Answer YES if the description accurately matches the state, or NO if it doesn't.
Think step by step and end with your answer in <answer>YES</answer> or <answer>NO</answer> format.

State Information:
{state_information_dict}

Description:
"{natural_language_description}"

Your answer should be within {max_tokens} tokens and MUST end with <answer>YES</answer> or <answer>NO</answer>.
"""

def load_config() -> DictConfig:
    """Load Hydra configuration properly"""
    from hydra import compose, initialize
    
    with initialize(version_base=None, config_path=CONFIG_PATH):
        config = compose(config_name=CONFIG_NAME)
    
    return config

async def llm_judge(
    inputs: List[Dict[str, Any]],
) -> List[float]:
    """
    Judge the content based on the state using a language model.
    
    Args:
        inputs: A list of dicts: {"id": id, "content": content, "state": state, "env_name": env_name}
        
    Returns:
        list: A list of scores: 0 or 1 for each input.
    """
    if not inputs:
        return []
    
    config = load_config()
    
    # Extract parameters from config
    model = config.get("name", "meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8")
    max_parallel_requests = config.get("max_parallel_requests", 10)
    temperature = config.get("temperature", 0.1)
    max_tokens = config.get("max_tokens", 200)
    max_retries = config.get("max_retries", 3)
    
    # Initialize async client
    async_client = AsyncTogether(api_key=os.getenv("TOGETHER_API_KEY"), max_retries=max_retries)
    
    # Prepare evaluation items
    eval_items = []
    for item in inputs:
        prompt = PROMPT_TEMPLATE.format(
            state_information_dict=json.dumps(item["state"], indent=2),
            natural_language_description=item["content"],
            max_tokens=max_tokens
        )
        
        eval_items.append({
            "id": item["id"],
            "env_name": item.get("env_name", "unknown"),
            "prompt": prompt
        })
    
    # Process items in parallel
    semaphore = asyncio.Semaphore(max_parallel_requests)
    
    async def process_item(item):
        try:
            async with semaphore:
                messages = [{"role": "user", "content": item["prompt"]}]
                
                response = await async_client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens
                )
                
                response_text = response.choices[0].message.content
                answer = extract_answer(response_text)
                
                return {
                    "id": item["id"],
                    "score": 1.0 if answer == "YES" else 0.0
                }
        except Exception as e:
            print(f"Error in API call for item {item['id']}: {str(e)}")
            return {
                "id": item["id"],
                "score": 0.0
            }
    
    # Execute all API calls
    eval_results = await asyncio.gather(*[process_item(item) for item in eval_items])
    
    # Return scores in the same order as inputs
    scores = []
    id_to_result = {result["id"]: result for result in eval_results}
    
    for item in inputs:
        result = id_to_result.get(item["id"])
        scores.append(result["score"] if result else 0.0)
    
    return scores

def extract_answer(response_text: str) -> str:
    """Extract YES/NO answer from LLM response"""
    match = re.search(r"<answer>(YES|NO)</answer>", response_text, re.IGNORECASE)
    if match:
        return match.group(1).upper()
    
    # Fallback: Check if YES or NO appears in the text
    if "YES" in response_text.upper() and "NO" not in response_text.upper():
        return "YES"
    elif "NO" in response_text.upper() and "YES" not in response_text.upper():
        return "NO"
    
    return None

if __name__ == "__main__":
    from vagen.env.frozenlake import FrozenLakeEnv, FrozenLakeEnvConfig
    
    # Create test environments
    base_configs = [
        FrozenLakeEnvConfig(size=5, is_slippery=False), 
        FrozenLakeEnvConfig(size=5, is_slippery=True)
    ]
    environments = [FrozenLakeEnv(config) for config in base_configs]
    
    # Setup evaluation inputs
    inputs = []
    for i, env in enumerate(environments):
        obs, _ = env.reset(seed=42+i)
        
        # Take step with simple action
        action = "<answer>Right</answer>" if i == 0 else "<answer>Down</answer>"
        obs, reward, done, info = env.step(action)
        
        # Get player position
        player_pos = env._get_player_position()
        row, col = player_pos
        
        # Create state information
        state = {
            "player_position": {"row": int(row), "col": int(col)},
            "map_size": env.config.size,
            "is_slippery": env.config.is_slippery,
            "map_description": [[cell.decode('utf-8') for cell in row] for row in env.gym_env.desc]
        }
        
        # Create natural language description - correct for env 0, incorrect for env 1
        content = (
            f"The player is at position ({row},{col}) on a {env.config.size}x{env.config.size} grid." 
            if i == 0 else 
            f"The player is at position ({row+1},{col-1}) on a {env.config.size}x{env.config.size} grid."
        )
        
        inputs.append({
            "id": i,
            "content": content,
            "state": state,
            "env_name": "frozenlake"
        })
    
    # Run evaluation
    print("Starting LLM judging...")
    start_time = time.time()
    
    scores = asyncio.run(llm_judge(inputs=inputs))
    
    print(f"Completed in {time.time() - start_time:.2f} seconds")
    
    # Print results
    for i, score in enumerate(scores):
        print(f"\nEnvironment {i}:")
        print(f"Score: {score}")
    
    # Clean up
    for env in environments:
        env.close()