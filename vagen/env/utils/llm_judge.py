from typing import List, Dict, Any, Optional, Tuple
import asyncio
import re
import json
import os
import time
import hydra
import uuid
from hydra.core.global_hydra import GlobalHydra
from omegaconf import DictConfig, OmegaConf
from datetime import datetime
from together import AsyncTogether
from pathlib import Path
import wandb
import threading
import random
from contextlib import contextmanager

# Global variables for wandb tracking per process
_WANDB_INITIALIZED = {}  # Track initialization status per process
_GLOBAL_STEPS = {}  # Track global step count per process
_PROCESS_LOCKS = {}  # Semaphore for each process

# Context manager to ensure proper cleanup of wandb sessions
@contextmanager
def wandb_run_context():
    """Context manager for wandb runs that ensures proper cleanup"""
    try:
        yield
    finally:
        # If wandb is running, finish the run
        if wandb.run is not None:
            wandb.finish()
            
def run_llm_judge(input_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Process input data through the LLM judge and log results to Weights & Biases.
    
    Args:
        input_data: List of dictionaries containing judgment inputs
        
    Returns:
        List of dictionaries with judgment results including scores
        
    This function manages wandb logging per process:
    1. Initializes wandb if not already done for the current process
    2. Uses a semaphore to ensure sequential calls within the same process
    3. Tracks global step count per process
    4. Logs statistics for the overall run and selected examples in wandb tables
    """
    # Get the process ID to manage per-process variables
    pid = os.getpid()
    
    # Initialize the lock for this process if it doesn't exist
    if pid not in _PROCESS_LOCKS:
        _PROCESS_LOCKS[pid] = threading.Semaphore(1)
    
    # Use the semaphore to ensure this function is called sequentially within the process
    with _PROCESS_LOCKS[pid]:
        # Initialize global step for this process if not already done
        if pid not in _GLOBAL_STEPS:
            _GLOBAL_STEPS[pid] = 0
        
        # Increment the global step
        _GLOBAL_STEPS[pid] += 1
        global_step = _GLOBAL_STEPS[pid]
        
        # Initialize wandb if not already done for this process
        if pid not in _WANDB_INITIALIZED or not _WANDB_INITIALIZED[pid]:
            # Get the directory containing the current script file
            script_dir = Path(__file__).parent
            
            # Initialize Hydra with the config file
            if not GlobalHydra.instance().is_initialized():
                hydra.initialize(config_path=str(script_dir))
                
            # Load the config
            config = hydra.compose(config_name="llm_as_judge")
            
            # Initialize wandb with values from config
            run_id = str(uuid.uuid4())[:8]
            wandb.init(
                project=config.wandb.project,
                name=f"{config.wandb.run_name}_{run_id}",
                config={
                    "model": config.model.name,
                    "temperature": config.model.temperature,
                    "max_tokens": config.model.max_tokens,
                    "max_retries": config.api.max_retries,
                    "request_timeout": config.api.request_timeout,
                    "correct_grounding_samples": config.wandb.correct_grounding_samples,
                    "incorrect_grounding_samples": config.wandb.incorrect_grounding_samples,
                    "correct_worldmodeling_samples": config.wandb.correct_worldmodeling_samples,
                    "incorrect_worldmodeling_samples": config.wandb.incorrect_worldmodeling_samples
                }
            )
            
            _WANDB_INITIALIZED[pid] = True
        
        # Get sampling parameters from wandb config
        config = wandb.config
        correct_grounding_samples = config.get("correct_grounding_samples", 3)
        incorrect_grounding_samples = config.get("incorrect_grounding_samples", 3)
        correct_worldmodeling_samples = config.get("correct_worldmodeling_samples", 3)
        incorrect_worldmodeling_samples = config.get("incorrect_worldmodeling_samples", 3)
        
        # Measure execution time
        start_time = time.time()
        
        # Process the judgments
        results = process_llm_judgments(input_data)
        
        # Calculate execution time
        execution_time = time.time() - start_time
        
        # Calculate statistics
        total_requests = len(results)
        completed_requests = sum(1 for r in results if r["success"])
        
        # Split by type
        grounding_results = [r for r in results if r["type"] == "grounding"]
        worldmodeling_results = [r for r in results if r["type"] == "worldmodeling"]
        
        # Calculate accuracy metrics
        overall_accuracy = sum(r["score"] for r in results) / total_requests if total_requests > 0 else 0
        
        grounding_accuracy = (
            sum(r["score"] for r in grounding_results) / len(grounding_results) 
            if grounding_results else 0
        )
        
        worldmodeling_accuracy = (
            sum(r["score"] for r in worldmodeling_results) / len(worldmodeling_results) 
            if worldmodeling_results else 0
        )
        
        # Calculate parse success rate
        parse_successes = sum(1 for r in results if r["success"] and (r["score"] == 1.0 or r["score"] == 0.0))
        parse_success_rate = parse_successes / completed_requests if completed_requests > 0 else 0
        
        # Log scalar metrics to wandb with step to ensure proper plotting
        wandb.log({
            "global_step": global_step,
            "execution_time": execution_time,
            "total_requests": total_requests,
            "completed_requests": completed_requests,
            "completion_rate": completed_requests / total_requests if total_requests > 0 else 0,
            "overall_accuracy": overall_accuracy,
            "grounding_count": len(grounding_results),
            "worldmodeling_count": len(worldmodeling_results),
            "grounding_accuracy": grounding_accuracy,
            "worldmodeling_accuracy": worldmodeling_accuracy,
            "parse_success_rate": parse_success_rate,
        }, step=global_step)
        
        # Create wandb tables for examples
        
        # Define common columns for all tables
        columns = ["id", "env_name", "prompt", "response", "parsed_answer"]
        
        # Split results by category
        correct_grounding = [r for r in grounding_results if r["success"] and r["score"] == 1.0]
        incorrect_grounding = [r for r in grounding_results if r["success"] and r["score"] == 0.0]
        correct_worldmodeling = [r for r in worldmodeling_results if r["success"] and r["score"] == 1.0]
        incorrect_worldmodeling = [r for r in worldmodeling_results if r["success"] and r["score"] == 0.0]
        parse_failed = [r for r in results if r["success"] and 
                       not re.search(r'<answer>(YES|NO)</answer>', r["response"], re.IGNORECASE)]
        
        # Function to extract answer from response
        def extract_parsed_answer(response):
            match = re.search(r'<answer>(YES|NO)</answer>', response, re.IGNORECASE)
            return match.group(1) if match else "PARSE_FAILED"
        
        # Function to sample and prepare table data
        def prepare_table_data(results_subset, max_samples):
            if not results_subset:
                return []
                
            # Sample results (or take all if fewer than max_samples)
            samples = random.sample(results_subset, min(max_samples, len(results_subset)))
            
            # Prepare data for table
            return [
                [
                    r["id"],
                    r["env_name"],
                    r["prompt"],
                    r["response"],
                    extract_parsed_answer(r["response"])
                ]
                for r in samples
            ]
        
        # Create and log tables
        correct_grounding_table = wandb.Table(
            columns=columns,
            data=prepare_table_data(correct_grounding, correct_grounding_samples)
        )
        
        incorrect_grounding_table = wandb.Table(
            columns=columns,
            data=prepare_table_data(incorrect_grounding, incorrect_grounding_samples)
        )
        
        correct_worldmodeling_table = wandb.Table(
            columns=columns,
            data=prepare_table_data(correct_worldmodeling, correct_worldmodeling_samples)
        )
        
        incorrect_worldmodeling_table = wandb.Table(
            columns=columns,
            data=prepare_table_data(incorrect_worldmodeling, incorrect_worldmodeling_samples)
        )
        
        parse_failed_table = wandb.Table(
            columns=columns,
            data=prepare_table_data(parse_failed, 3)  # Sample up to 3 parse failures
        )
        
        # Create table for errors
        error_columns = ["id", "env_name", "type", "error"]
        error_examples = [r for r in results if not r["success"]]
        error_data = [
            [r["id"], r["env_name"], r["type"], r["error"]]
            for r in error_examples[:3]  # Sample up to 3 errors
        ]
        error_table = wandb.Table(columns=error_columns, data=error_data)
        
        # Log all tables with the current step
        wandb.log({
            "correct_grounding_examples": correct_grounding_table,
            "incorrect_grounding_examples": incorrect_grounding_table,
            "correct_worldmodeling_examples": correct_worldmodeling_table,
            "incorrect_worldmodeling_examples": incorrect_worldmodeling_table,
            "parse_failed_examples": parse_failed_table,
            "error_examples": error_table
        }, step=global_step)
        
        return results


def process_llm_judgments(input_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Process a list of LLM judgment inputs and prepare prompts for evaluation.
    
    Args:
        input_data: List of dictionaries containing:
            - id: Unique identifier
            - content: Natural language description
            - state: State information dictionary
            - type: Type of judgment ("grounding" or "worldmodeling")
            - env_name: Environment name
    
    Returns:
        List of dictionaries with judgment results including scores
    """
    # Get the directory containing the current script file
    script_dir = Path(__file__).parent
    
    # Initialize Hydra with the config file if not already initialized
    if not GlobalHydra.instance().is_initialized():
        hydra.initialize(config_path=str(script_dir))
        
    # Load the config - Hydra automatically resolves the ${...} references
    config = hydra.compose(config_name="llm_as_judge")
    
    # Extract model configuration
    model_config = {
        "model": config.model.name,
        "temperature": config.model.temperature,
        "max_tokens": config.model.max_tokens,
        "max_retries": config.api.max_retries,
        "timeout": config.api.request_timeout
    }
    
    # Create prompts for each input
    prompts = []
    metadata = []  # Store additional info we'll need after getting responses
    
    for item in input_data:
        # Get the appropriate prompt template
        prompt_type = item["type"]  # "grounding" or "worldmodeling"
        env_name = item["env_name"]  # e.g., "sokoban"
        
        # Access the prompt template - Hydra has already resolved ${...} references
        template = config.prompt_templates[env_name][prompt_type]
        
        # Format the prompt template with the input data
        formatted_prompt = template.format(
            state_information_dict=item["state"],
            natural_language_description=item["content"],
            max_tokens=config.model.max_tokens
        )
        
        prompts.append(formatted_prompt)
        metadata.append({
            "id": item["id"],
            "type": item["type"],
            "env_name": item["env_name"]
        })
    
    # Call the request function to get LLM responses
    llm_responses = run_together_request(prompts, model_config)
    
    # Process the responses and extract scores
    results = []
    for i, response_data in enumerate(llm_responses):
        # Extract the YES/NO answer
        score = 0.0  # Default score (NO or failure)
        
        if response_data["success"]:
            # Use regex to find the answer tag
            answer_match = re.search(r'<answer>(YES|NO)</answer>', response_data["response"], re.IGNORECASE)
            if answer_match:
                answer = answer_match.group(1).upper()
                score = 1.0 if answer == "YES" else 0.0
        
        # Create the result dictionary
        result = {
            "id": metadata[i]["id"],
            "type": metadata[i]["type"],
            "env_name": metadata[i]["env_name"],
            "prompt": prompts[i],
            "response": response_data["response"],
            "success": response_data["success"],
            "score": score,
            "error": response_data["error"]
        }
        
        results.append(result)
    return results

def run_together_request(prompts: List[str], config: Dict[str, Any] = None) -> List[Dict[str, Any]]:
    """
    Process a list of prompts with Together AI, handling timeouts and retries.
    This function manages async operations internally but returns a normal list.
    
    Args:
        prompts: List of prompt strings to process
        config: Configuration dictionary which may include:
            - timeout: Maximum seconds to wait for all completions (default: 60)
            - max_retries: Maximum number of retries per prompt (default: 3)
            - retry_delay: Seconds to wait between retries (default: 1)
            - model: Model name to use
            - Other model parameters like temperature, max_tokens, etc.
    
    Returns:
        List of dictionaries matching the order of input prompts, each containing:
            - response: The completion text or error message
            - success: Boolean indicating if the request succeeded within timeout
            - retries: Number of retries performed (0 if succeeded on first try)
            - error: Error message if applicable (None if successful)
    """
    # Set default configuration
    default_config = {
        "timeout": 60,
        "max_retries": 3,
        "retry_delay": 1,
        "model": "meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8",
        "temperature": 0.1,
        "max_tokens": 500
    }
    
    # Merge default config with provided config
    if config is None:
        config = {}
    
    config = {**default_config, **config}
    
    # Define the async function to be run within this synchronous function
    async def _async_batch_completions():
        # Initialize async client
        async_client = AsyncTogether()
        
        # Initialize results list with placeholders for each prompt
        results = [{"response": "", "success": False, "retries": 0, "error": None} for _ in prompts]
        
        async def process_prompt(prompt: str, index: int) -> None:
            """Process a single prompt with retries"""
            retries = 0
            
            while retries <= config["max_retries"]:
                try:
                    # Extract model parameters from config
                    model_params = {
                        k: v for k, v in config.items() 
                        if k not in ["timeout", "max_retries", "retry_delay"]
                    }
                    
                    response = await async_client.completions.create(
                        prompt=prompt,
                        **model_params
                    )
                    
                    # Update result for this prompt
                    results[index] = {
                        "response": response.choices[0].text,
                        "success": True,
                        "retries": retries,
                        "error": None
                    }
                    return
                    
                except Exception as e:
                    retries += 1
                    if retries <= config["max_retries"]:
                        await asyncio.sleep(config["retry_delay"])
                    else:
                        # Update result with error info after all retries failed
                        results[index] = {
                            "response": f"Error after {retries} attempts",
                            "success": False,
                            "retries": retries,
                            "error": str(e)
                        }
                        return
        
        # Create tasks for each prompt
        tasks = [process_prompt(prompt, i) for i, prompt in enumerate(prompts)]
        
        try:
            # Run all tasks with timeout
            await asyncio.wait_for(asyncio.gather(*tasks), timeout=config["timeout"])
        except asyncio.TimeoutError:
            # Timeout occurred, some tasks may not have completed
            # Results list already has default "success": False for unfinished tasks
            pass
        
        return results
    
    # Run the async function in an event loop and get the results
    try:
        # Check if we're already in an event loop (for environments that may have one running)
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # Create a new event loop if the current one is already running
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                results = loop.run_until_complete(_async_batch_completions())
                loop.close()
            else:
                results = loop.run_until_complete(_async_batch_completions())
        except RuntimeError:
            # No event loop exists, create one
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            results = loop.run_until_complete(_async_batch_completions())
            loop.close()
    except Exception as e:
        # Handle any unexpected errors in the async execution
        return [{"response": f"Global error: {str(e)}", "success": False, "retries": 0, "error": str(e)} 
                for _ in prompts]
    
    return results