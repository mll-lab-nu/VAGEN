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
from vagen.server.together_batch_request import run_together_request
# Global variables for wandb tracking per process
_WANDB_INITIALIZED = {}  # Track initialization status per process
_GLOBAL_STEPS = {}  # Track global step count per process
_PROCESS_LOCKS = {}  # Semaphore for each process
_HYDRA_LOCKS = {}  # Semaphore for Hydra initialization
_HYDRA_INITIALIZED = {}  # Track Hydra initialization per process

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

def _get_hydra_config(pid: int) -> DictConfig:
    """
    Get Hydra configuration in a thread-safe and process-safe manner.
    
    Args:
        pid: Process ID
        
    Returns:
        Hydra configuration
    """
    # Create a lock for this process if it doesn't exist
    if pid not in _HYDRA_LOCKS:
        _HYDRA_LOCKS[pid] = threading.Lock()
    
    # Use the lock to ensure thread safety within the process
    with _HYDRA_LOCKS[pid]:
        # Check if Hydra is already initialized for this process
        if pid not in _HYDRA_INITIALIZED or not _HYDRA_INITIALIZED[pid]:
            # Check if Hydra is globally initialized and reset if needed
            if GlobalHydra.instance().is_initialized():
                GlobalHydra.instance().clear()
                
            # Initialize Hydra with the config file
            # Use a relative path for config_path
            hydra.initialize(config_path="config")
            
            # Mark as initialized for this process
            _HYDRA_INITIALIZED[pid] = True
        
        # Load and return the config
        return hydra.compose(config_name="llm_as_judge")
            
def run_llm_judge(input_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Process input data through the LLM judge and log results to Weights & Biases.
    
    Args:
        input_data: List of dictionaries containing judgment inputs
        
    Returns:
        List of dictionaries with judgment results including scores
    """
    # Skip if no inputs
    if not input_data:
        return []
    
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
        
        # Get Hydra config in a thread-safe and process-safe manner
        config = _get_hydra_config(pid)
        
        # Initialize wandb if not already done for this process
        if pid not in _WANDB_INITIALIZED or not _WANDB_INITIALIZED[pid]:
            # Initialize wandb with values from config
            run_id = str(uuid.uuid4())[:8]
            wandb.init(
                project=config.wandb.project,
                name=f"{config.wandb.run_name}_{run_id}",
                config=OmegaConf.to_container(config, resolve=True),
            )
            
            _WANDB_INITIALIZED[pid] = True
        
        # Get sampling parameters from wandb config
        wandb_config = wandb.config
        correct_grounding_samples = wandb_config.get("correct_grounding_samples", 3)
        incorrect_grounding_samples = wandb_config.get("incorrect_grounding_samples", 3)
        correct_worldmodeling_samples = wandb_config.get("correct_worldmodeling_samples", 3)
        incorrect_worldmodeling_samples = wandb_config.get("incorrect_worldmodeling_samples", 3)
        
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
        
        # Function to prepare table data where each row represents a global step
        # with columns for each sample and its fields
        def prepare_table_data(results_subset, max_samples, global_step):
            """
            Prepare data for wandb table where each row represents a global step
            with columns for each sample and its fields.
            
            Args:
                results_subset: List of result dictionaries
                max_samples: Maximum number of samples to include
                global_step: Current global step
                
            Returns:
                Dictionary with column names as keys and values as a single row
            """
            if not results_subset:
                return {}
            
            # Sample results (or take all if fewer than max_samples)
            samples = random.sample(results_subset, min(max_samples, len(results_subset)))
            
            # Create a dictionary where keys are column names and values are lists
            row_data = {"step": global_step}
            
            # Add columns for each sample and its fields
            for i, sample in enumerate(samples):
                # For each sample, add columns with index for each required field
                sample_idx = i + 1
                row_data[f"sample_{sample_idx}_id"] = sample["id"]
                row_data[f"sample_{sample_idx}_env_name"] = sample["env_name"]
                row_data[f"sample_{sample_idx}_prompt"] = sample["prompt"]
                row_data[f"sample_{sample_idx}_response"] = sample["response"]
                row_data[f"sample_{sample_idx}_parsed_answer"] = extract_parsed_answer(sample["response"])
            
            return row_data

        # Create and update tables with the global step structure
        def log_tables_with_step(global_step):
            # Define tables if they don't exist yet or get existing ones
            if "correct_grounding_table" not in wandb.run.summary:
                # Define columns for each table type
                correct_grounding_columns = ["step"] + [
                    f"sample_{i}_{field}" 
                    for i in range(1, correct_grounding_samples + 1) 
                    for field in ["id", "env_name", "prompt", "response", "parsed_answer"]
                ]
                correct_grounding_table = wandb.Table(columns=correct_grounding_columns)
                
                incorrect_grounding_columns = ["step"] + [
                    f"sample_{i}_{field}" 
                    for i in range(1, incorrect_grounding_samples + 1) 
                    for field in ["id", "env_name", "prompt", "response", "parsed_answer"]
                ]
                incorrect_grounding_table = wandb.Table(columns=incorrect_grounding_columns)
                
                correct_worldmodeling_columns = ["step"] + [
                    f"sample_{i}_{field}" 
                    for i in range(1, correct_worldmodeling_samples + 1) 
                    for field in ["id", "env_name", "prompt", "response", "parsed_answer"]
                ]
                correct_worldmodeling_table = wandb.Table(columns=correct_worldmodeling_columns)
                
                incorrect_worldmodeling_columns = ["step"] + [
                    f"sample_{i}_{field}" 
                    for i in range(1, incorrect_worldmodeling_samples + 1) 
                    for field in ["id", "env_name", "prompt", "response", "parsed_answer"]
                ]
                incorrect_worldmodeling_table = wandb.Table(columns=incorrect_worldmodeling_columns)
                
                parse_failed_columns = ["step"] + [
                    f"sample_{i}_{field}" 
                    for i in range(1, 4)  # Up to 3 parse failure samples 
                    for field in ["id", "env_name", "prompt", "response", "parsed_answer"]
                ]
                parse_failed_table = wandb.Table(columns=parse_failed_columns)
                
                # Initialize tables in wandb
                wandb.run.summary["correct_grounding_table"] = correct_grounding_table
                wandb.run.summary["incorrect_grounding_table"] = incorrect_grounding_table
                wandb.run.summary["correct_worldmodeling_table"] = correct_worldmodeling_table
                wandb.run.summary["incorrect_worldmodeling_table"] = incorrect_worldmodeling_table
                wandb.run.summary["parse_failed_table"] = parse_failed_table
            else:
                # Get existing tables
                correct_grounding_table = wandb.run.summary["correct_grounding_table"]
                incorrect_grounding_table = wandb.run.summary["incorrect_grounding_table"]
                correct_worldmodeling_table = wandb.run.summary["correct_worldmodeling_table"]
                incorrect_worldmodeling_table = wandb.run.summary["incorrect_worldmodeling_table"]
                parse_failed_table = wandb.run.summary["parse_failed_table"]
            
            # Prepare data rows for each table (one row per global step)
            correct_grounding_data = prepare_table_data(correct_grounding, correct_grounding_samples, global_step)
            incorrect_grounding_data = prepare_table_data(incorrect_grounding, incorrect_grounding_samples, global_step)
            correct_worldmodeling_data = prepare_table_data(correct_worldmodeling, correct_worldmodeling_samples, global_step)
            incorrect_worldmodeling_data = prepare_table_data(incorrect_worldmodeling, incorrect_worldmodeling_samples, global_step)
            parse_failed_data = prepare_table_data(parse_failed, 3, global_step)  # Up to 3 parse failures
            
            # Add data rows to tables
            if correct_grounding_data:
                correct_grounding_table.add_data(**correct_grounding_data)
            if incorrect_grounding_data:
                incorrect_grounding_table.add_data(**incorrect_grounding_data)
            if correct_worldmodeling_data:
                correct_worldmodeling_table.add_data(**correct_worldmodeling_data)
            if incorrect_worldmodeling_data:
                incorrect_worldmodeling_table.add_data(**incorrect_worldmodeling_data)
            if parse_failed_data:
                parse_failed_table.add_data(**parse_failed_data)
            
            # Update the tables in wandb
            wandb.run.summary["correct_grounding_table"] = correct_grounding_table
            wandb.run.summary["incorrect_grounding_table"] = incorrect_grounding_table
            wandb.run.summary["correct_worldmodeling_table"] = correct_worldmodeling_table
            wandb.run.summary["incorrect_worldmodeling_table"] = incorrect_worldmodeling_table
            wandb.run.summary["parse_failed_table"] = parse_failed_table
            
            # Also log the tables to the history
            wandb.log({
                "correct_grounding_examples": correct_grounding_table,
                "incorrect_grounding_examples": incorrect_grounding_table,
                "correct_worldmodeling_examples": correct_worldmodeling_table,
                "incorrect_worldmodeling_examples": incorrect_worldmodeling_table,
                "parse_failed_examples": parse_failed_table
            }, step=global_step)

        # Similarly update the error table:
        def prepare_error_table_data(error_examples, max_samples, global_step):
            if not error_examples:
                return {}
            
            samples = error_examples[:max_samples]  # Take up to max_samples errors
            
            row_data = {"step": global_step}
            for i, sample in enumerate(samples):
                sample_idx = i + 1
                row_data[f"error_{sample_idx}_id"] = sample["id"]
                row_data[f"error_{sample_idx}_env_name"] = sample["env_name"]
                row_data[f"error_{sample_idx}_type"] = sample["type"]
                row_data[f"error_{sample_idx}_error"] = sample["error"]
            
            return row_data

        # Process error examples
        error_examples = [r for r in results if not r["success"]]
        
        # Create and update error table
        if "error_table" not in wandb.run.summary:
            error_columns = ["step"] + [
                f"error_{i}_{field}" 
                for i in range(1, 4)  # Up to 3 error samples
                for field in ["id", "env_name", "type", "error"]
            ]
            error_table = wandb.Table(columns=error_columns)
            wandb.run.summary["error_table"] = error_table
        else:
            error_table = wandb.run.summary["error_table"]

        error_data = prepare_error_table_data(error_examples, 3, global_step)  # Sample up to 3 errors
        if error_data:
            error_table.add_data(**error_data)

        wandb.run.summary["error_table"] = error_table
        wandb.log({"error_examples": error_table}, step=global_step)
        
        # Replace the original table creation with the new approach
        log_tables_with_step(global_step)
        
        return results


def process_llm_judgments(input_data: List[Dict[str, Any]], config: Optional[DictConfig] = None) -> List[Dict[str, Any]]:
    """
    Process a list of LLM judgment inputs and prepare prompts for evaluation.
    
    Args:
        input_data: List of dictionaries containing:
            - id: Unique identifier
            - content: Natural language description
            - state: State information dictionary
            - type: Type of judgment ("grounding" or "worldmodeling")
            - env_name: Environment name
        config: Optional configuration object from Hydra. If None, loads config in a thread/process-safe way.
    
    Returns:
        List of dictionaries with judgment results including scores
    """
    # If config is not provided, load it in a thread/process-safe way
    if config is None:
        pid = os.getpid()
        config = _get_hydra_config(pid)
    
    # Extract model configuration from the config
    
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
            max_tokens=config.api.max_tokens
        )
        
        prompts.append(formatted_prompt)
        metadata.append({
            "id": item["id"],
            "type": item["type"],
            "env_name": item["env_name"]
        })
    
    # Call the request function to get LLM responses
    llm_responses = run_together_request(prompts,config.api)
    
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