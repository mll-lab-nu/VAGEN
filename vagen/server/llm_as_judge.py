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
from vagen.server.gpt_batch_request import run_gpt_request
# Global variables for wandb tracking per process
_WANDB_INITIALIZED = {}  # Track initialization status per process
_GLOBAL_STEPS = {}  # Track global step count per process
_PROCESS_LOCKS = {}  # Semaphore for each process
_HYDRA_LOCKS = {}  # Semaphore for Hydra initialization
_HYDRA_INITIALIZED = {}  # Track Hydra initialization per process
_PID_CONFIG= {}  # Store config per process
# Store wandb tables per process
_WANDB_TABLES = {}  # Store wandb tables for each process with structure:
# {pid: {
#   'correct_grounding_table': wandb.Table,
#   'incorrect_grounding_table': wandb.Table,
#   'correct_worldmodeling_table': wandb.Table,
#   'incorrect_worldmodeling_table': wandb.Table,
#   'parse_failed_table': wandb.Table
# }}
# Context manager to ensure proper cleanup of wandb sessions
@contextmanager
def wandb_run_context():
    """Context manager for wandb runs that ensures proper cleanup"""
    try:
        yield
    finally:
        # If wandb is running, finish the run
        if wandb.run is not None:
            # Clear global table storage for this process before finishing
            pid = os.getpid()
            if pid in _WANDB_TABLES:
                _WANDB_TABLES.pop(pid)
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
        
        if pid not in _PID_CONFIG:
            # Load the config for this process
            config = hydra.compose(config_name="llm_as_judge")
            _PID_CONFIG[pid] = config
        else:
            config = _PID_CONFIG[pid]
        return config
            
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
            _GLOBAL_STEPS[pid] = -1
        
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
        wandb_config = wandb.config.wandb
        correct_grounding_samples = wandb_config.get("correct_grounding_samples", 3)
        incorrect_grounding_samples = wandb_config.get("incorrect_grounding_samples", 3)
        correct_worldmodeling_samples = wandb_config.get("correct_worldmodeling_samples", 3)
        incorrect_worldmodeling_samples = wandb_config.get("incorrect_worldmodeling_samples", 3)
        parse_failed_samples = wandb_config.get("parse_failed_samples", 3)
        # Removed error_samples as we no longer log error data to wandb tables
        
        # Get table logging frequency (default to 10 if not specified)
        table_logging_frequency = wandb_config.get("table_logging_frequency", 10)
        
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
        parse_successes = sum(1 for r in results if r["parse_success"] and r["success"])
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
        correct_grounding = [r for r in grounding_results if r["success"] and r["score"] > 1e-6]
        incorrect_grounding = [r for r in grounding_results if r["success"] and r["score"] == 0.0]
        correct_worldmodeling = [r for r in worldmodeling_results if r["success"] and r["score"] > 1e-6]
        incorrect_worldmodeling = [r for r in worldmodeling_results if r["success"] and r["score"] == 0.0]
        parse_failed = [r for r in results if r["success"] and (not r["parse_success"])]
        
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
                List with values in the same order as the columns
            """
            if not results_subset:
                return None
            
            # Sample results (or take all if fewer than max_samples)
            samples = random.sample(results_subset, min(max_samples, len(results_subset)))
            
            # Create a list with values in the same order as the columns
            # First element is the step
            row_data = [global_step]
            
            # For each possible sample index (1 to max_samples)
            for i in range(max_samples):
                # If we have a sample at this index
                if i < len(samples):
                    sample = samples[i]
                    # Add each field value in order
                    row_data.append(sample["id"])
                    row_data.append(sample["env_name"])
                    row_data.append(sample["prompt"])
                    row_data.append(sample["response"])
                    row_data.append(extract_parsed_answer(sample["response"]))
                else:
                    # Add empty values for missing samples to ensure all rows have the same length
                    row_data.extend([""] * 5)  # 5 fields per sample
            
            return row_data

        # Removed prepare_error_table_data function since we no longer log error data to wandb tables

        # Create and log tables with the global step structure
        def log_tables_with_step(global_step):
            # Define columns for each table type
            correct_grounding_columns = ["step"] + [
                f"sample_{i}_{field}" 
                for i in range(1, correct_grounding_samples + 1) 
                for field in ["id", "env_name", "prompt", "response", "parsed_answer"]
            ]
            
            incorrect_grounding_columns = ["step"] + [
                f"sample_{i}_{field}" 
                for i in range(1, incorrect_grounding_samples + 1) 
                for field in ["id", "env_name", "prompt", "response", "parsed_answer"]
            ]
            
            correct_worldmodeling_columns = ["step"] + [
                f"sample_{i}_{field}" 
                for i in range(1, correct_worldmodeling_samples + 1) 
                for field in ["id", "env_name", "prompt", "response", "parsed_answer"]
            ]
            
            incorrect_worldmodeling_columns = ["step"] + [
                f"sample_{i}_{field}" 
                for i in range(1, incorrect_worldmodeling_samples + 1) 
                for field in ["id", "env_name", "prompt", "response", "parsed_answer"]
            ]
            
            parse_failed_columns = ["step"] + [
                f"sample_{i}_{field}" 
                for i in range(1, parse_failed_samples + 1)  # Use config for parse failure samples 
                for field in ["id", "env_name", "prompt", "response", "parsed_answer"]
            ]
            
            # Initialize or retrieve process tables from global storage
            pid = os.getpid()
            if pid not in _WANDB_TABLES:
                _WANDB_TABLES[pid] = {
                    'correct_grounding_table': wandb.Table(columns=correct_grounding_columns),
                    'incorrect_grounding_table': wandb.Table(columns=incorrect_grounding_columns),
                    'correct_worldmodeling_table': wandb.Table(columns=correct_worldmodeling_columns),
                    'incorrect_worldmodeling_table': wandb.Table(columns=incorrect_worldmodeling_columns),
                    'parse_failed_table': wandb.Table(columns=parse_failed_columns)
                }
            
            # Retrieve existing tables from the global storage
            correct_grounding_table = _WANDB_TABLES[pid]['correct_grounding_table']
            incorrect_grounding_table = _WANDB_TABLES[pid]['incorrect_grounding_table']
            correct_worldmodeling_table = _WANDB_TABLES[pid]['correct_worldmodeling_table']
            incorrect_worldmodeling_table = _WANDB_TABLES[pid]['incorrect_worldmodeling_table']
            parse_failed_table = _WANDB_TABLES[pid]['parse_failed_table']
            
            # Prepare data rows for each table (one row per global step)
            correct_grounding_data = prepare_table_data(correct_grounding, correct_grounding_samples, global_step)
            incorrect_grounding_data = prepare_table_data(incorrect_grounding, incorrect_grounding_samples, global_step)
            correct_worldmodeling_data = prepare_table_data(correct_worldmodeling, correct_worldmodeling_samples, global_step)
            incorrect_worldmodeling_data = prepare_table_data(incorrect_worldmodeling, incorrect_worldmodeling_samples, global_step)
            parse_failed_data = prepare_table_data(parse_failed, parse_failed_samples, global_step)  # Use config parameter
            
            # Add data rows to tables
            if correct_grounding_data:
                correct_grounding_table.add_data(*correct_grounding_data)
            if incorrect_grounding_data:
                incorrect_grounding_table.add_data(*incorrect_grounding_data)
            if correct_worldmodeling_data:
                correct_worldmodeling_table.add_data(*correct_worldmodeling_data)
            if incorrect_worldmodeling_data:
                incorrect_worldmodeling_table.add_data(*incorrect_worldmodeling_data)
            if parse_failed_data:
                parse_failed_table.add_data(*parse_failed_data)
            
            # Create new tables with all the accumulated data for logging
            # This ensures we don't lose the history when logging to wandb
            new_correct_grounding_table = wandb.Table(columns=correct_grounding_columns, data=correct_grounding_table.data)
            new_incorrect_grounding_table = wandb.Table(columns=incorrect_grounding_columns, data=incorrect_grounding_table.data)
            new_correct_worldmodeling_table = wandb.Table(columns=correct_worldmodeling_columns, data=correct_worldmodeling_table.data)
            new_incorrect_worldmodeling_table = wandb.Table(columns=incorrect_worldmodeling_columns, data=incorrect_worldmodeling_table.data)
            new_parse_failed_table = wandb.Table(columns=parse_failed_columns, data=parse_failed_table.data)
            
            # Log the tables directly to history without using summary
            wandb.log({
                "correct_grounding_examples": new_correct_grounding_table,
                "incorrect_grounding_examples": new_incorrect_grounding_table,
                "correct_worldmodeling_examples": new_correct_worldmodeling_table,
                "incorrect_worldmodeling_examples": new_incorrect_worldmodeling_table,
                "parse_failed_examples": new_parse_failed_table
            }, step=global_step)

        # Remove error data logging to wandb tables
        
        # Replace the original table logging with a frequency-based approach
        # Only log tables if the current step is divisible by the table logging frequency
        # This ensures we log tables at regular intervals rather than every step
        if global_step % table_logging_frequency == 0:
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
    llm_responses = run_gpt_request(prompts,config.api)
    
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
                parse_success = True
            else:
                parse_success = False
        
        # Create the result dictionary
        result = {
            "id": metadata[i]["id"],
            "type": metadata[i]["type"],
            "env_name": metadata[i]["env_name"],
            "prompt": prompts[i],
            "response": response_data["response"],
            "success": response_data["success"],
            "score": score,
            "error": response_data["error"],
            "parse_success": parse_success
        }
        
        results.append(result)
    return results