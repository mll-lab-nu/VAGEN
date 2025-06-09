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
from vagen.env.utils.parse_json_utils import parse_llm_json_response_flexible

# Global variables for wandb tracking per process
_WANDB_INITIALIZED = {}  # Track initialization status per process
_GLOBAL_STEPS = {}  # Track global step count per process
_PROCESS_LOCKS = {}  # Semaphore for each process
_HYDRA_LOCKS = {}  # Semaphore for Hydra initialization
_HYDRA_INITIALIZED = {}  # Track Hydra initialization per process
_PID_CONFIG= {}  # Store config per process
# Store wandb tables per process
_WANDB_TABLES = {}  # Store wandb tables for each process

# Context manager to ensure proper cleanup of wandb sessions
@contextmanager
def wandb_run_context(pid: int, wandb_tables: Dict[str, Any]):
    """
    Context manager for wandb runs that ensures proper cleanup
    
    Args:
        pid: Process ID
        wandb_tables: Dictionary storing wandb tables
    """
    try:
        yield
    finally:
        # If wandb is running, finish the run
        if wandb.run is not None:
            # Clear table storage for this process before finishing
            if pid in wandb_tables:
                wandb_tables.pop(pid)
            wandb.finish()

def _get_hydra_config(pid: int, 
                     hydra_locks: Dict[int, threading.Lock],
                     hydra_initialized: Dict[int, bool],
                     pid_config: Dict[int, DictConfig]) -> DictConfig:
    """
    Get Hydra configuration in a thread-safe and process-safe manner.
    
    Args:
        pid: Process ID
        hydra_locks: Dictionary of locks for each process
        hydra_initialized: Dictionary tracking Hydra initialization status
        pid_config: Dictionary storing config for each process
        
    Returns:
        Hydra configuration
    """
    # Create a lock for this process if it doesn't exist
    if pid not in hydra_locks:
        hydra_locks[pid] = threading.Lock()
    
    # Use the lock to ensure thread safety within the process
    with hydra_locks[pid]:
        # Check if Hydra is already initialized for this process
        if pid not in hydra_initialized or not hydra_initialized[pid]:
            # Check if Hydra is globally initialized and reset if needed
            if GlobalHydra.instance().is_initialized():
                GlobalHydra.instance().clear()
                
            # Initialize Hydra with the config file
            # Use a relative path for config_path
            hydra.initialize(config_path="config")
            
            # Mark as initialized for this process
            hydra_initialized[pid] = True
        
        if pid not in pid_config:
            # Load the config for this process
            config = hydra.compose(config_name="llm_as_judge")
            pid_config[pid] = config
        else:
            config = pid_config[pid]
        return config

def extract_parsed_answer(response: str) -> str:
    """Extract YES/NO answer from response text."""
    match = re.search(r'<answer>(YES|NO)</answer>', response, re.IGNORECASE)
    return match.group(1) if match else "PARSE_FAILED"

def prepare_table_data(results_subset: List[Dict[str, Any]], max_samples: int, global_step: int) -> Optional[List[Any]]:
    """
    Prepare data for wandb table where each row represents a global step
    with columns for each sample and its fields.
    
    Args:
        results_subset: List of result dictionaries
        max_samples: Maximum number of samples to include
        global_step: Current global step
        
    Returns:
        List with values in the same order as the columns, or None if no results
    """
    if not results_subset:
        return None
    
    # Sample results (or take all if fewer than max_samples)
    samples = random.sample(results_subset, min(max_samples, len(results_subset)))
    
    # Create a list with values in the same order as the columns
    row_data = [global_step]
    
    # For each possible sample index
    for i in range(max_samples):
        if i < len(samples):
            sample = samples[i]
            row_data.extend([
                sample["id"],
                sample["env_name"],
                sample["prompt"],
                sample["response"],
                extract_parsed_answer(sample["response"])
            ])
        else:
            # Add empty values for missing samples
            row_data.extend([""] * 5)  # 5 fields per sample
    
    return row_data

def create_data_categories(wandb_config: Any) -> Dict[str, Dict[str, Any]]:
    """
    Create the data categories dictionary with configuration from wandb config.
    
    Args:
        wandb_config: WandB configuration object
        
    Returns:
        Dictionary of data categories with their configuration
    """
    return {
        "correct_grounding": {
            "samples": wandb_config.get("correct_grounding_samples", 3),
            "filter": lambda r: r["type"] == "grounding" and r["success"] and r["parse_success"],
            "data": []
        },
        "incorrect_grounding": {
            "samples": wandb_config.get("incorrect_grounding_samples", 3),
            "filter": lambda r: r["type"] == "grounding" and r["success"] and not r["parse_success"],
            "data": []
        },
        "correct_worldmodeling": {
            "samples": wandb_config.get("correct_worldmodeling_samples", 3),
            "filter": lambda r: r["type"] == "worldmodeling" and r["success"] and r["parse_success"],
            "data": []
        },
        "incorrect_worldmodeling": {
            "samples": wandb_config.get("incorrect_worldmodeling_samples", 3),
            "filter": lambda r: r["type"] == "worldmodeling" and r["success"] and not r["parse_success"],
            "data": []
        },
        "parse_failed": {
            "samples": wandb_config.get("parse_failed_samples", 3),
            "filter": lambda r: r["success"] and not r["parse_success"],
            "data": []
        }
    }

def calculate_metrics(results: List[Dict[str, Any]]) -> Dict[str, float]:
    """
    Calculate various metrics from the results.
    
    Args:
        results: List of result dictionaries
        
    Returns:
        Dictionary containing calculated metrics
    """
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
    
    return {
        "total_requests": total_requests,
        "completed_requests": completed_requests,
        "completion_rate": completed_requests / total_requests if total_requests > 0 else 0,
        "overall_accuracy": overall_accuracy,
        "grounding_count": len(grounding_results),
        "worldmodeling_count": len(worldmodeling_results),
        "grounding_accuracy": grounding_accuracy,
        "worldmodeling_accuracy": worldmodeling_accuracy,
        "parse_success_rate": parse_success_rate,
    }

def filter_results_by_category(results: List[Dict[str, Any]], data_categories: Dict[str, Dict[str, Any]]) -> None:
    """
    Filter results into categories based on their filter functions.
    
    Args:
        results: List of result dictionaries
        data_categories: Dictionary of data categories with filter functions
    """
    for category_name, category_info in data_categories.items():
        category_info["data"] = [r for r in results if category_info["filter"](r)]

def log_tables_with_step(global_step: int, data_categories: Dict[str, Dict[str, Any]], wandb_tables: Dict[str, Any]) -> None:
    """
    Create and log wandb tables for all data categories.
    
    Args:
        global_step: Current global step
        data_categories: Dictionary of data categories with their data
        wandb_tables: Dictionary to store wandb tables
    """
    # Process each category
    for category_name, category_info in data_categories.items():
        # Define columns for this category
        columns = ["step"] + [
            f"sample_{i}_{field}" 
            for i in range(1, category_info["samples"] + 1) 
            for field in ["id", "env_name", "prompt", "response", "parsed_answer"]
        ]
        
        # Initialize table if not exists
        table_key = f"{category_name}_table"
        if table_key not in wandb_tables:
            wandb_tables[table_key] = wandb.Table(columns=columns)
        
        # Get existing table
        table = wandb_tables[table_key]
        
        # Prepare data row
        data_row = prepare_table_data(
            category_info["data"], 
            category_info["samples"], 
            global_step
        )
        
        # Add data row to table
        if data_row:
            table.add_data(*data_row)
        
        # Create new table with all accumulated data for logging
        new_table = wandb.Table(columns=columns, data=table.data)
        
        # Log the table
        wandb.log({
            f"{category_name}_examples": new_table
        }, step=global_step)

def initialize_wandb_for_process(config: DictConfig) -> None:
    """
    Initialize wandb for the current process.
    
    Args:
        config: Hydra configuration
    """
    run_id = str(uuid.uuid4())[:8]
    wandb.init(
        project=config.wandb.project,
        name=f"{config.wandb.run_name}_{run_id}",
        config=OmegaConf.to_container(config, resolve=True),
    )

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
        config = _get_hydra_config(pid, _HYDRA_LOCKS, _HYDRA_INITIALIZED, _PID_CONFIG)
        
        # Initialize wandb if not already done for this process
        if pid not in _WANDB_INITIALIZED or not _WANDB_INITIALIZED[pid]:
            initialize_wandb_for_process(config)
            _WANDB_INITIALIZED[pid] = True
        
        # Get sampling parameters from wandb config
        wandb_config = wandb.config.wandb
        
        # Create data categories
        data_categories = create_data_categories(wandb_config)
        
        # Get table logging frequency (default to 10 if not specified)
        table_logging_frequency = wandb_config.get("table_logging_frequency", 10)
        
        # Measure execution time
        start_time = time.time()
        
        # Process the judgments
        results = process_llm_judgments(input_data)
        
        # Calculate execution time
        execution_time = time.time() - start_time
        
        # Calculate metrics
        metrics = calculate_metrics(results)
        
        # Log scalar metrics to wandb with step to ensure proper plotting
        wandb.log({
            "global_step": global_step,
            "execution_time": execution_time,
            **metrics
        }, step=global_step)
        
        # Filter results into categories
        filter_results_by_category(results, data_categories)
        
        # Initialize or retrieve process tables from global storage
        if pid not in _WANDB_TABLES:
            _WANDB_TABLES[pid] = {}
        
        # Log tables based on frequency
        if global_step % table_logging_frequency == 0:
            log_tables_with_step(global_step, data_categories, _WANDB_TABLES[pid])
        
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
        config = _get_hydra_config(pid, _HYDRA_LOCKS, _HYDRA_INITIALIZED, _PID_CONFIG)
    
    # Create prompts for each input
    prompts = []
    metadata = []  # Store additional info we'll need after getting responses
    
    for item in input_data:
        prompts.append(item["prompt"])
        metadata.append({
            "id": item["id"],
            "type": item["type"],
            "env_name": item["env_name"]
        })
    
    # Call the request function to get LLM responses
    llm_responses = run_gpt_request(prompts, config.api)
    
    # Process the responses and extract scores
    results = []
    for i, response_data in enumerate(llm_responses):
        # Extract the YES/NO answer
        score = 0.0  # Default score (NO or failure)
        parsed_response = parse_llm_json_response_flexible(response_data["response"])
        parse_success = True if parsed_response else False
        
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
            "parse_success": parse_success,
            "parsed_response": parsed_response
        }
        
        results.append(result)
    return results