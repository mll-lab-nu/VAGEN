from typing import Dict, List, Tuple, Optional, Any, Union
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import os
import numpy as np
import time
import logging
import multiprocessing as mp
from queue import Empty
from functools import partial
import torch
from vagen.env.base.base_service import BaseService
from vagen.server.serial import serialize_observation

from .env import PrimitiveSkillEnv
from .env_config import PrimitiveSkillEnvConfig
from ..base.base_service_config import BaseServiceConfig
from mani_skill.utils.building.articulations.partnet_mobility import _load_partnet_mobility_dataset, PARTNET_MOBILITY
from vagen.env.utils.state_reward_text_utils import service_state_reward_wrapper

class PrimitiveSkillService(BaseService):
    """
    Service class for PrimitiveSkill environments that combines multiprocessing and multithreading.
    This implementation first creates the environments in subprocesses to avoid serialization issues,
    then communicates with these processes via function calls.
    """
    
    def __init__(self, config: BaseServiceConfig):
        """
        Initialize the MultiprocessPrimitiveSkillService.
        
        Args:
            config: Service configuration containing:
                max_process_workers: Maximum number of process workers (default: number of CPU cores)
                max_thread_workers: Maximum number of thread workers per process (default: 4)
                spawn_method: Multiprocessing start method ('fork' or 'spawn', default: 'fork')
                devices: List of GPU device IDs to distribute processes across (default: [0])
        """
        # Get the number of CPU cores available
        cpu_count = os.cpu_count() or 4
        
        # Set default values for process and thread workers
        self.max_process_workers = min(config.get('max_process_workers', cpu_count), cpu_count)
        self.max_thread_workers = config.get('max_thread_workers', 4)
        self.timeout = config.get('timeout', 120)  # Timeout for commands
        # Set multiprocessing start method
        self.spawn_method = config.get('spawn_method', 'fork')
        
        # Get GPU devices and assign them to processes
        self.devices = config.get('devices', [0])
        self.process_to_device = self._assign_devices_to_processes()
        
        # Set up environment mappings
        self.environments = {}  # Mapping from env_id to process_id
        self.env_configs = {}   # Store configurations for reference
        
        # List to track the processes
        self.processes = []
        self.config=config
        
        # Set up logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger('MultiprocessPrimitiveSkillService')
        
        # Initialize communication queues
        self._setup_mp_queues()
    
    def _assign_devices_to_processes(self):
        """
        Assign GPU devices to worker processes in a balanced way.
        
        Returns:
            Dictionary mapping process IDs to GPU device IDs
        """
        process_to_device = {}
        num_devices = len(self.devices)
        
        for pid in range(self.max_process_workers):
            # Assign devices in a round-robin fashion
            device_idx = pid % num_devices
            process_to_device[pid] = self.devices[device_idx]
            
        return process_to_device
    
    def _setup_mp_queues(self):
        """
        Set up multiprocessing queues for communication between main process and workers.
        """
        # Create queues for each worker process
        self.task_queues = []
        self.result_queues = []
        
        for _ in range(self.max_process_workers):
            self.task_queues.append(mp.Queue())
            self.result_queues.append(mp.Queue())
    
    def _worker_process(self, process_id, task_queue, result_queue, max_thread_workers, device_id):
        """
        Worker process function that handles environment operations.
        
        Args:
            process_id: ID of this worker process
            task_queue: Queue for receiving tasks
            result_queue: Queue for sending results
            max_thread_workers: Maximum number of thread workers
            device_id: GPU device ID to use for this process
        """
        # Set CUDA device for this process
        if torch.cuda.is_available():
            try:
                torch.cuda.set_device(device_id)
                result_queue.put((-1, "info", f"Process {process_id} using CUDA device {device_id}"))
            except Exception as e:
                result_queue.put((-1, "error", f"Process {process_id} failed to set CUDA device {device_id}: {str(e)}"))
        else:
            result_queue.put((-1, "warning", f"Process {process_id} cannot set CUDA device: CUDA not available"))
        
        # Dictionary to store environments in this process
        local_environments = {}
        local_env_configs = {}
        
        # Initialize PartNet Mobility dataset with a lock to prevent contention
        dataset_lock = mp.Lock()
        
        with dataset_lock:
            if PARTNET_MOBILITY is None or "model_urdf_paths" not in PARTNET_MOBILITY:
                try:
                    result_queue.put((-1, "info", f"Process {process_id} is loading PartNet Mobility dataset"))
                    _load_partnet_mobility_dataset()
                    result_queue.put((-1, "info", f"Process {process_id} finished loading PartNet Mobility dataset"))
                except Exception as e:
                    logging.error(f"Process {process_id} failed to load dataset: {str(e)}, artnet Mobility dataset not found. Download it by running python -m mani_skill.utils.download_asset partnet_mobility_cabinet")
                    result_queue.put((-1, "error", f"Process {process_id} failed to load dataset: {str(e)}"))
                    return  # Exit the process if we can't load the essential dataset
        
        # Create thread pool for local thread-based parallelism
        thread_pool = ThreadPoolExecutor(max_workers=max_thread_workers)
        
        # Main worker loop
        running = True
        while running:
            try:
                # Get task from queue
                command, task_id, args = task_queue.get()
                
                if command == "create":
                    # Create a new environment
                    env_id, config = args
                    
                    try:
                        # Verify environment type
                        env_name = config.get('env_name', 'primitive_skill')
                        if env_name != 'primitive_skill':
                            result_queue.put((task_id, "error", f"Expected environment type 'primitive_skill', got '{env_name}'"))
                            continue
                        
                        # Create environment config
                        env_config_dict = config.get('env_config', {})
                        env_config = PrimitiveSkillEnvConfig(**env_config_dict)
                        
                        # Create environment
                        env = PrimitiveSkillEnv(env_config)
                        
                        # Store locally
                        local_environments[env_id] = env
                        local_env_configs[env_id] = env_config
                        
                        result_queue.put((task_id, "success", env_id))
                    except Exception as e:
                        result_queue.put((task_id, "error", f"Error creating environment {env_id}: {str(e)}"))
                
                elif command == "reset":
                    # Reset an environment
                    env_id, seed = args
                    
                    try:
                        if env_id not in local_environments:
                            result_queue.put((task_id, "error", f"Environment {env_id} not found in process {process_id}"))
                            continue
                        
                        env = local_environments[env_id]
                        observation, info = env.reset(seed=seed)
                        serialized_observation = serialize_observation(observation)
                        
                        result_queue.put((task_id, "success", (serialized_observation, info)))
                    except Exception as e:
                        result_queue.put((task_id, "error", f"Error resetting environment {env_id}: {str(e)}"))
                
                elif command == "step":
                    # Step an environment
                    env_id, action = args
                    
                    try:
                        if env_id not in local_environments:
                            result_queue.put((task_id, "error", f"Environment {env_id} not found in process {process_id}"))
                            continue
                        
                        env = local_environments[env_id]
                        observation, reward, done, info = env.step(action)
                        serialized_observation = serialize_observation(observation)
                        
                        result_queue.put((task_id, "success", (serialized_observation, reward, done, info)))
                    except Exception as e:
                        result_queue.put((task_id, "error", f"Error stepping environment {env_id}: {str(e)}"))
                
                elif command == "compute_reward":
                    # Compute reward for an environment
                    env_id = args
                    
                    try:
                        if env_id not in local_environments:
                            result_queue.put((task_id, "error", f"Environment {env_id} not found in process {process_id}"))
                            continue
                        
                        env = local_environments[env_id]
                        reward = env.compute_reward()
                        
                        result_queue.put((task_id, "success", reward))
                    except Exception as e:
                        result_queue.put((task_id, "error", f"Error computing reward for environment {env_id}: {str(e)}"))
                
                elif command == "system_prompt":
                    # Get system prompt for an environment
                    env_id = args
                    
                    try:
                        if env_id not in local_environments:
                            result_queue.put((task_id, "error", f"Environment {env_id} not found in process {process_id}"))
                            continue
                        
                        env = local_environments[env_id]
                        prompt = env.system_prompt()
                        
                        result_queue.put((task_id, "success", prompt))
                    except Exception as e:
                        result_queue.put((task_id, "error", f"Error getting system prompt for environment {env_id}: {str(e)}"))
                
                elif command == "close":
                    # Close an environment
                    env_id = args
                    
                    try:
                        if env_id not in local_environments:
                            result_queue.put((task_id, "error", f"Environment {env_id} not found in process {process_id}"))
                            continue
                        
                        env = local_environments[env_id]
                        env.close()
                        
                        # Remove from local storage
                        local_environments.pop(env_id, None)
                        local_env_configs.pop(env_id, None)
                        
                        result_queue.put((task_id, "success", True))
                    except Exception as e:
                        result_queue.put((task_id, "error", f"Error closing environment {env_id}: {str(e)}"))
                
                elif command == "exit":
                    # Exit worker process
                    running = False
                    result_queue.put((task_id, "success", "Worker exiting"))
                
                else:
                    # Unknown command
                    result_queue.put((task_id, "error", f"Unknown command: {command}"))
            
            except Exception as e:
                # Handle any unexpected exceptions
                try:
                    result_queue.put((-1, "error", f"Worker process exception: {str(e)}"))
                except:
                    pass
        
        # Clean up before exiting
        thread_pool.shutdown()
        for env_id, env in local_environments.items():
            try:
                env.close()
            except:
                pass
    
    def _start_worker_processes(self):
        """
        Start worker processes that will handle environment operations.
        """
        for i in range(self.max_process_workers):
            # Get the device ID for this process
            device_id = self.process_to_device[i]
            
            # Create and start process
            p = mp.Process(
                target=self._worker_process,
                args=(i, self.task_queues[i], self.result_queues[i], self.max_thread_workers, device_id),
                daemon=True
            )
            p.start()
            self.processes.append(p)
            
            self.logger.info(f"Started worker process {i} with PID {p.pid} on GPU device {device_id}")
    
    def _assign_to_process(self, env_id):
        """
        Assign an environment to a process using a simple round-robin strategy.
        
        Args:
            env_id: Environment ID to assign
            
        Returns:
            Process ID that the environment is assigned to
        """
        # Get the number of environments assigned to each process
        process_loads = [0] * self.max_process_workers
        for pid in self.environments.values():
            process_loads[pid] += 1
        
        # Assign to the process with the lowest load
        target_pid = process_loads.index(min(process_loads))
        return target_pid
    
    def _send_command(self, process_id, command, env_id, args):
        """
        Send a command to a worker process and wait for the result.
        
        Args:
            process_id: Process ID to send the command to
            command: Command to execute
            env_id: Environment ID
            args: Command arguments
            
        Returns:
            Command result
        """
        # Generate a unique task ID
        task_id = hash(f"{command}_{env_id}_{time.time()}")
        
        # Send command to process
        self.task_queues[process_id].put((command, task_id, args))
        
        # Wait for result
        while True:
            try:
                result_task_id, status, result = self.result_queues[process_id].get(timeout=self.timeout)
                if result_task_id == task_id:
                    if status == "success":
                        return result
                    else:
                        raise Exception(f"Command {command} failed: {result}")
            except Empty:
                # Timeout waiting for result
                raise Exception(f"Timeout waiting for result of command {command} for environment {env_id}")
    
    def create_environments_batch(self, ids2configs: Dict[Any, Any]) -> None:
        """
        Create multiple PrimitiveSkill environments distributed across processes,
        ensuring environments are created sequentially to prevent conflicts.
        
        Args:
            ids2configs: A dictionary where each key is an environment ID and the corresponding
                        value is the configuration for that environment.
        """
        # Start worker processes if not already started
        if not self.processes:
            self._start_worker_processes()
        
        # Create a lock to ensure only one environment is being created at a time
        creation_lock = threading.Lock()
        
        # Function to create a single environment in a worker process
        def create_env_in_process(env_id, config):
            # Acquire lock before creating environment
            with creation_lock:
                self.logger.info(f"Starting creation of environment {env_id}")
                
                # Assign to a process
                process_id = self._assign_to_process(env_id)
                
                try:
                    # Send create command to process
                    result = self._send_command(process_id, "create", env_id, (env_id, config))
                    
                    # Record assignment and configuration
                    self.environments[env_id] = process_id
                    self.env_configs[env_id] = config.get('env_config', {})
                    
                    self.logger.info(f"Successfully created environment {env_id}")
                    return env_id, True
                except Exception as e:
                    self.logger.error(f"Failed to create environment {env_id}: {str(e)}")
                    return env_id, False
        
        # Use ThreadPoolExecutor, but with just 1 worker to ensure sequential creation
        with ThreadPoolExecutor(max_workers=1) as executor:
            # Submit all environment creation tasks
            futures = [
                executor.submit(create_env_in_process, env_id, config)
                for env_id, config in ids2configs.items()
            ]
            
            # Wait for all tasks to complete
            for future in as_completed(futures):
                future.result()
    
    def reset_batch(self, ids2seeds: Dict[Any, Any]) -> Dict[Any, Tuple[Any, Any]]:
        """
        Reset multiple PrimitiveSkill environments distributed across processes.
        
        Args:
            ids2seeds: A dictionary where each key is an environment ID and the corresponding
                     value is a seed value.
            
        Returns:
            A dictionary mapping environment IDs to tuples of the form (observation, info)
        """
        results = {}
        
        # Group environments by process
        process_envs = {}
        for env_id, seed in ids2seeds.items():
            if env_id in self.environments:
                process_id = self.environments[env_id]
                if process_id not in process_envs:
                    process_envs[process_id] = []
                process_envs[process_id].append((env_id, seed))
        
        # Function to reset environments in a process
        def reset_envs_in_process(process_id, env_seed_pairs):
            local_results = {}
            
            for env_id, seed in env_seed_pairs:
                try:
                    # Send reset command to process
                    result = self._send_command(process_id, "reset", env_id, (env_id, seed))
                    local_results[env_id] = result
                except Exception as e:
                    self.logger.error(f"Failed to reset environment {env_id}: {str(e)}")
                    local_results[env_id] = ({}, {"error": str(e)})
            
            return local_results
        
        # Use ThreadPoolExecutor for parallel reset requests
        with ThreadPoolExecutor(max_workers=self.max_process_workers) as executor:
            # Submit reset tasks grouped by process
            futures = [
                executor.submit(reset_envs_in_process, process_id, env_seed_pairs)
                for process_id, env_seed_pairs in process_envs.items()
            ]
            
            # Collect results
            for future in as_completed(futures):
                local_results = future.result()
                results.update(local_results)
        
        return results
    
    @service_state_reward_wrapper
    def step_batch(self, ids2actions: Dict[Any, Any]) -> Dict[Any, Tuple[Dict, float, bool, Dict]]:
        """
        Take a step in multiple PrimitiveSkill environments distributed across processes.
        
        Args:
            ids2actions: A dictionary where each key is an environment ID and the corresponding
                       value is the action to execute in that environment.
            
        Returns:
            A dictionary mapping environment IDs to tuples of the form (observation, reward, done, info)
        """
        results = {}
        
        # Group environments by process
        process_envs = {}
        for env_id, action in ids2actions.items():
            if env_id in self.environments:
                process_id = self.environments[env_id]
                if process_id not in process_envs:
                    process_envs[process_id] = []
                process_envs[process_id].append((env_id, action))
        
        # Function to step environments in a process
        def step_envs_in_process(process_id, env_action_pairs):
            local_results = {}
            
            for env_id, action in env_action_pairs:
                try:
                    # Send step command to process
                    result = self._send_command(process_id, "step", env_id, (env_id, action))
                    local_results[env_id] = result
                except Exception as e:
                    self.logger.error(f"Failed to step environment {env_id}: {str(e)}")
                    local_results[env_id] = ({}, 0.0, True, {"error": str(e)})
            
            return local_results
        
        # Use ThreadPoolExecutor for parallel step requests
        with ThreadPoolExecutor(max_workers=self.max_process_workers) as executor:
            # Submit step tasks grouped by process
            futures = [
                executor.submit(step_envs_in_process, process_id, env_action_pairs)
                for process_id, env_action_pairs in process_envs.items()
            ]
            
            # Collect results
            for future in as_completed(futures):
                local_results = future.result()
                results.update(local_results)
        
        return results
    
    def compute_reward_batch(self, env_ids: List[str]) -> Dict[Any, float]:
        """
        Compute the total reward for multiple PrimitiveSkill environments distributed across processes.
        
        Args:
            env_ids: A list of environment IDs
            
        Returns:
            A dictionary mapping each environment ID to its computed total reward
        """
        results = {}
        
        # Group environments by process
        process_envs = {}
        for env_id in env_ids:
            if env_id in self.environments:
                process_id = self.environments[env_id]
                if process_id not in process_envs:
                    process_envs[process_id] = []
                process_envs[process_id].append(env_id)
        
        # Function to compute rewards for environments in a process
        def compute_rewards_in_process(process_id, process_env_ids):
            local_results = {}
            
            for env_id in process_env_ids:
                try:
                    # Send compute_reward command to process
                    result = self._send_command(process_id, "compute_reward", env_id, env_id)
                    local_results[env_id] = result
                except Exception as e:
                    self.logger.error(f"Failed to compute reward for environment {env_id}: {str(e)}")
                    local_results[env_id] = 0.0
            
            return local_results
        
        # Use ThreadPoolExecutor for parallel reward computation requests
        with ThreadPoolExecutor(max_workers=self.max_process_workers) as executor:
            # Submit computation tasks grouped by process
            futures = [
                executor.submit(compute_rewards_in_process, process_id, process_env_ids)
                for process_id, process_env_ids in process_envs.items()
            ]
            
            # Collect results
            for future in as_completed(futures):
                local_results = future.result()
                results.update(local_results)
        
        return results
        
    def get_system_prompts_batch(self, env_ids: List[str]) -> Dict[Any, str]:
        """
        Get system prompts for multiple PrimitiveSkill environments distributed across processes.
        
        Args:
            env_ids: A list of environment IDs
            
        Returns:
            A dictionary mapping each environment ID to its corresponding system prompt string
        """
        results = {}
        
        # Group environments by process
        process_envs = {}
        for env_id in env_ids:
            if env_id in self.environments:
                process_id = self.environments[env_id]
                if process_id not in process_envs:
                    process_envs[process_id] = []
                process_envs[process_id].append(env_id)
        
        # Function to get system prompts for environments in a process
        def get_system_prompts_in_process(process_id, process_env_ids):
            local_results = {}
            
            for env_id in process_env_ids:
                try:
                    # Send system_prompt command to process
                    result = self._send_command(process_id, "system_prompt", env_id, env_id)
                    local_results[env_id] = result
                except Exception as e:
                    self.logger.error(f"Failed to get system prompt for environment {env_id}: {str(e)}")
                    local_results[env_id] = ""
            
            return local_results
        
        # Use ThreadPoolExecutor for parallel system prompt requests
        with ThreadPoolExecutor(max_workers=self.max_process_workers) as executor:
            # Submit system prompt tasks grouped by process
            futures = [
                executor.submit(get_system_prompts_in_process, process_id, process_env_ids)
                for process_id, process_env_ids in process_envs.items()
            ]
            
            # Collect results
            for future in as_completed(futures):
                local_results = future.result()
                results.update(local_results)
        
        return results
    
    def close_batch(self, env_ids: Optional[List[str]] = None) -> None:
        """
        Close multiple PrimitiveSkill environments and clean up resources across processes.
        
        Args:
            env_ids: Optional list of environment IDs to close. If None, close all environments.
        """
        # If no env_ids provided, close all environments
        if env_ids is None:
            env_ids = list(self.environments.keys())
        
        # Group environments by process
        process_envs = {}
        for env_id in env_ids:
            if env_id in self.environments:
                process_id = self.environments[env_id]
                if process_id not in process_envs:
                    process_envs[process_id] = []
                process_envs[process_id].append(env_id)
        
        # Function to close environments in a process
        def close_envs_in_process(process_id, process_env_ids):
            for env_id in process_env_ids:
                try:
                    # Send close command to process
                    self._send_command(process_id, "close", env_id, env_id)
                except Exception as e:
                    self.logger.error(f"Failed to close environment {env_id}: {str(e)}")
        
        # Use ThreadPoolExecutor for parallel close requests
        with ThreadPoolExecutor(max_workers=self.max_process_workers) as executor:
            # Submit close tasks grouped by process
            futures = [
                executor.submit(close_envs_in_process, process_id, process_env_ids)
                for process_id, process_env_ids in process_envs.items()
            ]
            
            # Wait for all tasks to complete
            for future in as_completed(futures):
                future.result()
        
        # Remove closed environments from dictionaries
        for env_id in env_ids:
            self.environments.pop(env_id, None)
            self.env_configs.pop(env_id, None)
    
    def __del__(self):
        """
        Clean up resources when the service is destroyed.
        """
        # Close all environments
        try:
            self.close_batch()
        except:
            pass
        
        # Send exit command to all worker processes
        for i, pipe in enumerate(self.task_queues):
            try:
                pipe.put(("exit", -1, None))
            except:
                pass
        
        # Wait for processes to exit
        for p in self.processes:
            try:
                p.join(timeout=self.timeout)
                if p.is_alive():
                    p.terminate()
            except:
                pass