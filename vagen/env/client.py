from typing import Dict, List, Tuple, Optional, Any, Union
import requests
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

class BatchEnvClient:
    """
    Client for interacting with the batch environment server.
    This client provides methods to create, reset, step and close environments remotely.
    """
    
    def __init__(self, base_url: str, timeout: int = 60, max_workers: int = 10, 
                 logger: Optional[logging.Logger] = None):
        """
        Initialize the BatchEnvClient.
        
        Args:
            base_url: Base URL of the environment server
            timeout: Timeout for HTTP requests in seconds
            max_workers: Maximum number of worker threads for parallel requests
            logger: Optional logger for client logs
        """
        self.base_url = base_url.rstrip('/')
        self.timeout = timeout
        self.max_workers = max_workers
        self.logger = logger or logging.getLogger(self.__class__.__name__)
        self.env_configs = {}  # Store configs for each environment for reference
        
    def _make_request(self, endpoint: str, method: str = "POST", data: Any = None, 
                      env_id: Optional[str] = None) -> Any:
        """
        Make an HTTP request to the environment server.
        
        Args:
            endpoint: API endpoint to call
            method: HTTP method (GET, POST, etc.)
            data: Data to send with the request
            env_id: Optional environment ID to include in the URL
            
        Returns:
            Response data from the server
            
        Raises:
            ConnectionError: If the request fails
        """
        url = f"{self.base_url}/{endpoint}"
        if env_id:
            url = f"{url}/{env_id}"
            
        headers = {"Content-Type": "application/json"}
        
        try:
            if method.upper() == "GET":
                response = requests.get(url, headers=headers, timeout=self.timeout)
            elif method.upper() == "POST":
                response = requests.post(url, headers=headers, json=data, timeout=self.timeout)
            elif method.upper() == "DELETE":
                response = requests.delete(url, headers=headers, json=data, timeout=self.timeout)
            else:
                raise ValueError(f"Unsupported HTTP method: {method}")
                
            response.raise_for_status()  # Raise an exception for 4XX/5XX responses
            return response.json()
            
        except requests.exceptions.RequestException as e:
            self.logger.error(f"Request to {url} failed: {str(e)}")
            raise ConnectionError(f"Failed to communicate with environment server: {str(e)}")
    
    def _process_batch_request(self, endpoint: str, items: List, data_key: str, 
                               method: str = "POST") -> List:
        """
        Process a batch of requests in parallel using a thread pool.
        
        Args:
            endpoint: API endpoint to call
            items: List of items to process (environment IDs or other data)
            data_key: Key for the data in the request
            method: HTTP method for the request
            
        Returns:
            List of results, one for each input item
        """
        results = [None] * len(items)
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = []
            
            for i, item in enumerate(items):
                if isinstance(item, tuple) and len(item) == 2:
                    # Handle case where we have (env_id, data) pairs
                    env_id, data = item
                    future = executor.submit(
                        self._make_request, 
                        endpoint, 
                        method, 
                        {data_key: data}, 
                        env_id
                    )
                else:
                    # Handle case where we just have env_ids
                    future = executor.submit(
                        self._make_request,
                        endpoint,
                        method,
                        None,
                        item
                    )
                    
                futures.append((i, future))
            
            for i, future in futures:
                try:
                    results[i] = future.result()
                except Exception as e:
                    self.logger.error(f"Error processing request for item {i}: {str(e)}")
                    results[i] = {"error": str(e)}
                    
        return results
    
    def check_server_health(self) -> Dict[str, Any]:
        """
        Check the health of the server.
        
        Returns:
            Health status information
        """
        try:
            return self._make_request("health", method="GET")
        except Exception as e:
            self.logger.error(f"Health check failed: {str(e)}")
            return {"status": "error", "message": str(e)}
    
    def wait_for_server(self, max_retries: int = 10, retry_delay: float = 1.0) -> bool:
        """
        Wait for the server to become available.
        
        Args:
            max_retries: Maximum number of retries
            retry_delay: Delay between retries in seconds
            
        Returns:
            True if server is available, False otherwise
        """
        for i in range(max_retries):
            try:
                health = self.check_server_health()
                if health.get("status") == "ok":
                    self.logger.info(f"Server available at {self.base_url}")
                    return True
            except Exception:
                pass
                
            self.logger.info(f"Waiting for server (attempt {i+1}/{max_retries})...")
            time.sleep(retry_delay)
            
        self.logger.error(f"Server not available after {max_retries} attempts")
        return False
        
    def create_environments(self, env_configs: List[Dict[str, Any]]) -> List[str]:
        """
        Create multiple environments based on the provided configurations.
        
        Args:
            env_configs: List of environment configurations
            
        Returns:
            List of environment IDs that can be used to reference these environments
        """
        response = self._make_request("environments", "POST", {"configs": env_configs})
        env_ids = response.get("env_ids", [])
        
        # Store the configs for reference
        for i, env_id in enumerate(env_ids):
            self.env_configs[env_id] = env_configs[i]
            
        return env_ids
    
    def reset_batch(self, env_ids: List[str], seeds: Optional[List[int]] = None) -> List[Tuple[Dict, Dict]]:
        """
        Reset multiple environments simultaneously.
        
        Args:
            env_ids: List of environment IDs to reset
            seeds: Optional list of seeds for resetting environments (one per environment)
            
        Returns:
            List of (observation, info) tuples, one for each environment
        """
        # Prepare data for each environment
        data = []
        for i, env_id in enumerate(env_ids):
            seed = seeds[i] if seeds and i < len(seeds) else None
            data.append((env_id, {"seed": seed} if seed is not None else {}))
            
        # Make parallel requests
        results = self._process_batch_request("reset", data, "reset_params")
        
        # Process results
        reset_results = []
        for result in results:
            if "error" in result:
                # Handle error case
                reset_results.append(({}, {"error": result["error"]}))
            else:
                # Extract observation and info
                obs = result.get("observation", {})
                info = result.get("info", {})
                reset_results.append((obs, info))
                
        return reset_results
    
    def step_batch(self, env_ids: List[str], actions: List[str]) -> List[Tuple[Dict, float, bool, Dict]]:
        """
        Take a step in multiple environments simultaneously.
        
        Args:
            env_ids: List of environment IDs
            actions: List of actions to take in each environment
            
        Returns:
            List of (observation, reward, done, info) tuples, one for each environment
        """
        if len(env_ids) != len(actions):
            raise ValueError("Number of environment IDs must match number of actions")
            
        # Prepare data for each environment
        data = [(env_id, {"action": action}) for env_id, action in zip(env_ids, actions)]
        
        # Make parallel requests
        results = self._process_batch_request("step", data, "step_params")
        
        # Process results
        step_results = []
        for result in results:
            if "error" in result:
                # Handle error case
                step_results.append(({}, 0.0, True, {"error": result["error"]}))
            else:
                # Extract observation, reward, done, and info
                obs = result.get("observation", {})
                reward = result.get("reward", 0.0)
                done = result.get("done", False)
                info = result.get("info", {})
                step_results.append((obs, reward, done, info))
                
        return step_results
    
    def compute_reward_batch(self, env_ids: List[str]) -> List[float]:
        """
        Compute the total reward for multiple environments.
        
        Args:
            env_ids: List of environment IDs
            
        Returns:
            List of rewards, one for each environment
        """
        # Make parallel requests
        results = self._process_batch_request("reward", env_ids, "", "GET")
        
        # Process results
        rewards = []
        for result in results:
            if "error" in result:
                rewards.append(0.0)  # Default to zero reward on error
            else:
                rewards.append(result.get("reward", 0.0))
                
        return rewards
    
    def get_system_prompts_batch(self, env_ids: List[str]) -> List[str]:
        """
        Get system prompts for multiple environments.
        
        Args:
            env_ids: List of environment IDs
            
        Returns:
            List of system prompts, one for each environment
        """
        # Make parallel requests
        results = self._process_batch_request("system_prompt", env_ids, "", "GET")
        
        # Process results
        prompts = []
        for result in results:
            if "error" in result:
                prompts.append("")  # Default to empty prompt on error
            else:
                prompts.append(result.get("system_prompt", ""))
                
        return prompts
    
    def close_batch(self, env_ids: Optional[List[str]] = None) -> None:
        """
        Close multiple environments and clean up resources.
        
        Args:
            env_ids: Optional list of environment IDs to close. If None, close all environments.
        """
        # If no env_ids provided, close all known environments
        if env_ids is None:
            env_ids = list(self.env_configs.keys())
            
        # Make parallel requests
        self._process_batch_request("close", env_ids, "", "DELETE")
        
        # Remove closed environments from tracking
        for env_id in env_ids:
            self.env_configs.pop(env_id, None)


class FrozenLakeBatchClient(BatchEnvClient):
    """
    Client for interacting with FrozenLake environments.
    Extends the BatchEnvClient with FrozenLake-specific operations.
    """
    
    def create_frozen_lake_environments(self, configs: List[Dict[str, Any]]) -> List[str]:
        """
        Create multiple FrozenLake environments with specific configurations.
        
        Args:
            configs: List of FrozenLake configurations
            
        Returns:
            List of environment IDs
        """
        env_configs = []
        
        for config in configs:
            env_config = {
                "env_name": "frozenlake",
                "env_config": config
            }
            env_configs.append(env_config)
            
        return self.create_environments(env_configs)
    
    def get_maps_batch(self, env_ids: List[str]) -> List[Optional[List[List[str]]]]:
        """
        Get maps for multiple FrozenLake environments.
        
        Args:
            env_ids: List of environment IDs
            
        Returns:
            List of maps, or None for environments where the request failed
        """
        # Make parallel requests
        results = self._process_batch_request("map", env_ids, "", "GET")
        
        # Process results
        maps = []
        for result in results:
            if "error" in result:
                maps.append(None)
            else:
                maps.append(result.get("map"))
                
        return maps
    
    def get_player_positions_batch(self, env_ids: List[str]) -> List[Optional[Tuple[int, int]]]:
        """
        Get player positions for multiple FrozenLake environments.
        
        Args:
            env_ids: List of environment IDs
            
        Returns:
            List of player positions, or None for environments where the request failed
        """
        # Make parallel requests
        results = self._process_batch_request("player_position", env_ids, "", "GET")
        
        # Process results
        positions = []
        for result in results:
            if "error" in result:
                positions.append(None)
            else:
                position = result.get("position")
                if position:
                    positions.append(tuple(position))
                else:
                    positions.append(None)
                
        return positions


if __name__ == "__main__":
    # Example usage of the client
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger("BatchEnvClient")
    
    # Create client
    client = FrozenLakeBatchClient(
        base_url="http://localhost:5000",
        timeout=10,
        max_workers=5,
        logger=logger
    )
    
    # Wait for server to be available
    if client.wait_for_server():
        try:
            # Create FrozenLake environments
            configs = [
                {"is_slippery": False, "size": 4, "render_mode": "text"},
                {"is_slippery": True, "size": 8, "render_mode": "vision"}
            ]
            
            logger.info("Creating environments...")
            env_ids = client.create_frozen_lake_environments(configs)
            logger.info(f"Created {len(env_ids)} environments: {env_ids}")
            
            # Reset environments
            logger.info("Resetting environments...")
            seeds = [42, 123]
            obs_infos = client.reset_batch(env_ids, seeds)
            
            # Get system prompts
            logger.info("Getting system prompts...")
            prompts = client.get_system_prompts_batch(env_ids)
            
            # Get maps
            logger.info("Getting maps...")
            maps = client.get_maps_batch(env_ids)
            
            # Step environments
            logger.info("Stepping environments...")
            actions = [
                "<think>Let me try going right first.</think><answer>Right</answer>",
                "<think>I'll start by going down.</think><answer>Down</answer>"
            ]
            results = client.step_batch(env_ids, actions)
            
            # Close environments
            logger.info("Closing environments...")
            client.close_batch(env_ids)
            
            logger.info("Done!")
            
        except Exception as e:
            logger.error(f"Error: {str(e)}")
    else:
        logger.error("Server not available")