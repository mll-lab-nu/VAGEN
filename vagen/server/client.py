from typing import Dict, List, Tuple, Optional, Any, Union
import requests
import time
from vagen.server.serial import deserialize_observation, deserialize_step_result

class BatchEnvClient:
    """
    Client for interacting with the batch environment server.
    Uses dictionary-based interface to match the server API and service interface.
    """
    
    def __init__(self, base_url: str, timeout: int = 600, max_workers: int = 10):
        """
        Initialize the BatchEnvClient.
        
        Args:
            base_url: Base URL of the environment server
            timeout: Timeout for HTTP requests in seconds
            max_workers: Maximum number of worker threads for parallel processing
        """
        self.base_url = base_url.rstrip('/')
        self.timeout = timeout
        self.max_workers = max_workers
        self.env_configs = {}  # Store configs for each environment for reference
        
    def _make_request(self, endpoint: str, method: str = "POST", data: Any = None) -> Any:
        """
        Make an HTTP request to the environment server.
        
        Args:
            endpoint: API endpoint to call
            method: HTTP method (GET, POST, etc.)
            data: Data to send with the request
            
        Returns:
            Response data from the server
            
        Raises:
            ConnectionError: If the request fails
        """
        url = f"{self.base_url}/{endpoint}"
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
            
        except Exception as e:
            print(f"Exception in _make_request: {str(e)}")
            raise
    
    def check_server_health(self) -> Dict[str, Any]:
        """
        Check the health of the server.
        
        Returns:
            Health status information
        """
        try:
            return self._make_request("health", method="GET")
        except Exception as e:
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
                    print(f"Server available at {self.base_url}")
                    return True
            except Exception:
                pass
                
            print(f"Waiting for server (attempt {i+1}/{max_retries})...")
            time.sleep(retry_delay)
            
        print(f"Server not available after {max_retries} attempts")
        return False
        
    def create_environments_batch(self, ids2configs: Dict[Any, Any]) -> None:
        """
        Create multiple environments based on the provided configurations.
        Implements BaseService.create_environments_batch interface.
        
        Args:
            ids2configs: Dictionary mapping environment IDs to their configurations
        """
        response = self._make_request("environments", "POST", {"ids2configs": ids2configs})
        if response.get("success") != True:
            raise Exception(f"Failed to create environments: {response.get('error', 'Unknown error')}")
        
        # Store the configs for reference
        for env_id in ids2configs:
            self.env_configs[env_id] = ids2configs[env_id]
    
    def reset_batch(self, ids2seeds: Dict[str, Any]) -> Dict[str, Tuple[Dict, Dict]]:
        """
        Reset multiple environments in batch.
        
        Args:
            ids2seeds: Dictionary mapping environment IDs to seeds
            
        Returns:
            Dictionary mapping environment IDs to (observation, info) tuples
        """
        response = self._make_request("batch/reset", "POST", {"ids2seeds": ids2seeds})
        results = response.get("results", {})
        
        # Deserialize observations
        deserialized_results = {}
        for env_id, (observation, info) in results.items():
            deserialized_results[env_id] = (deserialize_observation(observation), info)
            
        return deserialized_results
    
    def step_batch(self, ids2actions: Dict[str, str]) -> Dict[str, Tuple[Dict, float, bool, Dict]]:
        """
        Step multiple environments in batch.
        
        Args:
            ids2actions: Dictionary mapping environment IDs to actions
            
        Returns:
            Dictionary mapping environment IDs to (observation, reward, done, info) tuples
        """
        response = self._make_request("batch/step", "POST", {"ids2actions": ids2actions})
        results = response.get("results", {})
        
        # Deserialize observations
        deserialized_results = {}
        for env_id, serialized_result  in results.items():
            deserialized_results[env_id] = deserialize_step_result(serialized_result)
            
        return deserialized_results
    
    def compute_reward_batch(self, env_ids: List[str]) -> Dict[str, float]:
        """
        Compute rewards for multiple environments in batch.
        
        Args:
            env_ids: List of environment IDs
            
        Returns:
            Dictionary mapping environment IDs to reward values
        """
        response = self._make_request("batch/reward", "POST", {"env_ids": env_ids})
        return response.get("rewards", {})
    
    def get_system_prompts_batch(self, env_ids: List[str]) -> Dict[str, str]:
        """
        Get system prompts for multiple environments in batch.
        
        Args:
            env_ids: List of environment IDs
            
        Returns:
            Dictionary mapping environment IDs to system prompt strings
        """
        response = self._make_request("batch/system_prompt", "POST", {"env_ids": env_ids})
        return response.get("system_prompts", {})
    
    def close_batch(self, env_ids: Optional[List[str]] = None) -> None:
        """
        Close multiple environments and clean up resources.
        
        Args:
            env_ids: Optional list of environment IDs to close. If None, close all environments.
        """
        # If no env_ids provided, close all known environments
        if env_ids is None:
            env_ids = list(self.env_configs.keys())
            
        self._make_request("batch/close", "POST", {"env_ids": env_ids})
        
        # Remove closed environments from tracking
        for env_id in env_ids:
            self.env_configs.pop(env_id, None)
    
    # Convenience methods for single-environment operations
    
    def reset(self, env_id: str, seed: Any = None) -> Tuple[Dict, Dict]:
        """
        Reset a single environment.
        
        Args:
            env_id: Environment ID
            seed: Optional seed for resetting
            
        Returns:
            Tuple of (observation, info)
        """
        results = self.reset_batch({env_id: seed})
        return results.get(env_id, ({}, {"error": "Reset failed"}))
    
    def step(self, env_id: str, action: str) -> Tuple[Dict, float, bool, Dict]:
        """
        Take a step in a single environment.
        
        Args:
            env_id: Environment ID
            action: Action to take
            
        Returns:
            Tuple of (observation, reward, done, info)
        """
        results = self.step_batch({env_id: action})
        return results.get(env_id, ({}, 0.0, True, {"error": "Step failed"}))
    
    def compute_reward(self, env_id: str) -> float:
        """
        Compute reward for a single environment.
        
        Args:
            env_id: Environment ID
            
        Returns:
            Reward value
        """
        results = self.compute_reward_batch([env_id])
        return results.get(env_id, 0.0)
    
    def get_system_prompt(self, env_id: str) -> str:
        """
        Get system prompt for a single environment.
        
        Args:
            env_id: Environment ID
            
        Returns:
            System prompt string
        """
        results = self.get_system_prompts_batch([env_id])
        return results.get(env_id, "")
    
    def close(self, env_id: str) -> None:
        """
        Close a single environment.
        
        Args:
            env_id: Environment ID
        """
        self.close_batch([env_id])


if __name__ == "__main__":
    # Example usage of the client
    client = BatchEnvClient(base_url="http://localhost:5000", timeout=10)
    
    # Wait for server to be available
    if client.wait_for_server():
        try:
            # Create environments
            configs = [
                {
                    "env_name": "frozenlake",
                    "env_config": {"is_slippery": False, "size": 4, "render_mode": "text"}
                },
                {
                    "env_name": "frozenlake",
                    "env_config": {"is_slippery": True, "size": 8, "render_mode": "vision"}
                }
            ]
            
            print("Creating environments...")
            env_ids = client.create_environments_batchs(configs)
            print(f"Created {len(env_ids)} environments: {env_ids}")
            
            # Reset environments
            print("Resetting environments...")
            ids2seeds = {env_id: i*42 for i, env_id in enumerate(env_ids)}
            results = client.reset_batch(ids2seeds)
            
            # Get system prompts
            print("Getting system prompts...")
            prompts = client.get_system_prompts_batch(env_ids)
            
            # Step environments
            print("Stepping environments...")
            ids2actions = {
                env_ids[0]: "<think>Let me try going right first.</think><answer>Right</answer>",
                env_ids[1]: "<think>I'll start by going down.</think><answer>Down</answer>"
            }
            results = client.step_batch(ids2actions)
            
            # Close environments
            print("Closing environments...")
            client.close_batch(env_ids)
            
            print("Done!")
            
        except Exception as e:
            print(f"Error: {str(e)}")
    else:
        print("Server not available")