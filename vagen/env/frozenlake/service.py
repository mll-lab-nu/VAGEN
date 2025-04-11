from typing import Dict, List, Tuple, Optional, Any
import logging
from base_service import BaseService
from vagen.env.frozenlake.env import FrozenLakeEnv
from vagen.env.frozenlake.config import FrozenLakeConfig

class FrozenLakeService(BaseService):
    """
    Service class for FrozenLake environments.
    This class manages FrozenLake environments and provides additional methods
    specific to FrozenLake.
    """
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        """
        Initialize the FrozenLakeService.
        
        Args:
            logger: Optional logger for service logs
        """
        super().__init__(logger)
    
    def create_environment(self, config: Dict[str, Any]) -> str:
        """
        Create a FrozenLake environment.
        
        Args:
            config: Environment configuration
                - env_name: Must be "frozenlake"
                - env_config: FrozenLake specific configuration
                
        Returns:
            Environment ID
        """
        # Check environment type
        env_name = config.get('env_name', 'frozenlake')
        if env_name != 'frozenlake':
            raise ValueError(f"Expected environment type 'frozenlake', got '{env_name}'")
        
        # Get FrozenLake specific configuration
        env_config_dict = config.get('env_config', {})
        
        # Create environment config
        env_config = FrozenLakeConfig(**env_config_dict)
        
        # Create environment
        env = FrozenLakeEnv(env_config)
        
        # Generate and store environment ID
        env_id = self._generate_env_id()
        self.environments[env_id] = env
        self.env_configs[env_id] = env_config
        
        return env_id
    
    def reset(self, env_id: str, seed: Optional[int] = None) -> Tuple[Dict, Dict]:
        """
        Reset a FrozenLake environment.
        
        Args:
            env_id: Environment ID
            seed: Optional seed for resetting
            
        Returns:
            Tuple of (observation, info)
        """
        if env_id not in self.environments:
            raise ValueError(f"Environment {env_id} not found")
        
        env = self.environments[env_id]
        return env.reset(seed=seed)
    
    def step(self, env_id: str, action: str) -> Tuple[Dict, float, bool, Dict]:
        """
        Take a step in a FrozenLake environment.
        
        Args:
            env_id: Environment ID
            action: Action to take
            
        Returns:
            Tuple of (observation, reward, done, info)
        """
        if env_id not in self.environments:
            raise ValueError(f"Environment {env_id} not found")
        
        env = self.environments[env_id]
        return env.step(action)
    
    def compute_reward(self, env_id: str) -> float:
        """
        Compute the total reward for a FrozenLake environment.
        
        Args:
            env_id: Environment ID
            
        Returns:
            Reward value
        """
        if env_id not in self.environments:
            raise ValueError(f"Environment {env_id} not found")
        
        env = self.environments[env_id]
        return env.compute_reward()
    
    def get_system_prompt(self, env_id: str) -> str:
        """
        Get the system prompt for a FrozenLake environment.
        
        Args:
            env_id: Environment ID
            
        Returns:
            System prompt string
        """
        if env_id not in self.environments:
            raise ValueError(f"Environment {env_id} not found")
        
        env = self.environments[env_id]
        return env.system_prompt()
    
    def close(self, env_id: str) -> None:
        """
        Close a FrozenLake environment and clean up resources.
        
        Args:
            env_id: Environment ID
        """
        if env_id not in self.environments:
            raise ValueError(f"Environment {env_id} not found")
        
        env = self.environments[env_id]
        env.close()
        
        # Remove from dictionaries
        del self.environments[env_id]
        del self.env_configs[env_id]
    
    # Additional FrozenLake specific methods
    
    def get_map(self, env_id: str) -> List[List[str]]:
        """
        Get the map of a FrozenLake environment.
        
        Args:
            env_id: Environment ID
            
        Returns:
            2D list representing the map
        """
        if env_id not in self.environments:
            raise ValueError(f"Environment {env_id} not found")
        
        env = self.environments[env_id]
        if not hasattr(env, "gym_env") or not hasattr(env.gym_env, "desc"):
            raise ValueError("Map not available for this environment")
        
        # Convert bytes to strings
        return [[cell.decode('utf-8') for cell in row] for row in env.gym_env.desc]
    
    def get_player_position(self, env_id: str) -> Tuple[int, int]:
        """
        Get the player position in a FrozenLake environment.
        
        Args:
            env_id: Environment ID
            
        Returns:
            Tuple of (row, col)
        """
        if env_id not in self.environments:
            raise ValueError(f"Environment {env_id} not found")
        
        env = self.environments[env_id]
        if not hasattr(env, "_get_player_position"):
            raise ValueError("Player position not available for this environment")
        
        return env._get_player_position()
    
    def check_success(self, env_id: str) -> bool:
        """
        Check if the agent has reached the goal in a FrozenLake environment.
        
        Args:
            env_id: Environment ID
            
        Returns:
            True if successful, False otherwise
        """
        if env_id not in self.environments:
            raise ValueError(f"Environment {env_id} not found")
        
        env = self.environments[env_id]
        if not hasattr(env, "_success"):
            raise ValueError("Success check not available for this environment")
        
        return env._success()
    
    def is_done(self, env_id: str) -> bool:
        """
        Check if the episode is done in a FrozenLake environment.
        
        Args:
            env_id: Environment ID
            
        Returns:
            True if done, False otherwise
        """
        if env_id not in self.environments:
            raise ValueError(f"Environment {env_id} not found")
        
        env = self.environments[env_id]
        if not hasattr(env, "_finished"):
            raise ValueError("Done check not available for this environment")
        
        return env._finished()
    
    def get_action_lookup(self, env_id: str) -> Dict[str, int]:
        """
        Get the action lookup dictionary for a FrozenLake environment.
        
        Args:
            env_id: Environment ID
            
        Returns:
            Dictionary mapping action names to integers
        """
        if env_id not in self.environments:
            raise ValueError(f"Environment {env_id} not found")
        
        env = self.environments[env_id]
        if not hasattr(env, "ACTION_LOOKUP"):
            raise ValueError("Action lookup not available for this environment")
        
        return env.ACTION_LOOKUP
    
    def get_text_representation(self, env_id: str) -> List[List[int]]:
        """
        Get the text representation of a FrozenLake environment.
        
        Args:
            env_id: Environment ID
            
        Returns:
            2D list representing the text representation
        """
        if env_id not in self.environments:
            raise ValueError(f"Environment {env_id} not found")
        
        env = self.environments[env_id]
        if not hasattr(env, "_get_text_representation"):
            raise ValueError("Text representation not available for this environment")
        
        return env._get_text_representation().tolist()
    
    def get_batch_maps(self, env_ids: List[str]) -> List[Optional[List[List[str]]]]:
        """
        Get maps for multiple FrozenLake environments.
        
        Args:
            env_ids: List of environment IDs
            
        Returns:
            List of maps, or None for environments where maps are not available
        """
        maps = []
        for env_id in env_ids:
            try:
                map_data = self.get_map(env_id)
                maps.append(map_data)
            except Exception as e:
                self.logger.error(f"Error getting map for environment {env_id}: {str(e)}")
                maps.append(None)
        return maps
    
    def get_batch_player_positions(self, env_ids: List[str]) -> List[Optional[Tuple[int, int]]]:
        """
        Get player positions for multiple FrozenLake environments.
        
        Args:
            env_ids: List of environment IDs
            
        Returns:
            List of player positions, or None for environments where positions are not available
        """
        positions = []
        for env_id in env_ids:
            try:
                position = self.get_player_position(env_id)
                positions.append(position)
            except Exception as e:
                self.logger.error(f"Error getting player position for environment {env_id}: {str(e)}")
                positions.append(None)
        return positions