from flask import Flask, request, jsonify
import threading
import time
import importlib
from typing import Dict, List, Tuple, Optional, Any, Type
from vagen.env import REGISTERED_ENV
from vagen.env.base.base_service import BaseService
from vagen.env.base.base_service_config import BaseServiceConfig
import hydra
from omegaconf import DictConfig
from vagen.server.llm_as_judge import wandb_run_context

class BatchEnvServer:
    """
    A unified server for handling batch environment operations through HTTP requests.
    Uses environment services to handle operations and properly handle serialization.
    Exposes only the standard BaseService interface.
    """
    
    def __init__(self, config):
        """
        Initialize the BatchEnvServer.
        
        Args:
            host: Host address for the server
            port: Port to listen on
            debug: Whether to run Flask in debug mode
        """
        self.host = config.server.host
        self.port = config.server.port
        self.debug = config.server.debug
        self.config=config
        self.wandb_context = None
        
        # Dictionary to store services by environment type
        self.services = {}
        
        # Dictionary to track which service manages which environment ID
        self.env_to_service = {}
        
        # Create Flask app
        self.app = Flask(__name__)
        self._setup_routes()
        
        # Server state
        self.is_running = False
        self.server_thread = None
    
    def _setup_routes(self):
        """Set up HTTP routes for the Flask app"""
        
        @self.app.route('/health', methods=['GET'])
        def health_check():
            """Health check endpoint"""
            return jsonify({
                "status": "ok",
                "message": "Environment server is running",
                "registered_envs": list(REGISTERED_ENV.keys()),
                "active_services": list(self.services.keys()),
                "active_environments": len(self.env_to_service)
            }), 200
            
        @self.app.route('/environments', methods=['POST'])
        def create_environments_batch():
            """Create environments endpoint - implements BaseService interface"""
            data = request.json
            if not data or 'ids2configs' not in data:
                return jsonify({"error": "Missing required parameter: ids2configs"}), 400
                    
            ids2configs = data['ids2configs']
            self._create_environments_batch(ids2configs)
            return jsonify({"success": True}), 200
        
        @self.app.route('/batch/reset', methods=['POST'])
        def reset_batch():
            """Reset multiple environments endpoint"""
            data = request.json
            if not data or 'ids2seeds' not in data:
                return jsonify({"error": "Missing required parameter: ids2seeds"}), 400
                
            ids2seeds = data['ids2seeds']
            results = self._reset_batch(ids2seeds)
            return jsonify({"results": results}), 200
                
        @self.app.route('/batch/step', methods=['POST'])
        def step_batch():
            """Step multiple environments endpoint"""
            data = request.json
            if not data or 'ids2actions' not in data:
                return jsonify({"error": "Missing required parameter: ids2actions"}), 400
                
            ids2actions = data['ids2actions']
            results = self._step_batch(ids2actions)
            return jsonify({"results": results}), 200
                
        @self.app.route('/batch/reward', methods=['POST'])
        def compute_reward_batch():
            """Compute reward for multiple environments endpoint"""
            data = request.json
            if not data or 'env_ids' not in data:
                return jsonify({"error": "Missing required parameter: env_ids"}), 400
                
            env_ids = data['env_ids']
            rewards = self._compute_reward_batch(env_ids)
            return jsonify({"rewards": rewards}), 200
                
        @self.app.route('/batch/system_prompt', methods=['POST'])
        def get_system_prompts_batch():
            """Get system prompts for multiple environments endpoint"""
            data = request.json
            if not data or 'env_ids' not in data:
                return jsonify({"error": "Missing required parameter: env_ids"}), 400
                
            env_ids = data['env_ids']
            prompts = self._get_system_prompts_batch(env_ids)
            return jsonify({"system_prompts": prompts}), 200
                
        @self.app.route('/batch/close', methods=['POST'])
        def close_batch():
            """Close multiple environments endpoint"""
            data = request.json
            if not data or 'env_ids' not in data:
                return jsonify({"error": "Missing required parameter: env_ids"}), 400
                
            env_ids = data['env_ids']
            self._close_batch(env_ids)
            return jsonify({"status": "success"}), 200
        
        @self.app.route('/reset/<env_id>', methods=['POST'])
        def reset_environment(env_id):
            """Reset single environment endpoint"""
            data = request.json or {}
            seed = data.get('seed')
            results = self._reset_batch({env_id: seed})
            if env_id not in results:
                return jsonify({"error": f"Environment {env_id} not found"}), 404
                    
            obs, info = results[env_id]
            return jsonify({"observation": obs, "info": info}), 200
                
        @self.app.route('/step/<env_id>', methods=['POST'])
        def step_environment(env_id):
            """Step single environment endpoint"""
            data = request.json
            if not data or 'action' not in data:
                return jsonify({"error": "Missing required parameter: action"}), 400
                
            action = data['action']
            results = self._step_batch({env_id: action})
            if env_id not in results:
                return jsonify({"error": f"Environment {env_id} not found"}), 404
                    
            obs, reward, done, info = results[env_id]
            return jsonify({
                "observation": obs,
                "reward": reward,
                "done": done,
                "info": info
            }), 200
                
        @self.app.route('/reward/<env_id>', methods=['GET'])
        def compute_reward(env_id):
            """Compute reward for single environment endpoint"""
            rewards = self._compute_reward_batch([env_id])
            if env_id not in rewards:
                return jsonify({"error": f"Environment {env_id} not found"}), 404
            return jsonify({"reward": rewards[env_id]}), 200
                
        @self.app.route('/system_prompt/<env_id>', methods=['GET'])
        def get_system_prompt(env_id):
            """Get system prompt for single environment endpoint"""
            prompts = self._get_system_prompts_batch([env_id])
            if env_id not in prompts:
                return jsonify({"error": f"Environment {env_id} not found"}), 404
            return jsonify({"system_prompt": prompts[env_id]}), 200
                
        @self.app.route('/close/<env_id>', methods=['DELETE'])
        def close_environment(env_id):
            """Close single environment endpoint"""
            self._close_batch([env_id])
            return jsonify({"status": "success"}), 200
    
    def _get_service_for_env_name(self, env_name: str) -> BaseService:
        """
        Get or create a service for the specified environment type.
        
        Args:
            env_name: Name of the environment type
            
        Returns:
            Service instance for the environment type
        """
        if env_name not in self.services:
            # Check if environment is registered
            if env_name not in REGISTERED_ENV:
                raise ValueError(f"Unknown environment type: {env_name}")
                
            # Get the service class directly from REGISTERED_ENV
            if "service_cls" not in REGISTERED_ENV[env_name]:
                raise ValueError(f"No service class registered for environment type: {env_name}")
                
            service_class = REGISTERED_ENV[env_name]["service_cls"]
            service_config = REGISTERED_ENV[env_name].get("service_config_cls", BaseServiceConfig)(**self.config.get(env_name, {}))
            self.services[env_name] = service_class(service_config)
                
        return self.services[env_name]
    
    def _get_service_for_env(self, env_id: str) -> Tuple[BaseService, str]:
        """
        Get the service that manages the specified environment.
        
        Args:
            env_id: Environment ID
            
        Returns:
            Tuple of (service, environment_type)
        """
        if env_id not in self.env_to_service:
            raise ValueError(f"Environment {env_id} not found")
            
        env_name = self.env_to_service[env_id]
        service = self.services[env_name]
        return service, env_name
    
    def _create_environments_batch(self, ids2configs: Dict[Any, Any]) -> None:
        """
        Create multiple environments in batch.
        Implements BaseService.create_environments_batch.
        
        Args:
            ids2configs: Dictionary mapping environment IDs to their configurations
        """
        # Process each config to determine which service should handle it
        for env_id, config in ids2configs.items():
            env_name = config.get("env_name")
            if not env_name:
                raise ValueError(f"Config for environment {env_id} is missing 'env_name'")
                
            # Get or create the appropriate service
            if env_name not in self.services:
                self.services[env_name] = self._get_service_for_env_name(env_name)
                
            # Track which service manages this environment
            self.env_to_service[env_id] = env_name
        
        # Group configs by service type
        service_to_configs = {}
        for env_id, config in ids2configs.items():
            env_name = self.env_to_service[env_id]
            if env_name not in service_to_configs:
                service_to_configs[env_name] = {}
            service_to_configs[env_name][env_id] = config
        
        # Call create_environments_batch method on each service
        for env_name, configs in service_to_configs.items():
            service = self.services[env_name]
            service.create_environments_batch(configs)
    
    
    def _reset_batch(self, ids2seeds: Dict[str, Any]) -> Dict[str, Tuple[Any, Any]]:
        """
        Reset multiple environments.
        
        Args:
            ids2seeds: Dictionary mapping environment IDs to seeds
            
        Returns:
            Dictionary mapping environment IDs to (observation, info) tuples
        """
        # Group environment IDs by service
        service_groups = {}
        for env_id, seed in ids2seeds.items():
            service, env_name = self._get_service_for_env(env_id)
            if env_name not in service_groups:
                service_groups[env_name] = (service, {})
            service_groups[env_name][1][env_id] = seed
        
        # Reset environments through respective services
        results = {}
        for env_name, (service, group_ids2seeds) in service_groups.items():
            service_results = service.reset_batch(group_ids2seeds)
            results.update(service_results)
        
        return results
    
    def _step_batch(self, ids2actions: Dict[str, Any]) -> Dict[str, Tuple[Dict, float, bool, Dict]]:
        """
        Step multiple environments.
        
        Args:
            ids2actions: Dictionary mapping environment IDs to actions
            
        Returns:
            Dictionary mapping environment IDs to (observation, reward, done, info) tuples
        """
        # Group environment IDs by service
        service_groups = {}
        for env_id, action in ids2actions.items():
            service, env_name = self._get_service_for_env(env_id)
            if env_name not in service_groups:
                service_groups[env_name] = (service, {})
            service_groups[env_name][1][env_id] = action

        # Step environments through respective services
        results = {}
        for env_name, (service, group_ids2actions) in service_groups.items():
            service_results = service.step_batch(group_ids2actions)
            results.update(service_results)
        
        return results
    
    def _compute_reward_batch(self, env_ids: List[str]) -> Dict[str, float]:
        """
        Compute rewards for multiple environments.
        
        Args:
            env_ids: List of environment IDs
            
        Returns:
            Dictionary mapping environment IDs to reward values
        """
        # Group environment IDs by service
        service_groups = {}
        for env_id in env_ids:
            service, env_name = self._get_service_for_env(env_id)
            if env_name not in service_groups:
                service_groups[env_name] = (service, [])
            service_groups[env_name][1].append(env_id)
        
        # Compute rewards through respective services
        results = {}
        for env_name, (service, group_env_ids) in service_groups.items():
            service_results = service.compute_reward_batch(group_env_ids)
            results.update(service_results)
        
        return results
    
    def _get_system_prompts_batch(self, env_ids: List[str]) -> Dict[str, str]:
        """
        Get system prompts for multiple environments.
        
        Args:
            env_ids: List of environment IDs
            
        Returns:
            Dictionary mapping environment IDs to system prompt strings
        """
        # Group environment IDs by service
        service_groups = {}
        for env_id in env_ids:
            service, env_name = self._get_service_for_env(env_id)
            if env_name not in service_groups:
                service_groups[env_name] = (service, [])
            service_groups[env_name][1].append(env_id)
        
        # Get system prompts through respective services
        results = {}
        for env_name, (service, group_env_ids) in service_groups.items():
            service_results = service.get_system_prompts_batch(group_env_ids)
            results.update(service_results)
        
        return results
    
    def _close_batch(self, env_ids: List[str]) -> None:
        """
        Close multiple environments.
        
        Args:
            env_ids: List of environment IDs
        """
        # Group environment IDs by service
        service_groups = {}
        for env_id in env_ids:
            service, env_name = self._get_service_for_env(env_id)
            if env_name not in service_groups:
                service_groups[env_name] = (service, [])
            service_groups[env_name][1].append(env_id)
            # Remove from tracking
            del self.env_to_service[env_id]

        # Close environments through respective services
        for env_name, (service, group_env_ids) in service_groups.items():
            service.close_batch(group_env_ids)
    
    def _generate_env_id(self) -> str:
        """
        Generate a unique environment ID.
        
        Returns:
            A unique environment ID
        """
        import uuid
        return str(uuid.uuid4())
    
    def start(self, background: bool = True) -> None:
        """
        Start the server.
        
        Args:
            background: Whether to run the server in a background thread
        """
        if self.is_running:
            print("Server is already running")
            return
        
        if self.config.get("use_state_reward", False):
            self.wandb_context = wandb_run_context()
            self.wandb_context.__enter__()
            print("Initialized wandb for LLM Judge")
            
        if background:
            self.server_thread = threading.Thread(target=self._run_server)
            self.server_thread.daemon = True
            self.server_thread.start()
            self.is_running = True
            
            # Wait for server to start
            max_retries = 5
            retry_delay = 0.5
            for _ in range(max_retries):
                time.sleep(retry_delay)
                import requests
                response = requests.get(f"http://{self.host}:{self.port}/health", timeout=1)
                if response.status_code == 200:
                    print(f"Server started on http://{self.host}:{self.port}")
                    break
            else:
                print("Server may not have started properly")
        else:
            self.is_running = True
            self._run_server()
    
    def _run_server(self) -> None:
        """Run the Flask server"""
        self.app.run(host=self.host, port=self.port, debug=self.debug, use_reloader=False)
    
    def stop(self) -> None:
        """Stop the server and clean up resources"""
        if not self.is_running:
            return
            
        # Close all environments
        env_ids = list(self.env_to_service.keys())
        self._close_batch(env_ids)
        
        # Shut down the Flask server
        self.is_running = False
        if self.server_thread and self.server_thread.is_alive():
            import requests
            requests.post(f"http://{self.host}:{self.port}/shutdown")
        
        
        if self.wandb_context:
            self.wandb_context.__exit__(None, None, None)
            self.wandb_context = None
            print("Closed wandb for LLM Judge")
            
        print("Server stopped")


@hydra.main(version_base=None, config_path="config", config_name="server")
def main(cfg: DictConfig):
    """
    Main function to start the batch environment server.
    Uses Hydra for configuration management.
    
    Args:
        cfg: Configuration object from Hydra
    """
    # Create and start server with configuration
    print(cfg)
    server = BatchEnvServer(cfg)
    print(f"Starting Batch Environment Server on http://{cfg.server.host}:{cfg.server.port}")
    server.start(background=False)


if __name__ == "__main__":
    main()
