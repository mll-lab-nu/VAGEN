from flask import Flask, request, jsonify
import logging
import threading
import time
import importlib
from typing import Dict, List, Tuple, Optional, Any, Type
from vagen.env import REGISTERED_ENV


import json
import numpy as np
from PIL import Image
import base64
from io import BytesIO
from flask import Flask, jsonify

class EnvJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.int_, np.intc, np.intp, np.int8, np.int16, np.int32, np.int64)):
            return int(obj)
        elif isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, Image.Image):
            buffer = BytesIO()
            obj.save(buffer, format="PNG")
            return {
                "image_data": base64.b64encode(buffer.getvalue()).decode('utf-8'),
                "_image_type": "PIL"
            }
        return super().default(obj)

class BatchEnvServer:
    """
    A unified server for handling batch environment operations through HTTP requests.
    This server can work with any environment type registered in REGISTERED_ENV.
    """
    
    def __init__(self, host: str = "127.0.0.1", port: int = 5000, debug: bool = False, 
                 logger: Optional[logging.Logger] = None):
        """
        Initialize the BatchEnvServer.
        
        Args:
            host: Host address for the server
            port: Port to listen on
            debug: Whether to run Flask in debug mode
            logger: Optional logger for server logs
        """
        self.host = host
        self.port = port
        self.debug = debug
        self.logger = logger or logging.getLogger(self.__class__.__name__)
        
        # Dictionary to store environment instances by ID
        self.environments = {}
        
        # Create Flask app
        self.app = Flask(__name__)
        self.app.json_encoder = EnvJSONEncoder
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
                "active_environments": len(self.environments)
            }), 200
            
        @self.app.route('/environments', methods=['POST'])
        def create_environments():
            """Create environments endpoint"""
            data = request.json
            if not data or 'configs' not in data:
                return jsonify({"error": "Missing required parameter: configs"}), 400
                
            configs = data['configs']
            try:
                env_ids = self._create_environments(configs)
                return jsonify({"env_ids": env_ids}), 200
            except Exception as e:
                self.logger.error(f"Error creating environments: {str(e)}")
                return jsonify({"error": str(e)}), 500
                
        @self.app.route('/reset/<env_id>', methods=['POST'])
        def reset_environment(env_id):
            """Reset environment endpoint"""
            if env_id not in self.environments:
                return jsonify({"error": f"Environment {env_id} not found"}), 404
                
            data = request.json or {}
            seed = data.get('seed')
            
            try:
                obs, info = self._reset_environment(env_id, seed)
                return jsonify({"observation": obs, "info": info}), 200
            except Exception as e:
                self.logger.error(f"Error resetting environment {env_id}: {str(e)}")
                return jsonify({"error": str(e)}), 500
                
        @self.app.route('/step/<env_id>', methods=['POST'])
        def step_environment(env_id):
            """Step environment endpoint"""
            if env_id not in self.environments:
                return jsonify({"error": f"Environment {env_id} not found"}), 404
                
            data = request.json
            if not data or 'action' not in data:
                return jsonify({"error": "Missing required parameter: action"}), 400
                
            action = data['action']
            
            try:
                obs, reward, done, info = self._step_environment(env_id, action)
                return jsonify({
                    "observation": obs,
                    "reward": reward,
                    "done": done,
                    "info": info
                }), 200
            except Exception as e:
                self.logger.error(f"Error stepping environment {env_id}: {str(e)}")
                return jsonify({"error": str(e)}), 500
                
        @self.app.route('/reward/<env_id>', methods=['GET'])
        def compute_reward(env_id):
            """Compute reward endpoint"""
            if env_id not in self.environments:
                return jsonify({"error": f"Environment {env_id} not found"}), 404
                
            try:
                reward = self._compute_reward(env_id)
                return jsonify({"reward": reward}), 200
            except Exception as e:
                self.logger.error(f"Error computing reward for environment {env_id}: {str(e)}")
                return jsonify({"error": str(e)}), 500
                
        @self.app.route('/system_prompt/<env_id>', methods=['GET'])
        def get_system_prompt(env_id):
            """Get system prompt endpoint"""
            if env_id not in self.environments:
                return jsonify({"error": f"Environment {env_id} not found"}), 404
                
            try:
                system_prompt = self._get_system_prompt(env_id)
                return jsonify({"system_prompt": system_prompt}), 200
            except Exception as e:
                self.logger.error(f"Error getting system prompt for environment {env_id}: {str(e)}")
                return jsonify({"error": str(e)}), 500
                
        @self.app.route('/close/<env_id>', methods=['DELETE'])
        def close_environment(env_id):
            """Close environment endpoint"""
            if env_id not in self.environments:
                return jsonify({"error": f"Environment {env_id} not found"}), 404
                
            try:
                self._close_environment(env_id)
                return jsonify({"status": "success"}), 200
            except Exception as e:
                self.logger.error(f"Error closing environment {env_id}: {str(e)}")
                return jsonify({"error": str(e)}), 500
        
        # Add custom endpoints for environment-specific operations
        # For example, for FrozenLake:
        @self.app.route('/map/<env_id>', methods=['GET'])
        def get_map(env_id):
            """Get FrozenLake map endpoint"""
            if env_id not in self.environments:
                return jsonify({"error": f"Environment {env_id} not found"}), 404
                
            env_data = self.environments[env_id]
            if env_data["env_name"] != "frozenlake":
                return jsonify({"error": "Environment is not a FrozenLake environment"}), 400
                
            try:
                env = env_data["env"]
                if hasattr(env, "gym_env") and hasattr(env.gym_env, "desc"):
                    # Convert bytes to strings for JSON serialization
                    map_data = [[cell.decode('utf-8') for cell in row] for row in env.gym_env.desc]
                    return jsonify({"map": map_data}), 200
                else:
                    return jsonify({"error": "Map not available"}), 404
            except Exception as e:
                self.logger.error(f"Error getting map for environment {env_id}: {str(e)}")
                return jsonify({"error": str(e)}), 500
                
        @self.app.route('/player_position/<env_id>', methods=['GET'])
        def get_player_position(env_id):
            """Get FrozenLake player position endpoint"""
            if env_id not in self.environments:
                return jsonify({"error": f"Environment {env_id} not found"}), 404
                
            env_data = self.environments[env_id]
            if env_data["env_name"] != "frozenlake":
                return jsonify({"error": "Environment is not a FrozenLake environment"}), 400
                
            try:
                env = env_data["env"]
                if hasattr(env, "_get_player_position"):
                    position = env._get_player_position()
                    # 将位置转换为原生Python类型
                    position = tuple(int(x) for x in position)
                    return jsonify({"position": position}), 200
                else:
                    return jsonify({"error": "Player position not available"}), 404
            except Exception as e:
                self.logger.error(f"Error getting player position for environment {env_id}: {str(e)}")
                return jsonify({"error": str(e)}), 500
                
    def _generate_env_id(self) -> str:
        """
        Generate a unique environment ID.
        
        Returns:
            A unique environment ID
        """
        import uuid
        return str(uuid.uuid4())
    
    def _create_environments(self, configs: List[Dict[str, Any]]) -> List[str]:
        """
        Create environments from configurations.
        
        Args:
            configs: List of environment configurations
            
        Returns:
            List of environment IDs
        """
        env_ids = []
        
        for config_dict in configs:
            try:
                # Get environment type
                env_name = config_dict.get('env_name')
                if not env_name or env_name not in REGISTERED_ENV:
                    raise ValueError(f"Invalid or missing environment name: {env_name}")
                
                # Get configuration
                env_config_dict = config_dict.get('env_config', {})
                
                # Create environment
                env_cls = REGISTERED_ENV[env_name]['env_cls']
                config_cls = REGISTERED_ENV[env_name]['config_cls']
                env_config = config_cls(**env_config_dict)
                env = env_cls(env_config)
                
                # Generate ID and store environment
                env_id = self._generate_env_id()
                self.environments[env_id] = {
                    "env": env,
                    "env_name": env_name,
                    "config": env_config,
                    "seed": env_config_dict.get('seed')
                }
                
                env_ids.append(env_id)
                
            except Exception as e:
                self.logger.error(f"Error creating environment: {str(e)}")
                # Clean up any environments created so far
                for env_id in env_ids:
                    self._close_environment(env_id)
                raise
                
        return env_ids
    
    def _reset_environment(self, env_id: str, seed: Optional[int] = None) -> Tuple[Dict, Dict]:
        """
        Reset an environment.
        
        Args:
            env_id: Environment ID
            seed: Optional seed for resetting
            
        Returns:
            Tuple of (observation, info)
        """
        env_data = self.environments[env_id]
        env = env_data['env']
        
        # Use the provided seed or the original seed from config
        actual_seed = seed if seed is not None else env_data.get('seed')
        
        return env.reset(seed=actual_seed)
    
    def _step_environment(self, env_id: str, action: str) -> Tuple[Dict, float, bool, Dict]:
        """
        Take a step in an environment.
        
        Args:
            env_id: Environment ID
            action: Action to take
            
        Returns:
            Tuple of (observation, reward, done, info)
        """
        env_data = self.environments[env_id]
        env = env_data['env']
        
        return env.step(action)
    
    def _compute_reward(self, env_id: str) -> float:
        """
        Compute reward for an environment.
        
        Args:
            env_id: Environment ID
            
        Returns:
            Reward value
        """
        env_data = self.environments[env_id]
        env = env_data['env']
        
        return env.compute_reward()
    
    def _get_system_prompt(self, env_id: str) -> str:
        """
        Get system prompt for an environment.
        
        Args:
            env_id: Environment ID
            
        Returns:
            System prompt string
        """
        env_data = self.environments[env_id]
        env = env_data['env']
        
        return env.system_prompt()
    
    def _close_environment(self, env_id: str) -> None:
        """
        Close an environment.
        
        Args:
            env_id: Environment ID
        """
        if env_id in self.environments:
            env_data = self.environments[env_id]
            env = env_data['env']
            
            # Close the environment
            env.close()
            
            # Remove from environments dictionary
            del self.environments[env_id]
    
    def start(self, background: bool = True) -> None:
        """
        Start the server.
        
        Args:
            background: Whether to run the server in a background thread
        """
        if self.is_running:
            self.logger.warning("Server is already running")
            return
            
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
                try:
                    import requests
                    response = requests.get(f"http://{self.host}:{self.port}/health", timeout=1)
                    if response.status_code == 200:
                        self.logger.info(f"Server started on http://{self.host}:{self.port}")
                        break
                except Exception:
                    pass
            else:
                self.logger.warning("Server may not have started properly")
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
        env_ids = list(self.environments.keys())
        for env_id in env_ids:
            try:
                self._close_environment(env_id)
            except Exception as e:
                self.logger.error(f"Error closing environment {env_id}: {str(e)}")
                
        self.environments.clear()
        
        # Shut down the Flask server
        self.is_running = False
        if self.server_thread and self.server_thread.is_alive():
            # This doesn't actually stop Flask in a clean way
            # In a production environment, you would use a proper WSGI server
            import requests
            try:
                requests.post(f"http://{self.host}:{self.port}/shutdown")
            except Exception:
                pass
                
        self.logger.info("Server stopped")


def main():
    """
    Main function to start the batch environment server.
    """
    import argparse
    
    parser = argparse.ArgumentParser(description='Batch Environment Server')
    parser.add_argument('--host', type=str, default='127.0.0.1', help='Host address')
    parser.add_argument('--port', type=int, default=5000, help='Port number')
    parser.add_argument('--debug', action='store_true', help='Run in debug mode')
    
    args = parser.parse_args()
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger('BatchEnvServer')
    
    # Create and start server
    server = BatchEnvServer(
        host=args.host,
        port=args.port,
        debug=args.debug,
        logger=logger
    )
    
    logger.info(f"Starting Batch Environment Server on http://{args.host}:{args.port}")
    server.start(background=False)


if __name__ == "__main__":
    main()