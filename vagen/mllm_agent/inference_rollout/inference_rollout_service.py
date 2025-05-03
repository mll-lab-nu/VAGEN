import os
import time
from typing import List, Dict, Tuple, Optional, Any
from collections import defaultdict
from tqdm import tqdm
import PIL

from vagen.mllm_agent.base_rollout import BaseRollout
from vagen.server.client import BatchEnvClient
from vagen.env import REGISTERED_ENV

class InferenceRolloutService(BaseRollout):
    """
    Implementation of BaseRollout for inference.
    Handles batch-wise environment interaction and model inference
    without training-specific components.
    """
    
    def __init__(self,
                 config: Dict,
                 model_interface,
                 base_url: str = "http://localhost:5000",
                 timeout: int = 600,
                 max_workers: int = 10,
                 split: str = "test",
                 debug: bool = False):
        """
        Initialize the inference rollout service.
        
        Args:
            config: Configuration dictionary
            model_interface: Model interface for generating responses
            base_url: URL of the environment service
            timeout: Timeout for HTTP requests in seconds
            max_workers: Maximum number of worker threads
            split: Data split name (test/val)
            debug: Enable debug logging
        """
        self.config = config
        self.model_interface = model_interface
        self.base_url = base_url
        self.timeout = timeout
        self.max_workers = max_workers
        self.split = split
        self.debug = debug
        
        # Initialize environment client
        self.env_client = BatchEnvClient(
            base_url=base_url,
            timeout=timeout,
            max_workers=max_workers
        )
        
        # Environment tracking
        self.envs = {}  # Maps env_id to config instances
        self.env_states = {}  # Maps env_id to environment state
        self.recordings = {}  # Maps env_id to recorded trajectory
        self.system_prompts = {}  # Maps env_id to system prompt
        
        # Max number of steps from config
        self.max_steps = config.get("max_steps", 10)
        
        # Show progress bar
        self.show_progress = config.get("show_progress", True)
        
        # Debug messages
        if self.debug:
            print(f"Initialized InferenceRolloutService with model: {model_interface.get_model_info()['name']}")
    
    def reset(self, env_configs: List[Dict]) -> Dict[str, Tuple[Dict, Dict]]:
        """
        Reset environments based on provided configurations.
        
        Args:
            env_configs: List of environment configurations
            
        Returns:
            Dictionary mapping environment IDs to (observation, info) tuples
        """
        # Clean up existing environments if any
        if self.envs:
            self.close()
        
        # Reset tracking structures
        self.envs = {}
        self.env_states = {}
        self.recordings = {}
        self.system_prompts = {}
        
        # Prepare environment configurations
        ids2configs = {}
        ids2seeds = {}
        
        for i, cfg in enumerate(env_configs):
            env_id = f"{self.split}_{i}"
            ids2configs[env_id] = cfg
            ids2seeds[env_id] = cfg.get("seed", 42)
            
            # Store configuration for reference
            self.envs[env_id] = REGISTERED_ENV[cfg["env_name"]]["config_cls"](**cfg["env_config"])
        
        if self.debug:
            print(f"Creating {len(env_configs)} environments...")
        
        # Create and reset environments
        self.env_client.create_environments_batch(ids2configs)
        reset_results = self.env_client.reset_batch(ids2seeds)
        
        # Get system prompts
        self.system_prompts = self.env_client.get_system_prompts_batch(list(self.envs.keys()))
        
        # Initialize recordings and state tracking
        for env_id, (obs, info) in reset_results.items():
            # Initialize recording with system prompt and first observation
            self.recordings[env_id] = [
                {"role": "system", "content": self.system_prompts[env_id]},
                {"role": "user", "content": obs["obs_str"]}
            ]
            
            # Track multi-modal data if present
            if "multi_modal_data" in obs:
                # Store multimodal data with the message
                self.recordings[env_id][-1]["multi_modal_data"] = obs["multi_modal_data"]
            
            # Initialize environment state
            self.env_states[env_id] = {
                "step": 0,
                "done": False,
                "last_obs": obs,
                "last_info": info,
                "rewards": [],
                "metrics": {
                    "turn_metrics": defaultdict(list),
                    "traj_metrics": {}
                }
            }
        
        if self.debug:
            print(f"Reset {len(reset_results)} environments")
        
        return reset_results
    
    def run(self, max_steps: int = None) -> None:
        """
        Run inference on all environments until completion or max steps.
        
        Args:
            max_steps: Maximum number of steps to run for each environment,
                      overrides config value if provided
        """
        if max_steps is None:
            max_steps = self.max_steps
        
        # Track active environments
        active_envs = set(self.envs.keys())
        
        # Progress bar
        progress_iter = range(max_steps)
        if self.show_progress:
            progress_iter = tqdm(progress_iter, desc="Inference steps")
        
        # Main inference loop
        for step in progress_iter:
            if not active_envs:
                if self.debug:
                    print(f"All environments completed after {step} steps")
                break
            
            # Collect prompts and images for active environments
            env_messages = {}
            
            for env_id in active_envs:
                # Get conversation history for this environment
                env_messages[env_id] = self.recordings[env_id]
            
            # Generate responses using model interface
            start_time = time.time()
            responses = self._generate_batch_responses(env_messages)
            gen_time = time.time() - start_time
            
            # Step environments with responses
            next_active_envs = set()
            
            # Group responses for batch step
            ids2actions = {env_id: response for env_id, response in responses.items()}
            
            # Step environments using service
            step_results = self.env_client.step_batch(ids2actions)
            
            for env_id, (obs, reward, done, info) in step_results.items():
                # Update state
                self.env_states[env_id]["step"] += 1
                self.env_states[env_id]["done"] = done
                self.env_states[env_id]["last_obs"] = obs
                self.env_states[env_id]["last_info"] = info
                self.env_states[env_id]["rewards"].append(reward)
                
                # Store llm_raw_response in info
                info["llm_raw_response"] = responses[env_id]
                
                # Update metrics
                self.env_states[env_id]["metrics"]["traj_metrics"] = info["metrics"].get("traj_metrics", {})
                for k, v in info["metrics"].get("turn_metrics", {}).items():
                    self.env_states[env_id]["metrics"]["turn_metrics"][k].append(v)
                
                # Add assistant response to recording
                self.recordings[env_id].append({
                    "role": "assistant",
                    "content": responses[env_id]
                })
                
                # Add user observation to recording if not done
                if not done:
                    user_message = {"role": "user", "content": obs["obs_str"]}
                    
                    # Track multi-modal data if present
                    if "multi_modal_data" in obs:
                        user_message["multi_modal_data"] = obs["multi_modal_data"]
                    
                    self.recordings[env_id].append(user_message)
                    next_active_envs.add(env_id)
            
            # Update active environments for next iteration
            active_envs = next_active_envs
            
            if self.debug or (step % 5 == 0 and self.show_progress):
                # Print progress stats every 5 steps
                completion_rate = (len(self.envs) - len(active_envs)) / len(self.envs) * 100
                avg_steps = sum(self.env_states[env_id]["step"] for env_id in self.env_states) / len(self.env_states)
                print(f"Step {step+1}: {completion_rate:.1f}% environments completed, {len(active_envs)} active, avg steps: {avg_steps:.1f}, gen time: {gen_time:.3f}s")
    
    def _generate_batch_responses(self, env_messages: Dict[str, List[Dict]]) -> Dict[str, str]:
        """
        Generate responses for multiple environments.
        
        Args:
            env_messages: Dictionary mapping environment IDs to conversation histories
            
        Returns:
            Dictionary mapping environment IDs to generated responses
        """
        responses = {}
        env_ids = list(env_messages.keys())
        prompts = []
        
        for env_id in env_ids:
            messages = env_messages[env_id]
            # Check if any message has multimodal data
            has_images = any("multi_modal_data" in msg for msg in messages)
            
            if has_images:
                # Process multimodal input
                prompt = self._process_multimodal_messages(messages)
            else:
                # Process text-only input
                prompt = self.model_interface.format_prompt(messages)
            
            prompts.append(prompt)
        
        # Generate responses for all prompts
        batch_results = self.model_interface.generate(prompts)
        
        # Extract responses
        for i, env_id in enumerate(env_ids):
            responses[env_id] = batch_results[i]["text"]
        
        return responses
    
    def _process_multimodal_messages(self, messages: List[Dict]) -> Dict:
        """
        Process messages with multimodal data for the model interface.
        
        Args:
            messages: List of message dictionaries
            
        Returns:
            Processed input for the model interface
        """
        # Extract images from messages
        all_images = []
        for message in messages:
            if "multi_modal_data" in message:
                for key, values in message["multi_modal_data"].items():
                    for value in values:
                        # Handle different image formats
                        if isinstance(value, PIL.Image.Image):
                            all_images.append(value)
                        # Handle serialized images from the service
                        elif isinstance(value, dict) and "__pil_image__" in value:
                            from vagen.server.serial import deserialize_pil_image
                            all_images.append(deserialize_pil_image(value))
        
        # Process images
        processed_images = self.model_interface.process_images(all_images)
        
        # Return formatted input
        return {
            "messages": messages,
            "images": processed_images
        }
    
    def recording_to_log(self) -> List[Dict]:
        """
        Format and return results in a format compatible with logging.
        
        Returns:
            List of dictionaries with:
            - env_id: Environment ID
            - config_id: Configuration ID
            - output_str: Formatted output string
            - image_data: List of images (if applicable)
            - metrics: Dictionary of metrics for this environment
        """
        results = []
        
        # Recursive function to convert all NumPy types to native Python types
        def convert_numpy_types(obj):
            import numpy as np
            
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return convert_numpy_types(obj.tolist())
            elif isinstance(obj, dict):
                return {k: convert_numpy_types(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy_types(i) for i in obj]
            elif isinstance(obj, tuple):
                return tuple(convert_numpy_types(i) for i in obj)
            else:
                return obj
        
        # Get final rewards for all environments
        reward_results = self.env_client.compute_reward_batch(list(self.envs.keys()))
        
        for env_id in self.envs:
            # Get environment configuration ID
            config_id = self.envs[env_id].config_id()
            
            # Get step count
            step_count = self.env_states[env_id]["step"]
            
            # Extract images from all messages
            image_data = []
            
            for message in self.recordings[env_id]:
                if "multi_modal_data" in message:
                    for key, values in message["multi_modal_data"].items():
                        for value in values:
                            # Handle different image formats
                            if isinstance(value, PIL.Image.Image):
                                image_data.append(value)
                            # Handle serialized images from the service
                            elif isinstance(value, dict) and "__pil_image__" in value:
                                from vagen.server.serial import deserialize_pil_image
                                image_data.append(deserialize_pil_image(value))
            
            # Format the conversation
            output_str = ""
            for msg in self.recordings[env_id]:
                role = msg["role"]
                content = msg["content"]
                
                if role == "system":
                    output_str += f"System: {content}\n\n"
                elif role == "user":
                    output_str += f"User: {content}\n\n"
                elif role == "assistant":
                    output_str += f"Assistant: {content}\n\n"
            
            # Get completion status
            done = self.env_states[env_id]["done"]
            
            # Get final score/reward
            score = convert_numpy_types(reward_results[env_id])
            
            # Collect metrics
            metrics = {
                "score": score,
                "done": 1.0 if done else 0.0,
                "step": convert_numpy_types(step_count),
            }
            
            # Add turn metrics (averaged)
            turn_metrics = self.env_states[env_id]["metrics"]["turn_metrics"]
            for k, v in turn_metrics.items():
                if v:
                    metrics[f"avg_{k}"] = convert_numpy_types(sum(v) / len(v))
            
            # Add trajectory metrics
            traj_metrics = self.env_states[env_id]["metrics"]["traj_metrics"]
            for k, v in traj_metrics.items():
                metrics[k] = convert_numpy_types(v)
            
            # Add to results
            results.append({
                "env_id": env_id,
                "config_id": config_id,
                "output_str": output_str,
                "image_data": image_data,
                "metrics": convert_numpy_types(metrics),
            })
        
        return results
    
    def close(self) -> None:
        """
        Close all environments and clean up resources.
        """
        # Close environments through service
        if self.envs:
            self.env_client.close_batch(list(self.envs.keys()))
        
        # Clear tracking structures
        self.envs = {}
        self.env_states = {}
        self.recordings = {}
        self.system_prompts = {}
        
        if self.debug:
            print("Closed all environments and cleaned up resources")
            

def run_test():
    """Run a simple test of the InferenceRolloutService."""
    print("=== Testing InferenceRolloutService ===")
    
    # Create a model interface
    model = MockModelInterface()
    
    # Create a configuration
    config = {
        "max_steps": 5,
        "show_progress": True,
        "debug": True
    }
    
    # Create the service
    service = InferenceRolloutService(
        config=config,
        model_interface=model,
        debug=True
    )
    
    # Create environment configurations
    env_configs = [
        {
            "env_name": "frozenlake",
            "env_config": {
                "size": 4,
                "is_slippery": False,
                "render_mode": "vision"
            },
            "seed": 42
        },
        {
            "env_name": "frozenlake",
            "env_config": {
                "size": 4,
                "is_slippery": True,
                "render_mode": "text"
            },
            "seed": 43
        }
    ]
    
    try:
        # Reset environments
        print("\nResetting environments...")
        service.reset(env_configs)
        
        # Run inference
        print("\nRunning inference...")
        service.run()
        
        # Get results
        print("\nGetting results...")
        results = service.recording_to_log()
        
        # Print results
        print("\n=== Results ===")
        for result in results:
            print(f"Environment: {result['env_id']}")
            print(f"Config ID: {result['config_id']}")
            print(f"Steps: {result['metrics']['step']}")
            print(f"Done: {result['metrics']['done']}")
            print(f"Score: {result['metrics']['score']}")
            print("Metrics:", {k: v for k, v in result['metrics'].items() if k not in ['step', 'done', 'score']})
            print("---")
    
    finally:
        # Clean up
        print("\nCleaning up...")
        service.close()
    
    print("=== Test completed ===")

if __name__ == "__main__":
    run_test()