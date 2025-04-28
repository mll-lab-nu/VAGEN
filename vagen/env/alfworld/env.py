from vagen.env.base.base_env import BaseEnv
from vagen.env.utils.context_utils import parse_llm_raw_response, convert_numpy_to_PIL
from alfworld.agents.utils.misc import get_templated_task_desc
from .env_config import ALFWorldEnvConfig
from .prompt import system_prompt_text, system_prompt_vision, init_observation_template, action_template

import alfworld.agents.environment
import numpy as np
import torch

class ALFWorldEnv(BaseEnv):
    """ALFWorld environment adapter that maps the BaseEnv interface to ALFWorld interface"""
    
    def __init__(self, config: ALFWorldEnvConfig):
        """Initialize the ALFWorld environment"""
        super().__init__()
        self.config = config
        
        # Load ALFWorld config
        import yaml
        with open(self.config.alf_config_path) as reader:
            alf_config = yaml.safe_load(reader)
        
        # @TODO Force TextWorld environment type
        alf_config['env']['type'] = 'AlfredTWEnv'
        
        # Initialize TextWorld environment
        env = alfworld.agents.environment.AlfredTWEnv(alf_config, train_eval='train')
        self.env = env.init_env(batch_size=1)
        
        # Track state
        self.total_reward = 0
        self.prev_admissible_commands = None
        self.valid_actions = []
    
    def step(self, llm_raw_response):
        """Process LLM response and take a step in the environment
        
        Args:
            llm_raw_response: Raw text response from LLM
            
        Returns:
            Observation, reward, done, info
        """
        # Parse LLM response
        parsed = parse_llm_raw_response(
            response=llm_raw_response,
            special_token_list=self.config.special_token_list,
            action_sep=self.config.action_sep,
            max_actions=self.config.max_actions_per_step
        )
        
        # Extract and process action
        action_list = parsed['actions']
        action_text = action_list[0] if action_list else ""
        
        # Store valid action for observation formatting
        self.valid_actions = [action_text] if action_text else []
        
        # Check if action is valid
        action_is_valid = action_text in self.prev_admissible_commands
        
        # Take the step in ALFWorld env
        obs, reward, done, infos = self.env.step([action_text])
        
        # Simple tracking of state change (text environments don't have position)
        action_is_effective = len(obs[0]) > 10  # Basic check if we got a meaningful observation
        
        # Build observation and info
        observation = self._render(obs, infos)
        
        # Check if metrics are available in infos
        # Some environments might not provide all metrics
        success = False
        goal_condition_rate = 0.0
        
        if 'won' in infos:
            success = float(infos['won'][0]) if isinstance(infos['won'], (list, tuple)) else float(infos['won'])
        
        if 'goal_condition_success_rate' in infos:
            goal_condition_rate = float(infos['goal_condition_success_rate'][0]) if isinstance(infos['goal_condition_success_rate'], (list, tuple)) else float(infos['goal_condition_success_rate'])
        
        metrics = {
            "turn_metrics": {
                "action_is_valid": action_is_valid,
                "action_is_effective": action_is_effective,
            },
            "traj_metrics": {
                "success": success,
                "goal_condition_success_rate": goal_condition_rate
            },
        }
        
        info = {
            "metrics": metrics,
            "llm_raw_response": llm_raw_response,
            "llm_response": parsed
        }
        
        # Update total reward
        if isinstance(reward, tuple):
            reward_value = reward[0]
        elif isinstance(reward, (list, np.ndarray)):
            reward_value = reward[0]
        else:
            reward_value = reward
        self.total_reward += reward_value
        
        # Update admissible commands for next step
        self.prev_admissible_commands = infos['admissible_commands'][0]
        
        # Convert done to boolean if it's a list or array
        done_value = done[0] if isinstance(done, (list, np.ndarray)) else done
        
        return observation, reward_value, done_value, info
    
    def reset(self, seed=None):
        """Reset the environment
        
        Args:
            seed: Random seed for reproducibility
            
        Returns:
            Observation dict, info dict
        """
        # Handle seed manually if provided @TODO figure out better random way
        if seed is not None:
            import random
            random.seed(seed)
            
            np.random.seed(seed)
            
            if torch:
                torch.manual_seed(seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed(seed)
                
        obs, infos = self.env.reset()
        self.total_reward = 0
        self.prev_admissible_commands = infos['admissible_commands'][0]
        self.valid_actions = []
        return self._render(obs, infos, init_obs=True), infos
    
    def system_prompt(self):
        """Generate system prompt
        
        Returns:
            System prompt string
        """
        if self.config.render_mode == "vision":
            return system_prompt_vision.format(
                max_actions_per_step=self.config.max_actions_per_step,
                action_sep=self.config.action_sep
            )
        else:
            return system_prompt_text.format(
                max_actions_per_step=self.config.max_actions_per_step,
                action_sep=self.config.action_sep
            )
    
    def compute_reward(self):
        """Return total reward
        
        Returns:
            Total reward for the episode
        """
        return self.total_reward
    
    def close(self):
        """Close the environment and release resources"""
        self.env.close()
    
    def _render(self, obs, infos, init_obs=False):
        """Render the environment as observation
        
        This method creates a text representation of the environment state.
        In the future, it could be extended to support visual rendering.
        
        Args:
            obs: Raw observations from ALFWorld
            infos: Additional information from environment
            init_obs: Whether this is the initial observation
            
        Returns:
            Dict: Observation dictionary
        """
        # Get the observation text
        observation_text = obs[0]
        
        # Format the list of admissible commands
        commands_text = "\n".join([f"'{s}'" for s in self.prev_admissible_commands]) if self.prev_admissible_commands else ""
        
        # Select appropriate template based on whether this is initial observation
        if init_obs:
            obs_str = init_observation_template.format(
                observation=observation_text,
                commands=commands_text
            )
        else:
            # For non-initial observations, include action results
            obs_str = action_template.format(
                valid_action=self.valid_actions[0] if self.valid_actions else "None",
                observation=observation_text,
                commands=commands_text,
                reward=self.total_reward
            )
        
        # For text mode, just return the observation string
        if self.config.render_mode == "text":
            return {
                "obs_str": obs_str
            }
        # @TODO
        else:
            img_placeholder = self.config.image_placeholder
            return {
                "obs_str": obs_str,
                "multi_modal_data": {
                }
            }