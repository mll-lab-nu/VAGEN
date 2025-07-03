from vagen.env.base.base_env import BaseEnv
import numpy as np
import copy
from typing import Dict, List, Optional, Tuple, Any
from PIL import Image
from vagen.env.utils.env_utils import NoLoggerWarnings, set_seed
from vagen.env.utils.context_utils import convert_numpy_to_PIL
from vagen.env.utils.parse_utils import PARSE_FUNC_MAP
from .prompt import system_prompt, init_observation_template, action_template, format_prompt
from .env_config import BlackjackEnvConfig
from vagen.env.utils.state_reward_text_utils import env_state_reward_wrapper
from .blackjack import BlackjackEnv as GymBlackjackEnv 
from .blackjack import sum_hand, usable_ace, is_bust, is_natural

class BlackjackEnv(BaseEnv):
    """
    Blackjack Environment for training and evaluating language models as agents.
    
    This environment implements the classic Blackjack card game where an agent 
    must beat the dealer by getting closer to 21 without going over.
    """
    
    # Map action strings to Gymnasium action integers
    ACTION_LOOKUP = {
        "Hit": 1,
        "Stand": 0,
        "Stick": 0,  # Alternative name for Stand
    }
    
    def __init__(self, config: BlackjackEnvConfig):
        """Initialize the Blackjack environment."""
        BaseEnv.__init__(self)
        self.config = config
        
        # Initialize the underlying Gymnasium environment
        self.gym_env = GymBlackjackEnv(
            natural=self.config.natural,
            sab=self.config.sab,
            is_pixel=True  # Always use pixel mode for consistency
        )
        
        # Initialize episode state
        self.total_reward = 0
        self.valid_actions = []
        self.reward = 0
        
        # Store the format prompt function
        self.format_prompt_func = format_prompt[self.config.prompt_format]
        self.parse_func = PARSE_FUNC_MAP[self.config.prompt_format]

    def reset(self, seed=None):
        """Reset the environment to an initial state."""
        with NoLoggerWarnings():
            with set_seed(seed):
                self.gym_env.reset(seed=seed)
        self.total_reward = 0
        return self._render(init_obs=True), {}

    @env_state_reward_wrapper
    def step(self, action_str: str):
        """Take a step in the environment based on the agent's action."""
        # Parse the LLM's raw response to extract actions
        rst = self.parse_func(
            response=action_str,
            special_token_list=self.config.special_token_list,
            action_sep=self.config.action_sep,
            max_actions=self.config.max_actions_per_step
        )
        
        action_list = rst['actions']
        
        # Initialize metrics for this step
        metrics = {
            "turn_metrics": {
                "action_is_valid": len(action_list) != 0,
                "action_is_effective": False,
            },
            "traj_metrics": {
                "success": False,
            },
        }
        
        # Reset step-specific state
        self.reward = 0
        self.valid_actions = []
        done = False
        info = {}
        info.update(rst)
        
        # Execute the first valid action (Blackjack typically takes one action at a time)
        if action_list:
            action = action_list[0]  # Take first action only
            if action in self.ACTION_LOOKUP:
                # Convert string action to integer and execute in gym environment
                action_int = self.ACTION_LOOKUP[action]
                _, step_reward, terminated, truncated, gym_info = self.gym_env.step(action_int)
                self.reward = step_reward
                self.valid_actions.append(action)
                done = terminated or truncated
                
                # Action is always effective in Blackjack (changes game state)
                metrics["turn_metrics"]['action_is_effective'] = True
                
                # Check if successful (positive reward)
                if step_reward > 0:
                    metrics["traj_metrics"]['success'] = True
                
                # Update info with gym info
                info.update(gym_info)
            else:
                # Invalid action encountered
                metrics["turn_metrics"]['action_is_valid'] = False
                self.reward = -0.1  # Small penalty for invalid action
        else:
            # No valid actions parsed
            metrics["turn_metrics"]['action_is_valid'] = False
            self.reward = -0.1
        
        # Add format reward if actions were valid
        if metrics["turn_metrics"]['action_is_valid'] and rst["format_correct"]:
            self.reward += self.config.format_reward
            info["is_format_rewarded"] = True
        else:
            info["is_format_rewarded"] = False
        
        info["metrics"] = metrics
        self.total_reward += self.reward
        
        return self._render(init_obs=False), self.reward, done, info

    def system_prompt(self):
        """Get the system prompt for the environment."""
        format_prompt_text = self.format_prompt_func(
            max_actions_per_step=self.config.max_actions_per_step,
            action_sep=self.config.action_sep,
            add_example=True
        )
        
        return system_prompt(
            natural=self.config.natural,
            sab=self.config.sab
        ) + '\n' + format_prompt_text

    def close(self):
        self.gym_env.close()

    def _render(self, init_obs=False):
        """Render the environment as an observation."""
        multi_modal_data = None
        
        # Get format prompt without examples
        format_prompt_text = self.format_prompt_func(
            max_actions_per_step=self.config.max_actions_per_step,
            action_sep=self.config.action_sep,
            add_example=False
        )
        
        # Get current game state
        gym_obs = self.gym_env._get_obs()
        
        # Generate observation based on render mode
        if self.config.render_mode == 'vision':
            # For vision mode, use the gym environment's image
            img_placeholder = self.config.image_placeholder
            multi_modal_data = {
                img_placeholder: [convert_numpy_to_PIL(gym_obs)]
            }
            observation = img_placeholder
        else:
            # For text mode, create text representation
            player_sum = sum_hand(self.gym_env.player)
            dealer_showing = self.gym_env.dealer[0][0]
            has_usable_ace = "Yes" if usable_ace([card[0] for card in self.gym_env.player]) else "No"
            
            # Format player cards
            player_cards_str = ", ".join([
                f"{card[2]}{card[1]}" for card in self.gym_env.player
            ])
            
            # Format dealer showing card
            dealer_card = self.gym_env.dealer[0]
            dealer_showing_str = f"{dealer_card[2]}{dealer_card[1]}"
            
            observation = f"""Your hand: {player_cards_str} (Sum: {player_sum}, Usable Ace: {has_usable_ace})
Dealer showing: {dealer_showing_str} (Value: {dealer_showing})
Dealer has one hidden card.

Your goal: Get closer to 21 than the dealer without going over."""
        
        # Format the observation string
        if init_obs:
            obs_str = init_observation_template(observation=observation) + "\n" + format_prompt_text
        else:
            obs_str = action_template(
                valid_action=self.valid_actions,
                observation=observation,
            ) + "\n" + format_prompt_text
        
        # Return observation dictionary
        if multi_modal_data is not None:
            return {
                "obs_str": obs_str,
                "multi_modal_data": multi_modal_data,
            }
        else:
            return {
                "obs_str": obs_str,
            }

    def get_env_state(self):
        """Get the current state of the environment as a dictionary."""
        player_sum = sum_hand(self.gym_env.player)
        dealer_showing = self.gym_env.dealer[0][0]
        player_values = [card[0] for card in self.gym_env.player]
        
        return {
            "player_cards": self.gym_env.player,
            "dealer_cards": self.gym_env.dealer,
            "player_sum": player_sum,
            "dealer_showing": dealer_showing,
            "usable_ace": usable_ace(player_values),
            "is_player_bust": is_bust(self.gym_env.player),
            "is_natural": is_natural(self.gym_env.player)
        }

if __name__ == "__main__":
    """Example usage of the Blackjack environment."""
    kwargs = {
        'render_mode': 'vision',
        'natural': False,
        'sab': False
    }
    config = BlackjackEnvConfig(**kwargs)
    env = BlackjackEnv(config)
    
    print(env.system_prompt())
    obs, info = env.reset()
    print(obs["obs_str"])
    
    # Print current game state in text
    player_sum = sum_hand(env.gym_env.player)
    dealer_showing = env.gym_env.dealer[0][0]
    player_values = [card[0] for card in env.gym_env.player]
    has_usable_ace = usable_ace(player_values)
    
    print(f"\nCurrent Status:")
    print(f"Player hand: {player_sum} (Usable Ace: {'Yes' if has_usable_ace else 'No'})")
    print(f"Dealer showing: {dealer_showing}")
    
    i = 0
    import os
    if config.render_mode == 'vision':
        os.makedirs("./test_blackjack", exist_ok=True)
        img = obs["multi_modal_data"][config.image_placeholder][0]
        img.save(f"./test_blackjack/blackjack_{i}.png")
    
    print("\nAvailable actions: Hit, Stand")
    while True:
        i += 1
        action = input("Enter action (Hit/Stand): ")
        action = f"<think>Let me consider this move.</think><answer>{action}</answer>"
        
        obs, reward, done, info = env.step(action)
        print(obs["obs_str"])
        
        # Print updated game state in text
        player_sum = sum_hand(env.gym_env.player)
        dealer_showing = env.gym_env.dealer[0][0]
        player_values = [card[0] for card in env.gym_env.player]
        has_usable_ace = usable_ace(player_values)
        
        print(f"\nCurrent Status:")
        print(f"Player hand: {player_sum} (Usable Ace: {'Yes' if has_usable_ace else 'No'})")
        print(f"Dealer showing: {dealer_showing}")
        if done:
            # Show final dealer hand
            dealer_sum = sum_hand(env.gym_env.dealer)
            print(f"Final dealer hand: {dealer_sum}")
        
        print(f"Reward: {reward}, Done: {done}")
        
        if config.render_mode == 'vision':
            img = obs["multi_modal_data"][config.image_placeholder][0]
            img.save(f"./test_blackjack/blackjack_{i}.png")
        
        if done:
            if info['metrics']['traj_metrics']['success']:
                print("You won!")
            elif reward == 0:
                print("It's a tie!")
            else:
                print("You lost!")
            break
        
    print(f"Total reward: {env.total_reward}")
    print(f"Final metrics: {info['metrics']}")
    env.close()