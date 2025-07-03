# How to Create New Environments

This guide explains how to create new environments for VAGEN using **Blackjack** as an example. Understanding the BaseEnv interface is key to building effective LLM training environments.

> For the specific rules of Blackjack, please refer to [Blackjack Rules](https://en.wikipedia.org/wiki/Blackjack).
> For the gym-formated Blackjack details, please refer to [Blackjack-v0](https://gymnasium.farama.org/environments/toy_text/blackjack/).

## Directory Structure

```
vagen/env/blackjack/
├── env.py                # BlackjackEnv - main environment wrapper
├── env_config.py         # BlackjackEnvConfig - configuration class
├── prompt.py             # Prompt templates and format configurations
├── blackjack.py          # Core gym environment (standard gym interface)
└── __init__.py           # Environment registration
```

**File Responsibilities:**
- `blackjack.py`（Optional）: Your core game logic （usually a standard gym environment） (`step`, `reset`)
- `env_config.py`: Configuration parameters and settings for your environment
- `env.py`: VAGEN wrapper that bridges LLM responses to your game logic
- `prompt.py`: System prompts and LLM interaction format definitions
- `__init__.py`: Registration info to make your environment discoverable by VAGEN

## Understanding BaseEnv Interface

VAGEN environments inherit from `BaseEnv`, which defines the contract between your game logic and the LLM training system. Here's what each required method does:

### Core Methods Overview

**`step(llm_raw_response)`** - The heart of LLM interaction
- Takes the raw text response from the LLM (e.g., `"<think>I should hit</think><answer>Hit</answer>"`)
- Parses it to extract valid actions (e.g., `["Hit"]`)
- Executes actions in your game
- Returns the next observation, reward, completion status, and metrics

**`reset(seed)`** - Initialize a new episode (these seeds are read from train/test parquet file)
- Resets the game to starting state
- Uses seed for reproducible episodes
- Returns initial observation for the LLM

**`system_prompt()`** - Define the LLM's role
- Returns the system prompt that tells the LLM what game it's playing
- Includes rules, available actions, and formatting instructions

**`close()`** - Clean up resources
- Called when the environment is no longer needed

**`compute_reward()`** - Optional final reward
- Usually returns 0.0 since step rewards are accumulated
- Use only if you need extra reward at episode end

## Key Data Structures

### Observation Format
Every observation must follow this structure:
```python
{
    'obs_str': "You see <image> showing your cards. The dealer shows <image>.",
    'multi_modal_data': {
        '<image>': [player_cards_image, dealer_card_image],
        '<audio>': [shuffle_sound]  # optional
    }
}
```
The number of `<image>` placeholders in `obs_str` must match the length of the image list.

### Info Dictionary
Provides metrics and context for training:
```python
{
    "metrics": {
        'success': bool,              # Did LLM complete the task?
        'action_is_effective': bool,  # Did action change game state meaningfully?
        'action_is_valid': bool,      # Was action format correct?
    },
    "llm_raw_response": str,  # Original LLM response
    "llm_response": dict,     # Parsed action structure
}
```

## Implementation Components

Your environment needs three main files:

### 1. Environment Configuration (`env_config.py`)
Defines all parameters for your environment:

```python
from vagen.env.base.base_env_config import BaseEnvConfig
from dataclasses import dataclass

@dataclass
class BlackjackEnvConfig(BaseEnvConfig):
    env_name: str = "blackjack"
    render_mode: str = "vision"  # "text" or "vision"
    natural: bool = False        # Game-specific parameter
    max_actions_per_step: int = 1    # For Blackjack, we only allow one action per step, since it needs to interact with the dealer
    prompt_format: str = "free_think"
    
    def config_id(self) -> str:
        return f"BlackjackEnvConfig(mode={self.render_mode},format={self.prompt_format})"
```

### 2. Environment Implementation (`env.py`)
The main environment class implementing BaseEnv:

```python
from vagen.env.base.base_env import BaseEnv
from .blackjack import BlackjackEnv as GymBlackjackEnv  # Your gym environment
from .env_config import BlackjackEnvConfig

class BlackjackEnv(BaseEnv):
    def __init__(self, config: BlackjackEnvConfig):
        self.config = config
        self.gym_env = GymBlackjackEnv()  # Your underlying game
        # Initialize parsers and prompt functions...
    
    def step(self, llm_raw_response: str):
        # 1. Parse LLM response to extract actions
        parsed = self.parse_func(llm_raw_response)
        actions = parsed['actions']
        
        # 2. Execute actions in your game
        reward = 0
        done = False
        if actions and actions[0] in self.ACTION_LOOKUP:
            action_int = self.ACTION_LOOKUP[actions[0]]
            _, reward, done, _ = self.gym_env.step(action_int)
        
        # 3. Create metrics
        metrics = {
            "turn_metrics": {
                "action_is_valid": len(actions) > 0,
                "action_is_effective": reward != 0,
            },
            "traj_metrics": {
                "success": done and reward > 0,
            }
        }
        
        # 4. Generate next observation
        obs = self._render()
        info = {"metrics": metrics, "llm_raw_response": llm_raw_response}
        
        return obs, reward, done, info
    
    def reset(self, seed=None):
        self.gym_env.reset(seed=seed)
        return self._render(init_obs=True), {}
    
    def system_prompt(self):
        return "You are a Blackjack player. Actions: Hit, Stand. Goal: Beat dealer without busting."
    
    def _render(self, init_obs=False):
        # Generate observation based on render_mode
        if self.config.render_mode == 'vision':
            image = self.gym_env.render()
            return {
                'obs_str': "Current game state: <image>",
                'multi_modal_data': {'<image>': [image]}
            }
        else:
            text_desc = f"Your hand: {self.gym_env.player_sum}, Dealer: {self.gym_env.dealer_card}"
            return {'obs_str': text_desc}
    
    def close(self):
        self.gym_env.close()
```

### 3. Prompt Templates (`prompt.py`)
Defines how LLM interacts with your environment:

```python
def system_prompt(**kwargs):
    return """You are a Blackjack player.
Goal: Get closer to 21 than dealer without going over.
Actions: "Hit" (take card), "Stand" (keep hand)"""

def init_observation_template(observation="", **kwargs):
    return f"[Initial Hand]: {observation}\nDecide: Hit or Stand?"

# Format configurations for different reasoning types
FORMAT_CONFIGS = {
    "free_think": {
        "format": "<think>...</think><answer>...</answer>",
        "example": "<think>I have 16, dealer shows 10. Risky but need to improve.</think><answer>Hit</answer>"
    },
    "grounding": {
        "format": "<think><observation>...</observation><reasoning>...</reasoning></think><answer>...</answer>",
        "example": "<think><observation>Hand: 16, dealer: 10</observation><reasoning>Tough spot, hitting gives chance</reasoning></think><answer>Hit</answer>"
    }
}

def format_prompt_generator(format_type):
    def prompt_function(**kwargs):
        config = FORMAT_CONFIGS[format_type]
        return f"Respond in format: {config['format']}\ne.g. {config['example']}"
    return prompt_function

format_prompt = {fmt: format_prompt_generator(fmt) for fmt in FORMAT_CONFIGS}
```

## Environment Registration

Register your environment in `__init__.py`:

```python
from .env import BlackjackEnv
from .env_config import BlackjackEnvConfig

BLACKJACK_ENV_INFO = {
    "env_cls": BlackjackEnv,
    "config_cls": BlackjackEnvConfig,
    "description": "Classic Blackjack card game"
}

# Add to vagen/env/__init__.py:
# REGISTERED_ENV["blackjack"] = BLACKJACK_ENV_INFO
```

## Testing Your Environment

```python
# Basic functionality test
config = BlackjackEnvConfig(render_mode="text")
env = BlackjackEnv(config)

# Test reset
obs, info = env.reset(seed=42)
print("Initial obs:", obs['obs_str'])

# Test step with LLM response
response = "<think>Let me be conservative</think><answer>Stand</answer>"
next_obs, reward, done, info = env.step(response)

print("Valid action:", info['metrics']['action_is_valid'])
print("Effective action:", info['metrics']['action_is_effective'])
print("Success:", info['metrics']['success'])

env.close()
```

## Common Patterns

**Multi-modal environments**: Include images/audio in `multi_modal_data`
**Text-only environments**: Set `multi_modal_data = None` in observations  
**Multi-step episodes**: Track progress in `self.step_count` or similar
**Complex action spaces**: Parse multiple actions from LLM response
**Custom metrics**: Add domain-specific metrics to the metrics dictionary

The key is understanding that `step()` bridges the gap between raw LLM text and your game logic, while maintaining consistent observation and metric formats for the training system.