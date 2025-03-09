import gym
from gym_sokoban.envs.sokoban_env import SokobanEnv as GymSokobanEnv
import numpy as np
import re
import copy
from typing import Tuple, Dict, Optional, List
from dataclasses import dataclass

from vagen.env.register import register
from vagen.utils import NoLoggerWarnings
from vagen.env.sokoban.room_utils import generate_room
from vagen.utils import set_seed
from vagen.env.base import (
    BaseEnv,
    BaseGame,
    PromptTemplate,
    EnvFeedback,
    EnvFeedbackSingleStep,
    EnvObservation,
)

system_prompt = """
You are a helpful assistant. You first think about the reasoning process in the mind and then provides the user with the answer.
"""

instruction_template = """You are a Sokoban solver.

Sokoban Quick Guide
Goal: Push all boxes (X) onto targets (O).

Symbols:
# Wall | _ Floor | O Target | X Box | P You | √ = Box on Target | S = You on Target
The observation is a 2D grid of the current state of the Sokoban game.

Rules:
1. Push boxes (can't pull).
2. Avoid walls (#).

Actions you can take: Up, Down, Left, Right. You can only take one action at a time.
Up: move up to the cell above (to the above row).
Down: move down to the cell below (to the below row).
Left: move left to the cell to the left (to the left column).
Right: move right to the cell to the right (to the right column).

Rewards:
Move: -0.1
Box on target: +1.0
All boxes placed: +10.0
"""

init_observation_template = """
[Initial Observation]:
{observation}
Decide your next action.
"""

valid_action_template = """After you move {action}, the observation is: 
{observation}
reward: {reward}
done: {done}
"""

invalid_action_template = """Action is invalid. You stay in the same position. The observation is: 
{observation}
reward: {reward}
done: {done}
"""

template = PromptTemplate(
    system_prompt=system_prompt,
    instruction_prompt=instruction_template,
    init_observation_template=init_observation_template,
    valid_action_template=valid_action_template,
    invalid_action_template=invalid_action_template,
)


class SokobanEnv(BaseEnv, GymSokobanEnv):

    GRID_LOOKUP = {
        0: " # \t",  # wall
        1: " _ \t",  # floor
        2: " O \t",  # target
        3: " √ \t",  # box on target
        4: " X \t",  # box
        5: " P \t",  # player
        6: " S \t",  # player on target
        # Use tab separator to separate columns and \n\n to separate rows.
    }

    ACTION_LOOKUP = {
        0: "None",
        1: "Up",
        2: "Down",
        3: "Left",
        4: "Right",
    }

    def __init__(self, **kwargs):
        BaseEnv.__init__(self)
        self.search_depth = kwargs.pop('search_depth', 300)
        GymSokobanEnv.__init__(
            self,
            dim_room=kwargs.pop('dim_room', (6, 6)), 
            max_steps=kwargs.pop('max_steps', 100),
            num_boxes=kwargs.pop('num_boxes', 3),
            **kwargs
        )
        self.ACTION_SPACE = gym.spaces.discrete.Discrete(4, start=1)


    def reset(self, mode='text', seed=None):
        with NoLoggerWarnings():
            try:
                with set_seed(seed):
                    self.room_fixed, self.room_state, self.box_mapping, action_sequence = generate_room(
                        dim=self.dim_room,
                        num_steps=self.num_gen_steps,
                        num_boxes=self.num_boxes,
                        search_depth=self.search_depth
                    )
            except (RuntimeError, RuntimeWarning) as e:
                print("[SOKOBAN] Runtime Error/Warning: {}".format(e))
                print("[SOKOBAN] Retry . . .")
                next_seed = abs(hash(str(seed))) % (2 ** 32) if seed is not None else None
                return self.reset(mode, next_seed)
            
            # self.action_sequence = self._reverse_action_sequence(action_sequence)
            self.player_position = np.argwhere(self.room_state == 5)[0]
            self.num_env_steps = self.reward_last = self.boxes_on_target = 0
        

    def finished(self):
        return self.num_env_steps >= self.max_steps or self.success()

    def success(self):
        return self.boxes_on_target == self.num_boxes
    
    def step(self, action: int):
        """
        - Step the environment with the given action.
        - Check if the action is effective (whether player moves in the env).

        TODO modify here after definition of RolloutManager
        """
        assert not self.success()
        result = {
            'step_reward': 0,
            'done': False,
            'info': {},
        }
        
        prev_player_position = self.player_position
        _, step_reward, done, _ = GymSokobanEnv.step(self, action, observation_mode='rgb_array')
        
        result['step_reward'] = step_reward
        result['done'] = done
        result['info'] = {"action_is_effective": not np.array_equal(prev_player_position, self.player_position)}
        return result
     

    def render(self, mode='text'):
        assert mode in ['text', 'list', 'state', 'rgb_array']

        if mode == 'rgb_array':
            img = self.get_image(mode, scale=1) # numpy array
            return img


        if mode == 'state':
            return np.where((self.room_state == 5) & (self.room_fixed == 2), 6, self.room_state)
        
        room_state = self.render(mode='state').tolist()

        if mode == 'list':
            lookup = lambda cell: self.GRID_LOOKUP.get(cell, "?").strip("\t").strip()
            return [" ".join(lookup(cell) for cell in row) for row in room_state]
        
        if mode == 'text':
            lookup = lambda cell: self.GRID_LOOKUP.get(cell, "?")
            return "\n".join("".join(lookup(cell) for cell in row) for row in room_state)
    
        
    def copy(self):
        new_self = SokobanEnv(
            dim_room=self.dim_room,
            max_steps=self.max_steps,
            num_boxes=self.num_boxes,
            search_depth=self.search_depth
        )
        new_self.room_fixed = self.room_fixed.copy()
        new_self.room_state = self.room_state.copy()
        new_self.box_mapping = self.box_mapping.copy()
        new_self.action_sequence = self.action_sequence.copy()
        new_self.player_position = self.player_position.copy()
        new_self.reward = self.reward
        new_self._valid_actions = copy.deepcopy(self._valid_actions)
        return new_self
    

    def set_state(self, rendered_state):
        # from the rendered state, set the room state and player position
        self.room_state = np.where(rendered_state == 6, 5, rendered_state)
        self.player_position = np.argwhere(self.room_state == 5)[0]

    def close(self):
        GymSokobanEnv.close(self)




@dataclass
class PreprocessResult:
    action: List[int]
    action_valid: List[bool]
    extracted_answer: List[str]
    raw_text: str

    def to_dict(self):
        return {
            'action': self.action,
            'action_valid': self.action_valid,
            'extracted_answer': self.extracted_answer,
            'raw_text': self.raw_text,
        }


@register(name="sokoban")
class SokobanGame(BaseGame):

    INVALID_ACTION = 0
    PROMPT_TEMPLATE = template
    PENALTY_FOR_INVALID = -1

    def __init__(
            self,
            **env_config,
        ):
        super().__init__(**env_config)

        dim_room = self.env_config['dim_room']
        num_boxes = self.env_config['num_boxes']
        max_steps = self.env_config['max_steps']
        search_depth = self.env_config['search_depth']
        self.env = SokobanEnv(
            dim_room=dim_room,
            num_boxes=num_boxes,
            max_steps=max_steps,
            search_depth=search_depth
        )
        self.visual_env = self.env_config.get('visual_env', True)
        
    @classmethod
    def _extract_action(cls, text):
        """
        Extract action from text.
        - 0: Still (Invalid Action)
        - 1: Up
        - 2: Down
        - 3: Left
        - 4: Right
        """
        DIRECTION_MAP = {"Up": 1, "Down": 2, "Left": 3, "Right": 4}
        # TODO: originally, we parse either number (key of direction_map) or direction (value of direction_map).
        # here we remove numbers and preserve directions only, but regex has not been removed. please remove them later.
        pattern = r'^\s*(([1-4])\s*\((up|down|left|right)\)|(up|down|left|right)|([1-4]))\s*$'
        match = re.fullmatch(pattern, text.strip(), flags=re.IGNORECASE | re.X)
        
        if not match:
            return cls.INVALID_ACTION
        
        if match.group(2):   
            return int(match.group(2))
        elif match.group(4): 
            return DIRECTION_MAP[match.group(4).capitalize()]
        elif match.group(5): 
            return int(match.group(5))
        
        return cls.INVALID_ACTION
    
    def _get_observation(self):
        """
        Get the observation of the environment.
        If visual_env is True, return the visual observation (PIL RGBA image).
        """
        if self.visual_env:
            visual_observation = self.env.render('rgb_array')
            if isinstance(visual_observation, np.ndarray):
                visual_observation = self.convert_numpy_to_PIL(visual_observation)
            return {
                'text': self.env.render('text'),
                'visual': visual_observation,
            }
        else:
            return {
                'text': self.env.render('text'),
            }
    
    @classmethod
    def _preprocess(cls, text: str) -> PreprocessResult:
        """Preprocess the raw text from LLM into a list of actions.
        Ensure at least one action (may be invalid).

        Args:
            text: raw text from LLM

        Returns:
            PreprocessResult containing parsed actions and validity flags
        """
        preprocess_result = PreprocessResult(
            action=[],
            action_valid=[],
            extracted_answer=[],
            raw_text=text,
        )

        # 1. Extract answer from <answer>...</answer>
        match = re.search(r"<answer>(.*?)</answer>", text, re.DOTALL)
        if not match:
            # No valid answer format found
            preprocess_result.action = [cls.INVALID_ACTION]
            preprocess_result.action_valid = [False]
            preprocess_result.extracted_answer = [""]
            return preprocess_result

        answer = match.group(1).strip()
        preprocess_result.extracted_answer = [answer]

        # 2. Extract actions from answer
        # Split by newlines, commas, or semicolons to handle multiple actions
        action_texts = re.split(r'[,;\n]+', answer)
        
        valid_actions_found = False
        for action_text in action_texts:
            action_text = action_text.strip()
            if not action_text:
                continue
                
            action = cls._extract_action(action_text)
            if action != cls.INVALID_ACTION:
                preprocess_result.action.append(action)
                preprocess_result.action_valid.append(True)
                valid_actions_found = True
            else:
                preprocess_result.action.append(cls.INVALID_ACTION)
                preprocess_result.action_valid.append(False)
        
        # If no valid actions were found, return a single invalid action
        if not valid_actions_found and not preprocess_result.action:
            preprocess_result.action = [cls.INVALID_ACTION]
            preprocess_result.action_valid = [False]
            preprocess_result.extracted_answer = [""]
        
        return preprocess_result
        
    @classmethod
    def _postprocess(
        cls, 
        env_init: bool = False,
        action_valid: bool = True,
        observation: Dict = {},
        reward: float = 0,
        done: bool = False,
        info: Dict = {},
    ) -> EnvFeedbackSingleStep:
        """
        Postprocess the observation from environment to feedback for LLM.
        The returned observation_template is a string with placeholder,
            and multi_modal_observation defines mapping from placeholder to multi-modal observation.
        """
        env_observation = EnvObservation()

        if env_init:
            observation_template = cls.PROMPT_TEMPLATE.init_observation_template
            env_observation.create_observation(
                template=observation_template,
                contents=[observation['visual']],
                replace_keys=['{observation}']
            )
        else:
            if not action_valid:
                observation_template = cls.PROMPT_TEMPLATE.invalid_action_template
            else:
                observation_template = cls.PROMPT_TEMPLATE.valid_action_template
            
            env_observation.create_observation(
                template=observation_template,
                contents=[observation['visual'], reward, done],
                replace_keys=['{observation}', '{reward}', '{done}']
            )

        
        
        
        return EnvFeedbackSingleStep(
            step_observation = env_observation,
            step_reward = reward,
            step_done = done,
            step_info = info,
        )
    

    def step(self, raw_text: str) -> EnvFeedback:

        assert not self.finished(), "Environment finished before step"

        preprocess_result = self._preprocess(raw_text)
        env_feedback = EnvFeedback()
        actions = preprocess_result.action
        action_valid = preprocess_result.action_valid
        env_feedback.llm_raw_response = raw_text


        for action, valid in zip(actions, action_valid):
            if valid:
                step_result = self.env.step(action)
                reward = step_result['step_reward']
                done = step_result['done']
                info = step_result['info']
            else:
                reward = self.PENALTY_FOR_INVALID
                done = False
                info = {} # TODO
                action = self.INVALID_ACTION
            self.traj_reward += reward

            observation = self._get_observation()
            env_feedback_single_step = self._postprocess(
                action_valid=valid,
                observation=observation, 
                reward=reward, 
                done=done, 
                info={
                    'action_valid': valid,
                    'action_str': self.env.ACTION_LOOKUP[action],
                    **info,
                }
            )
            env_feedback.add_step(env_feedback_single_step)
            if done or self.finished():
                break

        return env_feedback
    
    def reset(self, seed: Optional[int] = None) -> EnvFeedback:
        """
        Reset the environment and return the observation at the first step.
        """
        self.env.reset(seed=seed)
        self.traj_reward = 0
        observation = self._get_observation()
        step_feedback = self._postprocess(
            env_init=True,
            observation=observation,
        )
        return EnvFeedback(env_feedbacks=[step_feedback])

    def finished(self) -> bool:
        return self.env.finished()

    def success(self) -> bool:
        return self.env.success()

    def close(self):
        self.env.close()

    @classmethod
    def config_repr(cls, config: Dict) -> str:
        """
        Create a string representation of the configuration.
        
        Args:
            config: Dictionary containing configuration
            
        Returns:
            String representation of the configuration
            
        Raises:
            ValueError: If required keys are missing from the configuration
        """
        required_keys = ['dim_room', 'num_boxes', 'max_steps', 'search_depth']
        
        # Check for required keys
        if not all(key in config for key in required_keys):
            missing_keys = [key for key in required_keys if key not in config]
            raise ValueError(f"Missing required keys in config: {missing_keys}")
            
        # Format the configuration string
        return (f"SokobanGame(dim_room={config['dim_room']}, "
                f"num_boxes={config['num_boxes']}, "
                f"max_steps={config['max_steps']}, "
                f"search_depth={config['search_depth']})")










GUIDE = """
### Sokoban Puzzle Instructions

In Sokoban, your goal is to move all the boxes to the target spots on the grid. This requires careful planning and strategic moves. Here's how it works:

---

#### Symbols and Their Meaning
- **Walls (`#`)**: These block movement. You can't move through or push anything into walls.
- **Floor (`_`)**: Open spaces where you can walk and move boxes.
- **Targets (`O`)**: The spots where boxes need to go.
- **Boxes (`X`)**: These are what you need to push onto the targets.
- **Player (`P`)**: That's you! You'll move around the grid to push boxes.
- **Box on Target (`√`)**: A box successfully placed on a target.
- **Player on Target (`S`)**: You standing on a target.

---

#### Your Goal
Push all the boxes (`X`) onto the target spots (`O`). Once all boxes are on targets, you win!

---

#### Rules to Remember
1. **You Can Only Push Boxes**: You can't pull them, so plan ahead to avoid getting stuck.
2. **No Moving Through Walls**: You can't walk through or push boxes into walls (`#`).
3. **Avoid Traps**: Don't push boxes into corners or against walls where they can't be moved again.

---

#### Controls
Use these outputs to move the player:
- `1`: Move **up**.
- `2`: Move **down**.
- `3`: Move **left**.
- `4`: Move **right**.

#### Rewards
- **Move**: Each step you take costs 0.1.
- **Push Box to Target**: Each box placed on a target gives you 1.0.
- **Achieve Goal**: When all boxes are on targets, you get a reward of 10.0.

---

#### Example Map
Here's an example of a Sokoban puzzle:

# 	 # 	 # 	 # 	 # 	 # 	 # 	 
# 	 _ 	 _ 	 # 	 # 	 # 	 # 	 
# 	 _ 	 # 	 # 	 # 	 O 	 # 	 
# 	 _ 	 _ 	 _ 	 O 	 _ 	 # 	 
# 	 _ 	 X 	 X 	 _ 	 _ 	 # 	 
# 	 _ 	 O 	 _ 	 X 	 P 	 # 	 
# 	 # 	 # 	 # 	 # 	 # 	 # 	 

Each puzzle will have a different layout, but the rules and goal remain the same.

---

#### Tips for Beginners
1. **Move Boxes Step by Step**: Push them one at a time toward the targets.
2. **Think Ahead**: Avoid pushing a box into a spot where you can’t move it again.

Enjoy the challenge!
"""

if __name__ == '__main__':
    # Test SokobanGame
    import matplotlib.pyplot as plt
    
    # Create a SokobanGame instance
    game = SokobanGame(dim_room=(6, 6), num_boxes=1, max_steps=100, search_depth=30)
    game.reset(seed=0)
    
    # # Test observation retrieval
    obs = game._get_observation()
    print("Initial game state:")
    print(obs['text'])
    
    # Save the initial visual observation
    plt.imsave('sokobangame_initial.png', obs['visual'])
    
    # # Test action extraction
    # print("\nTesting action extraction:")
    # test_inputs = ["Up", "Down", "Left", "Right", "1", "2", "3", "4", 
    #               "1 (up)", "2 (down)", "3 (left)", "4 (right)", "invalid input"]
    
    # for input_text in test_inputs:
    #     action = game._extract_action(input_text)
    #     print(f"'{input_text}' -> {action} ({game.env.ACTION_LOOKUP.get(action, 'Invalid')})")
    
    # # Test preprocessing
    # print("\nTesting preprocessing:")
    # test_text = "I'll move Right to push the box. <answer>Right</answer> This should move the player to the right and possibly push a box."
    
    # result = game._preprocess(test_text)
    # print(f"Extracted action: {result['action']}")
    # print(f"Action valid: {result['action_valid']}")
    # print(f"Extracted answer: {result['extracted_answer']}")
    
    # Test game step
    print("\nTesting game step with valid action:")
    # Assuming 'Right' is a valid action (4)
    feedback = game.step("<answer>down, down</answer>")[1]
    print(f"Observation after step: \n{feedback.observation_template}")
    print(f"Reward: {feedback.step_reward}, Done: {feedback.done}")
    
    # Save the visual observation after step
    feedback.multi_modal_observation['<image1>'].save('sokobangame_after_step_pil.png')
