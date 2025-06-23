def system_prompt(**kwargs):
    return """You are a FrozenLake solver.
FrozenLake Quick Guide
Goal: Reach the goal (G).
Symbols (If image is provided there are no symbols):
_ Frozen | O Hole | G Goal | P Player | X Player fell into hole | √ Player on goal
Rules:
1. Avoid falling into holes.
2. Frozen tiles are slippery, you may move perpendicular to your intended direction.
Actions you can take: Left, Down, Right, Up. 
"""

def init_observation_template(**kwargs):
    observation = kwargs.get("observation", "The player is on the above the target")
    return f"""[Initial Observation]:
{observation}
Decide your next action(s).
"""

def action_template(**kwargs):
    valid_action, observation= kwargs.get("valid_action", "Down"), kwargs.get("observation", "The player is on the above the target")
    return f"""After your answer, the extracted valid action is {valid_action}.
After that, the observation is:
{observation}
Decide your next action(s).
"""

# Format configurations defining the structure of each format
FORMAT_CONFIGS = {
    "free_think": {
        "format": "<think>...</think><answer>...</answer>",
        "description": "You should first give your reasoning, and then your answer.",
        "example": "<think>I can see the target is on my down left, I should go down then left to reach the target</think><answer>Down{action_sep}Left</answer>"
    },
    
    "no_think": {
        "format": "<answer>...</answer>",
        "description": "You should provide only your answer.",
        "example": "<answer>Down{action_sep}Left</answer>"
    },
    
    "grounding": {
        "format": "<think><observation>...</observation><reasoning>...</reasoning></think>",
        "description": "You should first describe the observation, then your reasoning, and finally your answer.",
        "example": "<think><observation>The player is on the above the target</observation><reasoning>I should go down then left to reach the target</reasoning></think><answer>Down{action_sep}Left</answer>",
        "additional_info": "Inside the <observation> tags, describe the position of the target (gift box) and each hole (blue) relative to the player. For each object, you must specify both its vertical (above, below, same) and horizontal (left, right, same) direction.",
    },
    
    "worldmodeling": {
        "format": "<think><reasoning>...</reasoning><prediction>...</prediction></think><answer>...</answer>",
        "description": "You should first give your reasoning, then predict the next state, and finally your answer.",
        "example": "<think><reasoning>I can see the target is on my down left, I should go down then left</reasoning><prediction>The player will reach the target</prediction></think><answer>Down{action_sep}Left</answer>",
        "additional_info": "Inside the <prediction> tags, describe the position of the target (gift box) and each hole (blue) relative to the player. For each object, you must specify both its vertical (above, below, same) and horizontal (left, right, same) direction.",
    },
    
    "grounding_worldmodeling": {
        "format": "<think><observation>...</observation><reasoning>...</reasoning><prediction>...</prediction></think><answer>...</answer>",
        "description": "You should first describe the observation, then your reasoning, then predict the next state, and finally your answer.",
        "additional_info": "For the content you provide within the `<observation>` and `<prediction>` tags, you must strictly describe the relative position of the `target` (the gift box) and any visible `hole` (blue circles) objects **relative to the player**. Your description/prediction must include **both** a vertical and a horizontal directional relationship for each object. Use ONLY the terms `above`, `below`, `left`, `right`, or `same` for describing these relationships.",
        "example": "<think><observation>The player is above and on the right side of target. There is a hole below and at the left of the player</observation><reasoning>I should go down twice first to reach th same row as the target</reasoning><prediction>The player will be in the same row and to the right of the target.</prediction></think><answer>Down{action_sep}Down</answer>"
    },
    
    "grounding_symbolic": {
        "format": "<think><observation>...</observation><reasoning>...</reasoning></think><answer>...</answer>",
        "description": "You should first describe the observation as a grid, then your reasoning, and finally your answer.",
        "additional_info": "The observation should be represented as a grid using the symbols: _ Frozen | O Hole | G Goal | P Player | X Player fell into hole | √ Player on goal.",
        "example": "<think><observation>_P__\nG___\n*OO*\n____</observation><reasoning>I should go down then left to reach the target</reasoning></think><answer>Down{action_sep}Left</answer>"
    },
    
    "worldmodeling_symbolic": {
        "format": "<think><reasoning>...</reasoning><prediction>...</prediction></think>",
        "description": "You should first give your reasoning, then predict the next state, and finally your answer.",
        "additional_info": "The prediction should be represented as a grid using the symbols: _ Frozen | O Hole | G Goal | P Player | X Player fell into hole | √ Player on goal.",
        "example": "<think><reasoning>I can see the target is on my down left, I should go down then left</reasoning><prediction>____\n√___\n*OO*\n____</prediction></think><answer>Down{action_sep}Left</answer>"
    },
    
    "grounding_worldmodeling_symbolic": {
        "format": "<think><observation>...</observation><reasoning>...</reasoning><prediction>...</prediction></think>",
        "description": "You should first describe the observation as a grid, then your reasoning, then predict the next state, and finally your answer.",
        "additional_info": "The observation and state should be represented as grids using the symbols: _ Frozen | O Hole | G Goal | P Player | X Player fell into hole | √ Player on goal.",
        "example": "<think><observation>_P__\nG___\n*OO*\n____</observation><reasoning>I should go down then left to reach the target</reasoning><prediction>____\n√___\n*OO*\n____</prediction></think><answer>Down{action_sep}Left</answer>"
    },
    "grounding_structured": {
        "format": "<think><observation>...</observation><reasoning>...</reasoning></think><answer>...</answer>",
        "description": "You should first describe the observation as a grid, then your reasoning, and finally your answer.",
        "additional_info": "The observation should be in the format of {{'player':(row,column),'target':(row,column)}}",
        "example": "<think><observation>{{'player':(2,3),'target':(3,2)}}</observation><reasoning>I should go down then left to reach the target</reasoning></think><answer>Down{action_sep}Left</answer>"
    },
    "worldmodeling_structured": {
        "format": "<think><reasoning>...</reasoning><prediction>...</prediction></think>",
        "description": "You should first give your reasoning, then predict the next state, and finally your answer.",
        "additional_info": "The prediction should be in the format of {{'player':(row,column),'target':(row,column)}}",
        "example": "<think><reasoning>I can see the target is on my down left, I should go down then left</reasoning><prediction>{{'player':(3,2),'target':(3,2)}}</prediction></think><answer>Down{action_sep}Left</answer>"
    },
    "grounding_worldmodeling_structured": {
        "format": "<think><observation>...</observation><reasoning>...</reasoning><prediction>...</prediction></think>",
        "description": "You should first describe the observation as a grid, then your reasoning, then predict the next state, and finally your answer.",
        "additional_info": "The observation and prediction should be in the format of {{'player':(row,column),'target':(row,column)}}",
        "example": "<think><observation>{{'player':(2,3),'target':(3,2)}}</observation><reasoning>I should go down then left to reach the target</reasoning><prediction>{{'player':(3,2),'target':(3,2)}}</prediction></think><answer>Down{action_sep}Left</answer>"
    },
}

def format_prompt_generator(format_type):
    """
    Generates a prompt function for the specified format type.
    
    Args:
        format_type (str): The format type to generate a prompt function for
        
    Returns:
        function: A function that generates a prompt for the specified format
    """
    def prompt_function(**kwargs):
        """
        Generate a prompt for the specified format.
        
        Args:
            max_actions_per_step (int): Maximum number of actions allowed per step
            action_sep (str): Separator between actions
            add_example (bool): Whether to add an example
            
        Returns:
            str: The formatted prompt
        """
        max_actions_per_step = kwargs.get("max_actions_per_step", 1)
        action_sep = kwargs.get("action_sep", "|")
        add_example = kwargs.get("add_example", False)
        config = FORMAT_CONFIGS[format_type]
        
        # Build the base prompt text
        base_prompt = f"""You can take up to {max_actions_per_step} action(s) at a time, separated by {action_sep}.
{config["description"]}"""
        
        # Add additional information if available
        if "additional_info" in config:
            base_prompt += f"\n{config['additional_info']}"
        
        # Add response format instruction
        base_prompt += f"""
Your response should be in the format of:
{config["format"]}"""
        
        # Add example if requested
        if add_example:
            example = config["example"].format(action_sep=action_sep)
            return base_prompt + '\n' + f"e.g. {example}"
        
        return base_prompt
    
    return prompt_function

# Generate the format prompt dictionary using the generator
format_prompt = {format_type: format_prompt_generator(format_type) 
                for format_type in FORMAT_CONFIGS}

visual_reasoning_reward_prompt="""You are a text parser assistant. Your task is to extract relative spatial relationships between objects and the 'player' from a given text description (either an observation or a prediction) and output this information in a structured JSON format.

**Input:** You will receive a block of text that describes a state or a predicted state.
**Output:** You must output a JSON array. Each object in the array describes the relationship of one specific non-player object to the player.

**Objects to Look For:**
- target (If the input text has the mention of "gift box", that could referring the target, if the input has the mention of player and hole and another object, that also could be referring the target)
- **hole** (Treat any mention of 'hole', 'hole0', 'hole1', etc., as referring to the general 'hole' type.)

**Required JSON Output Format:**
A JSON array [...] where each element is an object {{...}} with the following keys:
- "object_id": string. The identifier of the object. Must be "target", or "hole". **For any mention of any hole (hole0, hole1, 'a hole', the hole'), this MUST be "hole".**
- "vertical_relation": string or null. The vertical position relative to the player. Must be one of: "above", "below", "same", or null if not mentioned or unclear.
- "horizontal_relation": string or null. The horizontal position relative to the player. Must be one of: "left", "right", "same", or null if not mentioned or unclear.

**Instructions:**
1.  Read the input text carefully.
2.  Identify descriptions of relative position for any of the "Objects to Look For" with respect to the "player".
3.  **Crucially, for any mention of any hole (e.g., "hole0", "hole1", "the hole", "a hole"), extract its relationship but set its "object_id" to "hole" in the output JSON.** If multiple holes have distinct relationships, create a separate JSON object entry for each distinct relationship, all with "object_id": "hole".
4.  Extract the vertical and horizontal components of the relationship. **Map variations in phrasing (like "on the left side", "to the right of the player", "same position as player", "same spot as target") to the specific allowed terms ("above", "below", "same", "left", "right").**
5.  **Handling Absolute Positions (Edge Case):** If the text describes an object's position using *absolute* terms like "top-left corner", "bottom-right corner", and the player's relative position is implied (e.g., player is known to be at the opposite corner), attempt to infer the relative position *relative to the player* using the allowed terms ("above", "below", "left", "right", "same"). For example, if the text says "target is at the top-left corner" and the player is implied to be at the bottom-right, infer "target is above and left of player". *However, prioritize extraction from text that uses direct relative terminology.*
6.  **Handling Partial or Unclear Information:** If a specific object (target, or a hole) is mentioned but *only* its vertical or horizontal relationship is described (or only implicitly), set the other relationship to null. If a relationship is too vague ("near") or doesn't use terms that can be mapped to the allowed set, set both vertical_relation and horizontal_relation to null for that object, or consider omitting the object's entry if the description is entirely unmappable.
7.  **Ignoring Irrelevant Information:** Ignore any part of the text that does not describe the relative position of an object from the "Objects to Look For" list with respect to the player, especially descriptions of actions, reasons, or generic objects without IDs.
8.  If an object from the "Objects to Look For" list is *not* mentioned in the input text with a relevant description, do NOT include it in the output JSON array.
9.  If the input text contains no extractable relationships for the listed objects using the allowed terms and inference rules, output an empty JSON array [].
10. Your output must be *only* the JSON string, and it must be valid JSON.

**Example 1 (Handling Absolute Corners):**
Input Text:
The player is at the bottom-right corner, the target is at the top-left corner, there are two ice blocks on the left and top sides of the player.

Objects to Look For: target, hole

Expected JSON Output:
```json
[
  {{
    "object_id": "target",
    "vertical_relation": "above", // Inferred from target at top-left and player at bottom-right
    "horizontal_relation": "left" // Inferred from target at top-left and player at bottom-right
  }} 
]
```
*(Note: "ice blocks" are ignored if they are not in the "Objects to Look For" list with specific IDs or if their description ("on the left and top sides") is considered too vague or not using the exact required terms for parsing by this small model.  The example assumes a strict interpretation focusing on target/hole and explicit relative terms or simple corner inferences.)*

**Example 2 (Handling Multiple Holes & Partial Info):**
Input Text:
Hole0 is above. The target is right of the player. Hole1 is below and same column. Avoid the ice.

Objects to Look For: target, hole

Expected JSON Output:
```json
[
  {{ 
    "object_id": "hole",
    "vertical_relation": "above",
    "horizontal_relation": null // Horizontal relation not mentioned for Hole0
  }}, {{ 
    "object_id": "target",
    "vertical_relation": null, // Vertical relation is not mentioned for Target
    "horizontal_relation": "right"
  }}, {{
    "object_id": "hole",
    "vertical_relation": "below",
    "horizontal_relation": "same" // same column maps to same horizontal
  }} 
]
```
*(Note: Hole0 and Hole1 are both mapped to object_id "hole". Their distinct relationships are preserved. Text like "Avoid the ice" is ignored.)*

**Example 3 (Mapping "Same Place" and Ignoring Irrelevant):**
Input Text:
The player will reach the target. There is a pit nearby. Hole2 is same spot as player.

Objects to Look For: target, hole

Expected JSON Output:
```json
[
  {{ 
    "object_id": "target",
    "vertical_relation": "same", // "reach the target" or similar implies "same place"
    "horizontal_relation": "same"
  }}, {{ 
    "object_id": "hole",
    "vertical_relation": "same", // "same spot" maps to "same place"
    "horizontal_relation": "same"
  }} 
]
```
*(Note: "a pit nearby" is ignored if not in the "Objects to Look For" or described with required terms. "same spot" maps to "same" for both relations.)*

**Example 4 (No Relevant Info):**
Input Text:
I should move down and left now.

Objects to Look For: target, hole

Expected JSON Output:
```json
[]
```

**Example 5 (No Relevant Info):**
Input Text:
The player should move to the gift box to avoid the hole and reach the target.

Objects to Look For: target, hole

Expected JSON Output:
```json
[]
```
---
Input Text to Parse:
{prediction}

Objects to Look For: target, hole

JSON Output:
"""
if __name__ == "__main__":
    # Example usage
    max_actions_per_step = 2
    action_sep = "|"
    
    for key, func in format_prompt.items():
        print(f"{key} format prompt:")
        print(func(max_actions_per_step=max_actions_per_step, action_sep=action_sep, add_example=True))
        print("\n" + "="*50 + "\n")