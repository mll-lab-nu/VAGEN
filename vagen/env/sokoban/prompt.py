def system_prompt(**kwargs):
    return """You are a Sokoban solver.
Sokoban Quick Guide
Goal: Push all boxes onto targets.
Symbols (If image is provided there are no symbols):
# Wall | _ Floor | O Target | X Box | P You | √ Box on Target | S You on Target
Rules:
1. Push boxes (can't pull).
2. Avoid walls.
Actions you can take: Left, Down, Right, Up."""

def init_observation_template(**kwargs):
    observation = kwargs.get("img_str", "The player is near a box")
    return f"""[Initial Observation]:
{observation}
Decide your next action(s)."""

def action_template(**kwargs):
    valid_action = kwargs.get("valid_action", "Down")
    observation = kwargs.get("img_str", "The player pushed the box closer to the target")
    return f"""After your answer, the extracted valid action is {valid_action}.
After that, the observation is:
{observation}
Decide your next action(s)."""

# Format configurations defining the structure of each format
FORMAT_CONFIGS = {
    "free_think": {
        "format": "<think>...</think><answer>...</answer>",
        "description": "You should first give your reasoning, and then your answer.",
        "example": "<think><reasoning>The box is one step below me, and the target is two steps below me, I need to go down then push the box down to the target.</reasoning></think><answer>Down{action_sep}Down</answer>"
    },
    
    "no_think": {
        "format": "<answer>...</answer>",
        "description": "You should provide only your answer.",
        "example": "<answer>Down{action_sep}Down</answer>"
    },
    
    "grounding": {
        "format": "<think><observation>...</observation><reasoning>...</reasoning></think><answer>...</answer>",
        "description": "You should first give the description of your observation, then your reasoning, and finally your answer.",
        "example": "<think><observation>The box is below the player and the target is below the box</observation><reasoning>I need to go down then push the box down to the target</reasoning></think><answer>Down{action_sep}Down</answer>",
        "additional_info": "Inside the `<observation>` tags, describe the position of the `target` (red) and each `box`(yellow) relative to the player. For each object, you must specify **both** its vertical (`above`, `below`, `same`) and horizontal (`left`, `right`, `same`) direction.",
    },
    
    "worldmodeling": {
        "format": "<think><reasoning>...</reasoning><prediction>...</prediction></think><answer>...</answer>",
        "description": "You should first give your reasoning, then predict the next state, and finally your answer.",
        "example": "<think><reasoning>I need to go right then push the box down to the target.</reasoning><prediction>The player will be above the box, the target and box will be at the same place.</prediction></think><answer>Right{action_sep}Down</answer>",
        "additional_info": "Inside the `<prediction>` tags, describe the position of the `target` (red) and each `box`(yellow) relative to the player. For each object, you must specify **both** its vertical (`above`, `below`, `same`) and horizontal (`left`, `right`, `same`) direction.",
    },
    
    "grounding_worldmodeling": {
        "format": "<think><observation>...</observation><reasoning>...</reasoning><prediction>...</prediction></think><answer>...</answer>",
        "description": "You should first give the description of your observation, then your reasoning, then predict the next state, and finally your answer.",
        "additional_info": "For the content you provide within the `<observation>` and `<prediction>` tags, you must strictly describe the relative position of the `target` and any visible `box` objects **relative to the player**. Your description/prediction must include **both** a vertical and a horizontal directional relationship for each object. Use ONLY the terms `above`, `below`, `left`, `right`, or `same` for describing these relationships.",
        "example": "<think><observation>The box is below and to right of the player and the target is below and to the right of the player</observation><reasoning>I need to go right then go down to push the box down to the target</reasoning><prediction>The player will be above the box and target, the target and box will be at the same row.</prediction></think><answer>Right{action_sep}Down</answer>"
    },
    
    "grounding_symbolic": {
        "format": "<think><observation>...</observation><reasoning>...</reasoning></think><answer>...</answer>",
        "description": "You should first give the description of your observation as a grid, then your reasoning, and finally your answer.",
        "additional_info": "The state should be represented as a grid using the symbols: # Wall | _ Floor | O Target | X Box | P You | √ Box on Target | S You on Target.",
        "example": "<think><observation>####\n#_P#\n#__#\n#_X#\n#_O#</observation><reasoning>I need to go down then push the box down to reach the target</reasoning></think><answer>Down{action_sep}Down</answer>"
    },
    
    "worldmodeling_symbolic": {
        "format": "<think><reasoning>...</reasoning><prediction>...</prediction></think><answer>...</answer>",
        "description": "You should first give your reasoning, then predict the next state as a grid, and finally your answer.",
        "additional_info": "The state should be represented as a grid using the symbols: # Wall | _ Floor | O Target | X Box | P You | √ Box on Target | S You on Target.",
        "example": "<think><reasoning>I need to go down then push the box down to reach the target</reasoning><prediction>####\n#__#\n#__#\n#_P#\n#_√#</prediction></think><answer>Down{action_sep}Down</answer>"
    },
    
    "grounding_worldmodeling_symbolic": {
        "format": "<think><observation>...</observation><reasoning>...</reasoning><prediction>...</prediction></think><answer>...</answer>",
        "description": "You should first give the description of your observation as a grid, then your reasoning, then predict the next state as a grid, and finally your answer.",
        "additional_info": "The observation and state should be represented as grids using the symbols: # Wall | _ Floor | O Target | X Box | P You | √ Box on Target | S You on Target.",
        "example": "<think><observation>####\n#_P#\n#__#\n#_X#\n#_O#</observation><reasoning>I need to go down then push the box down to reach the target</reasoning><prediction>####\n#__#\n#__#\n#_P#\n#_√#</prediction></think><answer>Down{action_sep}Down</answer>"
    },
    
    "grounding_structured": {
        "format": "<think><observation>...</observation><reasoning>...</reasoning></think><answer>...</answer>",
        "description": "You should first give the description of your observation, then your reasoning, and finally your answer.",
        "additional_info": "The observation should be in the format of {{\"player\":(row,column),\"box\":(row,column),\"target\":(row,column)}}",
        "example": "<think><observation>{{\"player\":(1,2),\"box\":(3,2),\"target\":(4,2)}}</observation><reasoning>I need to go down then push the box down to the target</reasoning></think><answer>Down{action_sep}Down</answer>"
    },
    
    "worldmodeling_structured": {
        "format": "<think><reasoning>...</reasoning><prediction>...</prediction></think><answer>...</answer>",
        "description": "You should first give your reasoning, then predict the next state, and finally your answer.",
        "additional_info": "The prediction should be in the format of {{\"player\":(row,column),\"box\":(row,column),\"target\":(row,column)}}",
        "example": "<think><reasoning>I need to go down then push the box down to the target</reasoning><prediction>{{\"player\":(3,2),\"box\":(4,2),\"target\":(4,2)}}</prediction></think><answer>Down{action_sep}Down</answer>"
    },
    
    "grounding_worldmodeling_structured": {
        "format": "<think><observation>...</observation><reasoning>...</reasoning><prediction>...</prediction></think><answer>...</answer>",
        "description": "You should first give the description of your observation, then your reasoning, then predict the next state, and finally your answer.",
        "additional_info": "The observation and prediction should be in the format of [{{\"object_id\":target,\"vertical_relation\":\"xxx\",\"horizontal_relation\":\"xxx\"}},{{\"object_id\":box,\"vertical_relation\":\"xxx\",\"horizontal_relation\":\"xxx\"}}]",
        "example": "<think><observation>[{{\"object_id\":\"target\",\"vertical_relation\":\"below\",\"horizontal_relation\":\"same\"}},{{\"object_id\":\"box\",\"vertical_relation\":\"below\",\"horizontal_relation\":\"same\"}}]</observation><reasoning>I need to go down then push the box down to the target</reasoning></think><prediction>[{{\"object_id\":\"target\",\"vertical_relation\":\"below\",\"horizontal_relation\":\"same\"}},{{\"object_id\":\"box\",\"vertical_relation\":\"below\",\"horizontal_relation\":\"same\"}}]</prediction></think><answer>Down{action_sep}Down</answer>"
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
        add_example = kwargs.get("add_example", True)
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


visual_reasoning_reward_prompt="""You are a text parser assistant. Your task is to extract relative spatial relationships between objects and the 'player' from a given text description (either an observation or a prediction) in a Sokoban environment, and output this information in a structured JSON format.

**Input:** You will receive a block of text that describes a state or a predicted state.
**Output:** You must output a JSON array. Each object in the array describes the relationship of one specific non-player object to the player.

**Objects to Look For:**
- target
- **box** (Treat any mention of 'box', 'box0', 'box1', etc., as referring to the general 'box' type.)

**Required JSON Output Format:**
A JSON array [...] where each element is an object {{...}} with the following keys:
- "object_id": string. The identifier of the object. Must be "target", or "box". **For any mention of any box (box0, box1, 'a box', the box'), this MUST be "box".**
- "vertical_relation": string or null. The vertical position relative to the player. Must be one of: "above", "below", "same", or null if not mentioned or unclear.
- "horizontal_relation": string or null. The horizontal position relative to the player. Must be one of: "left", "right", "same", or null if not mentioned or unclear.

**Instructions:**
1.  Read the input text carefully.
2.  Identify descriptions of relative position for any of the "Objects to Look For" with respect to the "player".
3.  **Crucially, for any mention of any box (e.g., "box0", "box1", "the box", "a box"), extract its relationship but set its "object_id" to "box" in the output JSON.** If multiple boxes have distinct relationships, create a separate JSON object entry for each distinct relationship, all with "object_id": "box".
4.  Extract the vertical and horizontal components of the relationship. **Map variations in phrasing (like "on the left side", "to the right of the player", "same position as player", "same spot as target", "same spot as a box") to the specific allowed terms ("above", "below", "same", "left", "right").**
5.  **Handling Absolute Positions (Edge Case):** If the text describes an object's position using *absolute* terms like "top-left corner", "bottom-right corner", and the player's relative position is implied (e.g., player is known to be at the opposite corner), attempt to infer the relative position *relative to the player* using the allowed terms ("above", "below", "left", "right", "same"). For example, if the text says "target is at the top-left corner" and the player is implied to be at the bottom-right, infer "target is above and left of player". *However, prioritize extraction from text that uses direct relative terminology.*
6.  **Handling Partial or Unclear Information:** If a specific object (target, box) is mentioned but *only* its vertical or horizontal relationship is described (or only implicitly), set the other relationship to null. If a relationship is too vague ("near") or doesn't use terms that can be mapped to the allowed set, set both vertical_relation and horizontal_relation to null for that object, or consider omitting the object's entry if the description is entirely unmappable.
7.  **Ignoring Irrelevant Information:** Ignore any part of the text that does not describe the relative position of an object from the "Objects to Look For" list with respect to the player, especially descriptions of actions, reasons, or generic objects without IDs.
8.  If an object from the "Objects to Look For" list is *not* mentioned in the input text with a relevant description, do NOT include it in the output JSON array.
9.  If the input text contains no extractable relationships for the listed objects using the allowed terms and inference rules, output an empty JSON array [].
10. Your output must be *only* the JSON string, and it must be valid JSON.

**Example 1 (Target and Box):**
Input Text:
The target is above and left of the player. A box is right of the player.

Objects to Look For: target, box

Expected JSON Output:
```json
[
  {{
    "object_id": "target",
    "vertical_relation": "above",
    "horizontal_relation": "left"
  }},
  {{
    "object_id": "box",
    "vertical_relation": null, // Vertical relation not mentioned
    "horizontal_relation": "right"
  }}
]
```
*(Note: The box's vertical relation was not specified, so it's null.)*

**Example 2 (Multiple Boxes):**
Input Text:
There is a box above me and another box left of me. The target is far above and left.

Objects to Look For: target, box

Expected JSON Output:
```json
[
  {{
    "object_id": "box",
    "vertical_relation": "above",
    "horizontal_relation": null // Horizontal relation not mentioned for the first box
  }},
  {{
    "object_id": "box",
    "vertical_relation": null, // Vertical relation not mentioned for the second box
    "horizontal_relation": "left"
  }},
  {{
    "object_id": "target",
    "vertical_relation": "above",
    "horizontal_relation": "left" // "far above and left" maps to "above" and "left"
  }}
]
```
*(Note: Two distinct box relationships are listed, both with object_id "box". Vertical/horizontal relations are null if not explicitly mentioned.)*

**Example 3 (Box on Target / Same Place):**
Input Text:
The box is on the target, and they are both at the same spot as the player.

Objects to Look For: target, box

Expected JSON Output:
```json
[
  {{
    "object_id": "box",
    "vertical_relation": "same",
    "horizontal_relation": "same"
  }},
  {{
    "object_id": "target",
    "vertical_relation": "same",
    "horizontal_relation": "same"
  }}
]
```
*(Note: "on the target" combined with "same spot as player" implies the box is also at the same spot as the player.)*

**Example 4 (Absolute Corners - Sokoban Context):**
Input Text:
The player is at the bottom-right corner. The target is at the top-right. A box is at the bottom-left.

Objects to Look For: target, box

Expected JSON Output:
```json
[
  {{
    "object_id": "target",
    "vertical_relation": "above", // Target at top-right relative to player at bottom-right
    "horizontal_relation": "same"
  }},
  {{
    "object_id": "box",
    "vertical_relation": "same", // Box at bottom-left relative to player at bottom-right
    "horizontal_relation": "left"
  }}
]
```
*(Note: Inferring relative positions from absolute corner descriptions based on player's known corner.)*

**Example 5 (No Relevant Info):**
Input Text:
I need to find the correct path.

Objects to Look For: target, box

Expected JSON Output:
```json
[]
```

---
Input Text to Parse:
{prediction}

Objects to Look For: target, box # Ensure this list matches the main list above if needed

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
    print(visual_reasoning_reward_prompt.format(prediction="123"))