"""
JSON extraction and validation module for cognitive maps.

This module provides functions to:
1. Extract JSON from text responses
2. Validate JSON structure for cognitive maps
3. Determine the format type (complex vs. simple)
"""

import json
import re
from typing import Dict, List, Tuple, Optional, Any, Union


def extract_response_from_text(text: str) -> Optional[str]:
    """
    Extract the response from the text.
    """
    # response is like: "<think>...</think><answer>...</answer>"
    # "..." can be any text, even empty
    # we want to see if we can parse such format from the text

    pattern = r'<think>(.*?)</think><answer>(.*?)</answer>'
    matches = re.findall(pattern, text)
    if matches:
        return matches[0]
    return None

def extract_json_from_text(text: str) -> Optional[Dict]:
    """
    Extract JSON cognitive map from text response.
    Returns the JSON object if found, otherwise None.
    
    Args:
        text: Text containing a JSON object
        
    Returns:
        Extracted JSON object or None
    """
    if not text:
        return None
        
    # Look for JSON pattern with { } brackets
    pattern = r'\{[\s\S]*\}'
    matches = re.findall(pattern, text)
    
    if not matches:
        return None
    
    # If multiple matches, select the longest one
    matches.sort(key=len, reverse=True)
    json_str = matches[0]
    
    # Try direct JSON parsing first
    try:
        return json.loads(json_str)
    except json.JSONDecodeError:
        # Try to clean up and parse again
        return clean_and_parse_json(json_str)

def clean_and_parse_json(json_str: str) -> Optional[Dict]:
    """
    Attempt to clean and parse a malformed JSON string.
    
    Args:
        json_str: A potentially malformed JSON string
        
    Returns:
        Parsed JSON object or None
    """
    try:
        # Remove comments
        clean_json = re.sub(r'//.*', '', json_str)
        # Remove newlines, tabs
        clean_json = re.sub(r'[\n\r\t]', ' ', clean_json)
        
        # Fix unquoted keys
        clean_json = re.sub(r'(\s*?)(\w+)(\s*?):', r'\1"\2"\3:', clean_json)
        # Fix trailing commas
        clean_json = re.sub(r',\s*}', '}', clean_json)
        clean_json = re.sub(r',\s*]', ']', clean_json)
        
        return json.loads(clean_json)
    except:
        # As a final attempt, try to extract in "key-value" format
        try:
            # Extract pairs like "object_name": { "position": [...], "facing": ... }
            pairs_pattern = r'"([^"]+)":\s*{([^{}]*(?:{[^{}]*}[^{}]*)*)}'
            pairs = re.findall(pairs_pattern, json_str)
            
            if pairs:
                result = {}
                for key, value in pairs:
                    try:
                        # Parse the value part
                        value_str = '{' + value + '}'
                        # Fix unquoted keys
                        value_str = re.sub(r'(\s*?)(\w+)(\s*?):', r'\1"\2"\3:', value_str)
                        # Fix trailing commas
                        value_str = re.sub(r',\s*}', '}', value_str)
                        value_str = re.sub(r',\s*]', ']', value_str)
                        
                        value_obj = json.loads(value_str)
                        result[key] = value_obj
                    except:
                        continue
                
                if result:
                    return result
        except:
            pass
        
        return None

def is_complex_format(cogmap: Dict) -> bool:
    """
    Determine if the cognitive map uses complex format (with objects/views arrays)
    or simple key-value format.
    
    Args:
        cogmap: The cognitive map JSON
        
    Returns:
        True if complex format, False if simple format
    """
    # Check if cogmap is a dictionary
    if not isinstance(cogmap, dict):
        return False
        
    return "objects" in cogmap and isinstance(cogmap.get("objects"), list)

def is_valid_position(position: Any) -> bool:
    """
    Check if a position value is valid (a list of 2 numeric values).
    
    Args:
        position: The position value to check
        
    Returns:
        True if valid, False otherwise
    """
    if not isinstance(position, list):
        return False
    
    if len(position) < 2:
        return False
    
    try:
        # Check if first two elements are numeric
        float(position[0])
        float(position[1])
        return True
    except (ValueError, TypeError):
        return False

def truncate_position_list_into_one(positions: List[Dict] | Dict) -> Dict:
    """
    Truncate a list of positions into a single position.
    """
    if isinstance(positions, list):
        return positions[0]
    return positions

def trucate_object_position(raw_cogmap: Dict) -> Dict:
    """
    Truncate a list of positions into a single position.
    """
    # Check if raw_cogmap is a dictionary
    if not isinstance(raw_cogmap, dict):
        return {}
        
    return {k: truncate_position_list_into_one(v) for k, v in raw_cogmap.items()}

def is_valid_facing(facing: Any) -> bool:
    """
    Check if a facing value is valid (one of: up, down, left, right, inner, outer).
    
    Args:
        facing: The facing value to check
        
    Returns:
        True if valid, False otherwise
    """
    if facing is None:
        return True  # Facing is optional
        
    if isinstance(facing, list):
        if not facing:
            return True
        facing = facing[0]
    
    if not isinstance(facing, str):
        return False
    
    # Normalize
    facing = facing.lower().strip()
    
    valid_facings = {
        "up", "down", "left", "right", "inner", "outer", 
        "top", "bottom", "north", "south", "east", "west",
        "front", "back", "into", "out", "inside", "outside",
        "forward", "backward"
    }
    
    return facing in valid_facings

def validate_cogmap_format(cogmap: Dict) -> Tuple[bool, List[str]]:
    """
    Validate if a cognitive map has the correct format.
    
    Args:
        cogmap: The cognitive map to validate
        
    Returns:
        Tuple of (is_valid, error_messages)
    """
    if not isinstance(cogmap, dict):
        return False, ["Cognitive map is not a dictionary"]
    
    errors = []
    
    # Check format type
    is_complex = is_complex_format(cogmap)
    
    if is_complex:
        # Validate complex format (objects/views)
        if not isinstance(cogmap.get("objects", []), list):
            errors.append("'objects' field is not a list")
        
        # Check each object
        for i, obj in enumerate(cogmap.get("objects", [])):
            if not isinstance(obj, dict):
                errors.append(f"Object {i} is not a dictionary")
                continue
                
            if "name" not in obj:
                errors.append(f"Object {i} is missing 'name' field")
            
            if "position" in obj and not is_valid_position(obj["position"]):
                errors.append(f"Object {i} has invalid 'position' format")
            
            if "facing" in obj and not is_valid_facing(obj["facing"]):
                errors.append(f"Object {i} has invalid 'facing' value")
        
        # Check views (if present)
        for i, view in enumerate(cogmap.get("views", [])):
            if not isinstance(view, dict):
                errors.append(f"View {i} is not a dictionary")
                continue
                
            if "name" not in view:
                errors.append(f"View {i} is missing 'name' field")
            
            if "position" in view and not is_valid_position(view["position"]):
                errors.append(f"View {i} has invalid 'position' format")
            
            if "facing" in view and not is_valid_facing(view["facing"]):
                errors.append(f"View {i} has invalid 'facing' value")
    else:
        # Validate simple format (key-value)
        for obj_name, obj_data in cogmap.items():
            if not isinstance(obj_data, dict):
                errors.append(f"Object '{obj_name}' is not a dictionary")
                continue
            
            if "position" in obj_data and not is_valid_position(obj_data["position"]):
                errors.append(f"Object '{obj_name}' has invalid 'position' format")
            
            if "facing" in obj_data and not is_valid_facing(obj_data["facing"]):
                errors.append(f"Object '{obj_name}' has invalid 'facing' value")
            
            # Check for unknown fields
            unknown_fields = [f for f in obj_data.keys() if f not in ["position", "facing"]]
            if unknown_fields:
                errors.append(f"Object '{obj_name}' has unknown fields: {', '.join(unknown_fields)}")
    
    return len(errors) == 0, errors

def extract_answer(text: str) -> Optional[str]:
    """
    Extract the answer from model response text using regular expressions.
    Returns the last occurrence of the letter of the answer (A, B, C, D, or E)
    based on pattern priority - tries higher priority patterns first.
    
    Args:
        text: The model response text
        
    Returns:
        The last answer letter found by the highest priority matching pattern,
        or None if not found
    """
    if not text:
        return None
    
    # Patterns in order of priority (higher priority first)
    patterns = [
        r'(?:Answer: )?([A-E])\. [A-Za-z0-9 \-\(\)\'",]+(?=(?:\n|$|\.|"))',  # Full answer with description
        r'(?:Answer: )?([A-E])\. [A-Za-z0-9 \-\(\)\'"]+',  # Answer with partial description
        r'(?:^|\n)(?:Answer: )?([A-E])(?:\.|$|\s)',  # Answer at line beginning
        r'(?:<answer>)([A-E])(?:[\.\s]*)(?:</answer>)',  # XML-style answer tag
    ]
    
    # Try each pattern in order of priority
    for pattern in patterns:
        matches = list(re.finditer(pattern, text))
        if matches:
            # Return the last match found by this pattern
            return matches[-1].group(1)
    
    # If none of the priority patterns match, try line-by-line parsing
    # First, try the more specific pattern on each line
    lines = text.split('\n')
    line_matches = []
    
    for i, line in enumerate(lines):
        # Look for full answer pattern in each line
        match = re.search(r'([A-E])\. [A-Za-z0-9 \-\(\)\'",]+', line)
        if match:
            line_matches.append((i, match.group(1)))
    
    if line_matches:
        # Return the answer from the last line that matched
        return line_matches[-1][1]
    
    # Finally, try the most general pattern on each line
    for i in reversed(range(len(lines))):  # Start from bottom
        line = lines[i]
        match = re.search(r'\b([A-E])\b', line)
        if match:
            return match.group(1)
    
    return None  # No answer found

def get_setting_from_id(item_id: str) -> str:
    """
    Determine the setting category based on the item ID.
    Focuses on four categories: around, rotation, translation, among.
    
    Args:
        item_id: The item identifier string
        
    Returns:
        Setting category ('around', 'rotation', 'translation', 'among', or 'other')
    """
    if 'around' in item_id.lower():
        return 'around'
    elif 'rotation' in item_id.lower():
        return 'rotation'
    elif 'translation' in item_id.lower():
        return 'translation'
    elif 'among' in item_id.lower():
        return 'among'
    else:
        return 'other'


def format_checking_pipeline(answer_str: str) -> Tuple[bool, int]:
    """
    Check the format of the answer string.
    """
    # 0. extract the response from the text
    response = extract_response_from_text(answer_str)
    if response is None:
        print(f"No response found in the answer string")
        return False, 0 # "No response found in the answer string"
    
    # 1. extract the json from the text
    json_obj = extract_json_from_text(answer_str)
    if json_obj is None:
        print(f"No JSON object found in the answer string")
        return False, 1 # "No JSON object found in the answer string"
    
    # 2. validate the format of the json
    is_valid, errors = validate_cogmap_format(json_obj)
    if not is_valid:
        print(f"Invalid CogMap format: {errors}")
        return False, 2 # "Invalid CogMap format: {errors}"
    
    # 3. extract the answer from the text
    answer = extract_answer(answer_str)
    if answer is None:
        print(f"No answer found in the answer string")
        return False, 3 # "No answer found in the answer string"
    
    return True, -1


# --- Example usage ---
def test_extract_answer():
    answer_str = "<think>I need to determine how I moved from the viewpoint in image 1 to the viewpoint in image 2. In image 1, I can see blue ball in front of the sliding door. In image 2, I can see blue ball in front of the blue ball. I notice that blue ball is visible in both images, but from different angles. I analyze how the viewpoint changed from image 1 to image 2 by analyzing how the structural features of objects on the platform and relative positions of these objects transform between the two views. This involves tracking how key features shift positions, observing which elements become more prominent or less visible, and comparing if these observed changes align with the expected spatial transformations that would occur when viewing the object from its left or right side. Image 2 seems to show the blue ball's left side compared to the first one. This suggests that, to transition from the viewpoint in the first image to the viewpoint in the second image, I need to move forward and to the left. Therefore, the answer is A. Forward-left</think><answer>A. Forward-left</answer>"
    print(f"Testing the extract answer pipeline with the answer string. ")
    print(f"Result: {extract_answer(answer_str)}")

def format_checking_usage_example_pipeline():
    # you may want to check the format of the output is correct or not
    # this can be break down into three steps:
    # 1. extract the json from the text
    # 2. validate the format of the json
    # 3. extract the answer from the text

    example_answer_str_correct_1 = "{\n  \"objects\": [\n    {\"name\": \"computer screen\", \"position\": [7, 4], \"facing\": \"up\"},\n    {\"name\": \"green stool\", \"position\": [3, 2]}\n  ],\n  \"views\": [\n    {\"name\": \"Image 1\", \"position\": [0, 0], \"facing\": \"front\"},\n    {\"name\": \"Image 2\", \"position\": [0, 0], \"facing\": \"left\"},\n    {\"name\": \"Image 3\", \"position\": [0, 0], \"facing\": \"back\"}\n  ]\n}\n\n<Reasoning>\n1. In Image 2 (viewpoint from the left), the scene shows a computer screen located towards the right of the room, with the green stool to the left and in front of the observer.\n2. The viewpoint in Image 2 faces left, so the observer’s direction looks towards the left side of the room.\n3. The computer screen is positioned roughly to the right of this viewpoint, indicating that turning right from this position would be facing towards or near the direction of the computer.\n4. Moving forward after turning right (which is now visually aligned with the computer's position), the observer would move closer to the computer screen.\n5. Therefore, given the spatial layout, turning right from image 2's viewpoint and then moving forward will bring the observer closer to the computer.\n\n<Answer>\nTherefore, my answer is A. Above"

    example_answer_str_correct_2 = "Based on my observations, I will build the cognitive map as follows:\n <CogMap>\n```json\n{\"toy train\": {\"position\": [5, 5]}, \"window\": {\"position\": [5, 8]}, \"black table\": {\"position\": [2, 5]}, \"wall\": {\"position\": [5, 2]}, \"printed glass door\": {\"position\": [8, 5]}}\n```\nAfter generating the cognitive map, I will provide my answer to the question:\n<Answer>\nB. Yes"

    example_answer_str_incorrect_json = "Based on my observations, I will build the cognitive map as follows:\n <CogMap>\n```json\ntoy train: {\"position\": [5, 5]}, window: {\"position\": [5, 8]}, black table: {\"position\": [2, 5]}, wall: {\"position\": [5, 2]}, printed glass door: {\"position\": [8, 5]}\n```\nAfter generating the cognitive map, I will provide my answer to the question:\n<Answer>\nB. Yes"

    example_answer_str_incorrect_answer = "Based on my observations, I will build the cognitive map as follows:\n <CogMap>\n```json\n{\"toy train\": {\"position\": [5, 5]}, \"window\": {\"position\": [5, 8]}, \"black table\": {\"position\": [2, 5]}, \"wall\": {\"position\": [5, 2]}, \"printed glass door\": {\"position\": [8, 5]}}\n```\nAfter generating the cognitive map, I will provide my answer to the question:\n<answer>\nYes"

    print(f"Testing the format checking pipeline with the correct answer 1. ")
    print(f"Result: {format_checking_pipeline(example_answer_str_correct_1)}")
    
    print(f"Testing the format checking pipeline with the correct answer 2. ")
    print(f"Result: {format_checking_pipeline(example_answer_str_correct_2)}")
    
    print(f"Testing the format checking pipeline with the incorrect json. ")
    print(f"Result: {format_checking_pipeline(example_answer_str_incorrect_json)}")

    print(f"Testing the format checking pipeline with the incorrect answer. ")
    print(f"Result: {format_checking_pipeline(example_answer_str_incorrect_answer)}")


if __name__ == "__main__":
    format_checking_usage_example_pipeline()
    test_extract_answer()