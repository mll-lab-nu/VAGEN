import re
from typing import Dict, List
import json

def parse_freethink(response: str, special_token_list=None, action_sep=',', max_actions=3) -> Dict:
    """
    Parse response in format: <think>...</think><answer>...</answer>
    
    Returns a dict with keys:
    - llm_raw_response: the original response
    - llm_response: the response with <think> and <answer> tags
    - think_content: the content inside <think> tag
    - action_content: the content inside <answer> tag
    - actions: a list of actions extracted from action_content
    - format_correct: whether the response strictly follows the expected format
    """
    #Pattern to check for content strictly in the format <think>...</think><answer>...</answer>
    strict_pattern = r'^\s*<think>(.*?)</think>\s*<answer>(.*?)</answer>\s*$'
    strict_match = re.match(strict_pattern, response.strip(), re.DOTALL)
    
    
    # Pattern to extract content from think and answer tags
    extraction_pattern = r'<think>(.*?)</think>\s*<answer>(.*?)</answer>'
    match = re.search(extraction_pattern, response, re.DOTALL)
    format_correct = strict_match is not None
    
    if not strict_match:
        think_content, action_content, actions = "", "", []
    else:
        think_content, action_content = match.group(1), match.group(2)
        if special_token_list is not None:
            for special_token in special_token_list: # remove all special tokens in responses to forbid confusion in training
                action_content = action_content.replace(special_token, "").strip()
                think_content = think_content.replace(special_token, "").strip()
        actions = [action.strip() for action in action_content.split(action_sep) if action.strip()]
        if len(actions) > max_actions:
            actions = actions[:max_actions] #Only the first MAX_ACTIONS actions are kept in the rollout.
            action_content = (" " + action_sep + " ").join(actions)

    llm_response = "<think>" + think_content.strip() + "</think>" + "<answer>" + action_content.strip() + "</answer>"
    return {
        "llm_raw_response": response,
        "llm_response": llm_response,
        "think_content": think_content,
        "action_content": action_content,
        "actions": actions,
        "format_correct": format_correct
    }

def parse_no_think(response: str, special_token_list=None, action_sep=',', max_actions=3) -> Dict:
    """
    Parse response in format: <answer>...</answer>
    
    Returns a dict with keys:
    - llm_raw_response: the original response
    - llm_response: the response with <answer> tag
    - think_content: empty string (no think content in this format)
    - action_content: the content inside <answer> tag
    - actions: a list of actions extracted from action_content
    - format_correct: whether the response strictly follows the expected format
    """
    # Pattern to check for content strictly in the format <answer>...</answer>
    strict_pattern = r'^\s*<answer>(.*?)</answer>\s*$'
    strict_match = re.match(strict_pattern, response.strip(), re.DOTALL)
    format_correct = strict_match is not None
    
    # Pattern to extract content from answer tag
    extraction_pattern = r'<answer>(.*?)</answer>'
    match = re.search(extraction_pattern, response, re.DOTALL)
    #format_correct = match is not None
    
    if not strict_match:
        think_content, action_content, actions = "", "", []
    else:
        action_content = match.group(1)
        think_content = ""  # No think content in this format
        if special_token_list is not None:
            for special_token in special_token_list:
                action_content = action_content.replace(special_token, "").strip()
        actions = [action.strip() for action in action_content.split(action_sep) if action.strip()]
        if len(actions) > max_actions:
            actions = actions[:max_actions]
            action_content = (" " + action_sep + " ").join(actions)

    llm_response = "<answer>" + action_content.strip() + "</answer>"
    return {
        "llm_raw_response": response,
        "llm_response": llm_response,
        "think_content": think_content,
        "action_content": action_content,
        "actions": actions,
        "format_correct": format_correct
    }

def parse_grounding(response: str, special_token_list=None, action_sep=',', max_actions=3) -> Dict:
    """
    Parse response in format: <state>...</state><think>...</think><answer>...</answer>
    
    Returns a dict with keys:
    - llm_raw_response: the original response
    - llm_response: the response with all tags
    - state_content: the content inside <state> tag
    - think_content: the content inside <think> tag
    - action_content: the content inside <answer> tag
    - actions: a list of actions extracted from action_content
    - format_correct: whether the response strictly follows the expected format
    """
    # Pattern to check for content strictly in the expected format
    strict_pattern = r'^\s*<state>(.*?)</state>\s*<think>(.*?)</think>\s*<answer>(.*?)</answer>\s*$'
    strict_match = re.match(strict_pattern, response.strip(), re.DOTALL)
    format_correct = strict_match is not None
    
    # Pattern to extract content from tags
    extraction_pattern = r'<state>(.*?)</state>\s*<think>(.*?)</think>\s*<answer>(.*?)</answer>'
    match = re.search(extraction_pattern, response, re.DOTALL)
    #format_correct = match is not None
    
    if not strict_match:
        state_content, think_content, action_content, actions = "", "", "", []
    else:
        state_content, think_content, action_content = match.group(1), match.group(2), match.group(3)
        if special_token_list is not None:
            for special_token in special_token_list:
                state_content = state_content.replace(special_token, "").strip()
                action_content = action_content.replace(special_token, "").strip()
                think_content = think_content.replace(special_token, "").strip()
        actions = [action.strip() for action in action_content.split(action_sep) if action.strip()]
        if len(actions) > max_actions:
            actions = actions[:max_actions]
            action_content = (" " + action_sep + " ").join(actions)

    llm_response = "<state>" + state_content.strip() + "</state>" + \
                   "<think>" + think_content.strip() + "</think>" + \
                   "<answer>" + action_content.strip() + "</answer>"
    return {
        "llm_raw_response": response,
        "llm_response": llm_response,
        "state_content": state_content,
        "think_content": think_content,
        "action_content": action_content,
        "actions": actions,
        "format_correct": format_correct
    }

def parse_worldmodeling(response: str, special_token_list=None, action_sep=',', max_actions=3) -> Dict:
    """
    Parse response in format: <think>...</think><answer>...</answer><state>...</state>
    
    Returns a dict with keys:
    - llm_raw_response: the original response
    - llm_response: the response with all tags
    - think_content: the content inside <think> tag
    - action_content: the content inside <answer> tag
    - state_content: the content inside <state> tag
    - actions: a list of actions extracted from action_content
    - format_correct: whether the response strictly follows the expected format
    """
    # Pattern to check for content strictly in the expected format
    strict_pattern = r'^\s*<think>(.*?)</think>\s*<answer>(.*?)</answer>\s*<state>(.*?)</state>\s*$'
    strict_match = re.match(strict_pattern, response.strip(), re.DOTALL)
    format_correct = strict_match is not None
    
    # Pattern to extract content from tags
    extraction_pattern = r'<think>(.*?)</think>\s*<answer>(.*?)</answer>\s*<state>(.*?)</state>'
    match = re.search(extraction_pattern, response, re.DOTALL)
    #format_correct = match is not None
    
    if not strict_match:
        think_content, action_content, state_content, actions = "", "", "", []
    else:
        think_content, action_content, state_content = match.group(1), match.group(2), match.group(3)
        if special_token_list is not None:
            for special_token in special_token_list:
                action_content = action_content.replace(special_token, "").strip()
                think_content = think_content.replace(special_token, "").strip()
                state_content = state_content.replace(special_token, "").strip()
        actions = [action.strip() for action in action_content.split(action_sep) if action.strip()]
        if len(actions) > max_actions:
            actions = actions[:max_actions]
            action_content = (" " + action_sep + " ").join(actions)

    llm_response = "<think>" + think_content.strip() + "</think>" + \
                   "<answer>" + action_content.strip() + "</answer>" + \
                   "<state>" + state_content.strip() + "</state>"
    return {
        "llm_raw_response": response,
        "llm_response": llm_response,
        "think_content": think_content,
        "action_content": action_content,
        "state_content": state_content,
        "actions": actions,
        "format_correct": format_correct
    }

def parse_grounding_worldmodeling(response: str, special_token_list=None, action_sep=',', max_actions=3) -> Dict:
    """
    Parse response in format: <state>...</state><think>...</think><answer>...</answer><state>...</state>
    
    Returns a dict with keys:
    - llm_raw_response: the original response
    - llm_response: the response with all tags
    - state_content: the content inside <state> tag
    - think_content: the content inside <think> tag
    - action_content: the content inside <answer> tag
    - state_content: the content inside <state> tag
    - actions: a list of actions extracted from action_content
    - format_correct: whether the response strictly follows the expected format
    """
    # Pattern to check for content strictly in the expected format
    strict_pattern = r'^\s*<state>(.*?)</state>\s*<think>(.*?)</think>\s*<answer>(.*?)</answer>\s*<state>(.*?)</state>\s*$'
    strict_match = re.match(strict_pattern, response.strip(), re.DOTALL)
    format_correct = strict_match is not None
    
    # Pattern to extract content from tags
    extraction_pattern = r'<state>(.*?)</state>\s*<think>(.*?)</think>\s*<answer>(.*?)</answer>\s*<state>(.*?)</state>'
    match = re.search(extraction_pattern, response, re.DOTALL)
    #format_correct = match is not None
    
    if not strict_match:
        state_content_current, think_content, action_content, state_content_next, actions = "", "", "", "", []
    else:
        state_content_current, think_content, action_content, state_content_next = match.group(1), match.group(2), match.group(3), match.group(4)
        if special_token_list is not None:
            for special_token in special_token_list:
                state_content_current = state_content_current.replace(special_token, "").strip()
                action_content = action_content.replace(special_token, "").strip()
                think_content = think_content.replace(special_token, "").strip()
                state_content_next = state_content_next.replace(special_token, "").strip()
        actions = [action.strip() for action in action_content.split(action_sep) if action.strip()]
        if len(actions) > max_actions:
            actions = actions[:max_actions]
            action_content = (" " + action_sep + " ").join(actions)

    llm_response = "<state>" + state_content_current.strip() + "</state>" + \
                   "<think>" + think_content.strip() + "</think>" + \
                   "<answer>" + action_content.strip() + "</answer>" + \
                   "<state>" + state_content_next.strip() + "</state>"
    return {
        "llm_raw_response": response,
        "llm_response": llm_response,
        "state_content_current": state_content_current,
        "think_content": think_content,
        "action_content": action_content,
        "state_content_next": state_content_next,
        "actions": actions,
        "format_correct": format_correct
    }
    
parse_function_map = {
    "free_think": parse_freethink,
    "no_think": parse_no_think,
    "grounding": parse_grounding,
    "worldmodeling": parse_worldmodeling,
    "grounding_worldmodeling": parse_grounding_worldmodeling,
}

if __name__ == "__main__":
    # Define format options
    format_map = {
        '1': 'free_think',
        '2': 'no_think',
        '3': 'grounding',
        '4': 'worldmodeling',
        '5': 'grounding_worldmodeling',
        'q': 'quit'
    }
    
    # Print available formats
    print("Available formats:")
    print("1: free_think - <think>...</think><answer>...</answer>")
    print("2: no_think - <answer>...</answer>")
    print("3: grounding - <state>...</state><think>...</think><answer>...</answer>")
    print("4: worldmodeling - <think>...</think><answer>...</answer><state>...</state>")
    print("5: grounding_worldmodeling - <state>...</state><think>...</think><answer>...</answer><state>...</state>")
    print("q: quit")
    
    while True:
        # Get format choice
        format_choice = input("\nEnter format number (1-5) or 'q' to quit: ").strip().lower()
        
        if format_choice == 'q':
            print("Exiting program. Goodbye!")
            break
            
        if format_choice not in format_map:
            print("Invalid choice. Please enter a number between 1-5 or 'q'.")
            continue
            
        format_name = format_map[format_choice]
        
        # Get response from user
        print(f"\nEnter a raw LLM response for {format_name} format:")
        raw_response = input()
        
        # Parse the response
        parse_function = parse_function_map[format_name]
        result = parse_function(raw_response)
        
        # Print the result
        print("\n=== Parsing Result ===")
        print(json.dumps(result, indent=2))
        
        print("\n" + "-"*50)  # Separator