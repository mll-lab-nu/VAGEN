import re
import json
from typing import List, Dict, Any, Optional

def parse_llm_json_response_flexible(llm_output_string: str) -> Optional[List[Dict[str, Any]]]:
    """
    Parses the JSON array from the LLM's output string.
    It first attempts to find the JSON within a ```json ... ``` block
    after "Output JSON:". If that fails, it searches the entire string
    for the first valid JSON array of dictionaries.

    Args:
        llm_output_string: The raw string output received from the LLM.

    Returns:
        A Python list representing the parsed JSON array, or None if parsing fails
        or the JSON block/list is not found in the expected format.
    """
    if not isinstance(llm_output_string, str):
        print("Input is not a string.")
        return None

    # Handle the potential {{}} based on the prompt examples.
    # Replace them with { and } *before* any parsing attempts.
    processed_string = llm_output_string.replace('{{', '{').replace('}}', '}')

    # --- Attempt 1: Find JSON within ```json ... ``` block after "Output JSON:" ---
    print("Attempting to parse from ```json ... ``` block after 'Output JSON:'")
    
    # More specific pattern to ensure we get the right block
    pattern = r"Output JSON:.*?```json\s*\n(.*?)\n\s*```"
    matches = re.finditer(pattern, processed_string, re.DOTALL)
    
    for match in matches:
        json_content = match.group(1).strip()
        if json_content:
            try:
                parsed_json = json.loads(json_content)
                if isinstance(parsed_json, list) and all(isinstance(item, dict) for item in parsed_json):
                    print("Successfully parsed from ```json block.")
                    return parsed_json
                else:
                    print(f"Parsed content from ```json block is not a list of dictionaries.")
            except json.JSONDecodeError as e:
                print(f"JSON parsing failed for ```json block content: {e}")
                print(f"Attempted to parse string:\n---\n{json_content}\n---")
            except Exception as e:
                print(f"An unexpected error occurred during parsing ```json block content: {e}")

    print("Could not find valid JSON in 'Output JSON:' and ```json ... ``` block structure.")

    # --- Attempt 2: Search for the first valid JSON array in the entire string ---
    print("Attempting to find the first valid JSON array in the entire string.")
    
    # Use a more efficient approach with bracket matching
    return find_json_array_in_string(processed_string)


def find_json_array_in_string(text: str) -> Optional[List[Dict[str, Any]]]:
    """
    Efficiently find and parse the first valid JSON array in a string.
    
    Args:
        text: The string to search in.
        
    Returns:
        The first valid JSON array found, or None if no valid array is found.
    """
    i = 0
    while i < len(text):
        # Find the next '[' character
        start = text.find('[', i)
        if start == -1:
            break
            
        # Try to find the matching ']' using bracket counting
        bracket_count = 0
        in_string = False
        escape_next = False
        end = start
        
        for j in range(start, len(text)):
            char = text[j]
            
            # Handle string literals to avoid counting brackets inside strings
            if not escape_next:
                if char == '"' and not in_string:
                    in_string = True
                elif char == '"' and in_string:
                    in_string = False
                elif char == '\\' and in_string:
                    escape_next = True
                    continue
                    
                # Count brackets only when not inside a string
                if not in_string:
                    if char == '[':
                        bracket_count += 1
                    elif char == ']':
                        bracket_count -= 1
                        
                    if bracket_count == 0:
                        end = j
                        break
            else:
                escape_next = False
        
        # If we found a matching ']', try to parse
        if bracket_count == 0 and end > start:
            candidate = text[start:end+1]
            try:
                parsed = json.loads(candidate)
                
                # Verify it's a list of dictionaries
                if isinstance(parsed, list):
                    # Empty list is valid, or all items must be dictionaries
                    if not parsed or all(isinstance(item, dict) for item in parsed):
                        print("Successfully parsed the first valid JSON array found.")
                        return parsed
                    else:
                        print("Found a JSON list, but its items are not all dictionaries. Continuing search.")
            except json.JSONDecodeError:
                # Not valid JSON, continue searching
                pass
            except Exception as e:
                print(f"Unexpected error during parsing: {e}")
        
        # Move past this '[' and continue searching
        i = start + 1
    
    print("Failed to find and parse a valid JSON array in the expected format.")
    return None


# Example usage and test cases
if __name__ == "__main__":
    # Test case 1: JSON in code block
    test1 = '''
    Some text here
    Output JSON:
    Here's the result:
    ```json
    [{"name": "Alice", "as": null}, {"name": "Bob", "age": 25}]
    ```
    '''
    print(parse_llm_json_response_flexible(test1))
    
    # Test case 2: JSON with double braces
    test2 = '''
    Output JSON:
    ```json
    [{{"name": "Charlie", "age": 35}}, {{"name": "David", "age": 40}}]
    ```
    '''
    print(parse_llm_json_response_flexible(test2))
    # Test case 3: JSON not in code block
    
    