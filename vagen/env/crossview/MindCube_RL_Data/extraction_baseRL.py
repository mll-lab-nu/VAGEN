"""
JSON extraction and validation module for cognitive maps.

This module provides functions to:
1. Extract JSON from text responses
2. Validate JSON structure for cognitive maps
3. Determine the format type (complex vs. simple)
"""

import re
from typing import Optional, Tuple

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

def format_checking_pipeline(answer_str: str) -> Tuple[bool, int]:
    """
    Check the format of the answer string.
    """
    # 0. extract the response from the text
    response = extract_response_from_text(answer_str)
    if response is None:
        print(f"No response found in the answer string")
        return False, 0 # "No response found in the answer string"
    
    # 1. extract the answer from the text
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


if __name__ == "__main__":
    test_extract_answer()