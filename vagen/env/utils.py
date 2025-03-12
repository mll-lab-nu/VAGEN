import re
from PIL import Image
import numpy as np


def preprocess_text(text: str) -> dict:
    """Preprocess the raw text from llm to a list of strings

    1. Extract think from the first <think> ... </think>
    2. Extract answer from the first <answer> ... </answer>
    3. Split the answer by comma into a list of strings
    
    Args:
        text: raw text from llm

    Returns:
        dict with keys: llm_raw_response, think, answer_list
    """
    # Extract content from <think> tags if they exist
    think_match = re.search(r'<think>(.*?)</think>', text, re.DOTALL)
    
    # Extract content from <answer> tags
    answer_match = re.search(r'<answer>(.*?)</answer>', text, re.DOTALL)

    answer_list, thinking, answer_content = [], "", ""
    
    if think_match:
        thinking = think_match.group(1).strip()
    
    if answer_match:
        # Get the answer content and split by comma
        answer_content = answer_match.group(1).strip()
        # Split by comma and strip whitespace from each item
        answer_list = [item.strip() for item in answer_content.split(',') if item.strip()]
    
    return {
        'llm_raw_response': text,
        'answer_list': answer_list,
        'think': thinking,
        'answer': answer_content
    }

def convert_numpy_to_PIL(numpy_array: np.ndarray) -> Image.Image:
        """Convert a numpy array to a PIL RGB image."""
        if numpy_array.shape[-1] == 3:
            # Convert numpy array to RGB PIL Image
            return Image.fromarray(numpy_array, mode='RGB')
        else:
            raise ValueError(f"Unsupported number of channels: {numpy_array.shape[-1]}. Expected 3 (RGB).")


if __name__ == "__main__":
    text = """
    <think>
    I am thinking about the problem.
    </think>
    <answer>
    answer1, answer2, answer3
    </answer>
    """
    print(preprocess_text(text))