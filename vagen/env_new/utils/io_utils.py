import functools
from typing import Dict, Tuple, Any, Callable, TypeVar
from PIL import Image
import re

# Assuming this is defined elsewhere in your code
IMAGE_PLACEHOLDER = "<image>"

T = TypeVar('T')

def validate_step_io(func: Callable[..., T]) -> Callable[..., T]:
    """Decorator to validate input and output types for the step method."""
    @functools.wraps(func)
    def wrapper(self, action, *args, **kwargs):
        # Validate input
        assert isinstance(action, str), f"action must be str, got {type(action)}"
        
        # Call the function
        result = func(self, action, *args, **kwargs)
        
        # Validate output structure
        assert isinstance(result, tuple) and len(result) == 4, f"step must return a tuple of length 4, got {type(result)} of length {len(result) if isinstance(result, tuple) else 'N/A'}"
        
        obs, reward, done, info = result
        
        # Validate types of returned values
        assert isinstance(reward, (int, float)), f"reward must be int or float, got {type(reward)}"
        assert isinstance(done, bool), f"done must be bool, got {type(done)}"
        assert isinstance(info, dict), f"info must be dict, got {type(info)}"
        assert isinstance(obs, dict), f"obs must be dict, got {type(obs)}"
        
        # Validate required keys and their types
        assert "llm_raw_response" in info, f"info must contain 'llm_raw_response' key"
        assert isinstance(info["llm_raw_response"], str), f"info['llm_raw_response'] must be str, got {type(info['llm_raw_response'])}"
        assert "text_template" in obs, f"obs must contain 'text_template' key"
        assert isinstance(obs["text_template"], str), f"obs['text_template'] must be str, got {type(obs['text_template'])}"
        
        # Validate multi_modal_data if present
        if "multi_modal_data" in obs:
            if IMAGE_PLACEHOLDER in obs["multi_modal_data"]:
                assert isinstance(obs["multi_modal_data"][IMAGE_PLACEHOLDER], list), f"obs['multi_modal_data']['<image>'] must be list, got {type(obs['multi_modal_data'][IMAGE_PLACEHOLDER])}"
                
                for image in obs["multi_modal_data"][IMAGE_PLACEHOLDER]:
                    assert isinstance(image, Image.Image), f"image must be PIL.Image.Image, got {type(image)}"
                
                len_of_images = len(obs["multi_modal_data"][IMAGE_PLACEHOLDER])
                len_of_image_in_text_template = len(re.findall(IMAGE_PLACEHOLDER, obs["text_template"]))
                assert len_of_images == len_of_image_in_text_template, f"len_of_images must be equal to len_of_image_in_text_template, got {len_of_images} and {len_of_image_in_text_template}"
        
        return result
    
    return wrapper