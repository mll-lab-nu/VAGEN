# test_vllm_model.py
import sys
import os
from PIL import Image
import numpy as np
import logging
import json

# Setup logging
logging.basicConfig(level=logging.INFO)

# Add parent directory to path to import the module
sys.path.append("../")

from vagen.mllm_agent.model_interface.providers.vllm_model import VLLMModelInterface

def create_test_image():
    """Create a simple test image"""
    # Create a 100x100 RGB image with a gradient
    img_array = np.zeros((100, 100, 3), dtype=np.uint8)
    
    # Create a simple gradient
    for i in range(100):
        for j in range(100):
            img_array[i, j, 0] = i * 255 // 100  # Red channel
            img_array[i, j, 1] = j * 255 // 100  # Green channel
            img_array[i, j, 2] = 100             # Blue channel
    
    # Convert to PIL Image
    img = Image.fromarray(img_array)
    return img

def test_text_model():
    """Test the text-only Qwen model"""
    print("\n===== Testing Qwen2.5-0.5B-Instruct (Text) =====")
    
    # Configuration for text model
    config = {
        "model_name": "Qwen/Qwen2.5-0.5B-Instruct",
        "model_family": "qwen",
        "max_tokens": 256,
        "temperature": 0.7
    }
    
    try:
        # Initialize model
        model = VLLMModelInterface(config)
        print(f"Successfully initialized model: {model.get_model_info()['name']}")
        
        # Test conversation
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What is the capital of France?"}
        ]
        
        # Generate response
        print("Generating response...")
        results = model.generate([messages])
        
        # Print result
        if results and len(results) > 0:
            print("\nResponse:")
            print(results[0]["text"])
            print(f"\nToken usage: {results[0]['usage']}")
        else:
            print("No response generated.")
            
    except Exception as e:
        print(f"Error testing text model: {str(e)}")

def test_multimodal_model():
    """Test the multimodal Qwen model"""
    print("\n===== Testing Qwen2.5-VL-3B-Instruct (Multimodal) =====")
    
    # Configuration for multimodal model
    config = {
        "model_name": "Qwen/Qwen2.5-VL-3B-Instruct",
        "model_family": "qwen",
        "max_tokens": 256,
        "temperature": 0.7
    }
    
    try:
        # Initialize model
        model = VLLMModelInterface(config)
        print(f"Successfully initialized model: {model.get_model_info()['name']}")
        
        # Create test image
        test_image = create_test_image()
        
        # Test conversation with image
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What do you see in this image?"}
        ]
        
        # Generate response with image
        print("Generating response with image...")
        prompt = {
            "messages": messages,
            "images": [test_image]
        }
        
        results = model.generate([prompt])
        
        # Print result
        if results and len(results) > 0:
            print("\nResponse:")
            print(results[0]["text"])
            print(f"\nToken usage: {results[0]['usage']}")
        else:
            print("No response generated.")
            
    except Exception as e:
        print(f"Error testing multimodal model: {str(e)}")

if __name__ == "__main__":
    # Test text model
    test_text_model()
    
    # Test multimodal model
    test_multimodal_model()
    
    print("\nTests completed!")