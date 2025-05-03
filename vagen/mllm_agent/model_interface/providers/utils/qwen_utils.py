"""
Utility functions for Qwen models in vLLM.
"""

import base64
import io
from typing import List, Dict, Any
from PIL import Image

def format_qwen_prompt(messages: List[Dict[str, Any]]) -> str:
    """
    Format conversation messages into a Qwen compatible prompt string.
    
    Args:
        messages: List of message dictionaries with 'role' and 'content'
        
    Returns:
        Formatted prompt string for Qwen models
    """
    # Initialize with Qwen chat format
    formatted_prompt = ""
    
    for message in messages:
        role = message.get("role", "").lower()
        content = message.get("content", "")
        
        # Handle images that might be in multi_modal_data
        if "multi_modal_data" in message and "images" in message["multi_modal_data"]:
            # Get images
            images = message["multi_modal_data"]["images"]
            # Insert images into the prompt according to Qwen format
            if role == "user" and images:
                content = insert_qwen_image_tags(content, images)
        
        # Apply Qwen chat format
        if role == "system":
            formatted_prompt += f"<|im_start|>system\n{content}<|im_end|>\n"
        elif role == "user":
            formatted_prompt += f"<|im_start|>user\n{content}<|im_end|>\n"
        elif role == "assistant":
            formatted_prompt += f"<|im_start|>assistant\n{content}<|im_end|>\n"
    
    # Add assistant prefix to generate response
    formatted_prompt += "<|im_start|>assistant\n"
    
    return formatted_prompt

def insert_qwen_image_tags(content: str, images: List[Any]) -> str:
    """
    Insert Qwen-specific image tags into content string.
    
    Args:
        content: Original text content
        images: List of images (PIL Images or serialized image data)
        
    Returns:
        Content with image tags inserted
    """
    # Convert images to base64 strings
    image_tags = []
    
    for img in images:
        # Handle different image formats
        if isinstance(img, Image.Image):
            # Convert PIL Image to base64
            buffer = io.BytesIO()
            img.save(buffer, format="PNG")
            img_b64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
            image_tags.append(f"<img>{img_b64}</img>")
        elif isinstance(img, dict) and "__pil_image__" in img:
            # Already serialized image
            img_b64 = img["__pil_image__"]
            image_tags.append(f"<img>{img_b64}</img>")
        elif isinstance(img, str) and img.startswith("data:image"):
            # Data URL format
            img_b64 = img.split(',')[1]
            image_tags.append(f"<img>{img_b64}</img>")
        elif isinstance(img, bytes):
            # Raw bytes
            img_b64 = base64.b64encode(img).decode('utf-8')
            image_tags.append(f"<img>{img_b64}</img>")
    
    # Add image tags before the content
    formatted_content = "".join(image_tags) + content
    return formatted_content

def process_qwen_images(images: List[Image.Image]) -> List[Image.Image]:
    """
    Process images for Qwen multimodal models.
    
    Args:
        images: List of PIL Image objects
        
    Returns:
        Processed images ready for the model
    """
    # For Qwen models, convert to RGB and ensure proper format
    processed_images = []
    for img in images:
        # Ensure image is in RGB mode
        if img.mode != "RGB":
            img = img.convert("RGB")
        
        # Qwen already handles resizing internally,
        # but we can add any preprocessing here if needed
        processed_images.append(img)
        
    return processed_images