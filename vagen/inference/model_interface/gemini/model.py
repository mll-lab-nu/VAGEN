import base64
import logging
import os
from typing import List, Dict, Any, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor
from PIL import Image
import io
import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold

from vagen.inference.model_interface.base_model import BaseModelInterface
from .model_config import GeminiModelConfig

logger = logging.getLogger(__name__)

class GeminiModelInterface(BaseModelInterface):
    """Model interface for Google Gemini API with Qwen format compatibility."""
    
    def __init__(self, config: GeminiModelConfig):
        super().__init__(config)
        self.config = config
        
        # Initialize Gemini client
        api_key = config.api_key or os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("Gemini API key must be provided in config or GOOGLE_API_KEY environment variable")
        
        # Configure Gemini API
        genai.configure(api_key=api_key)
        
        # Get model
        self.model = genai.GenerativeModel(config.model_name)
        
        # Thread pool for batch processing
        self.executor = ThreadPoolExecutor(max_workers=10)
        
        logger.info(f"Initialized Gemini interface with model {config.model_name}")
    
    def generate(self, prompts: List[Any], **kwargs) -> List[Dict[str, Any]]:
        """Generate responses using Gemini API."""
        # Make parallel API calls (one per prompt)
        futures = []
        for prompt in prompts:
            future = self.executor.submit(
                self._process_single_prompt,
                prompt,
                **kwargs
            )
            futures.append(future)
        
        # Collect results
        results = []
        for future in futures:
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                logger.error(f"API call failed: {e}")
                results.append({
                    "text": f"Error: {str(e)}",
                    "error": str(e)
                })
        
        return results
    
    def _process_single_prompt(self, prompt: List[Dict], **kwargs) -> Dict[str, Any]:
        """Process a single prompt through Gemini API."""
        try:
            # Extract system prompt if present in the message list
            system_prompt = self._extract_system_prompt(prompt)
            
            # Convert Qwen format to Gemini format
            content_parts = self._convert_qwen_to_gemini_format(prompt)
            
            # Prepare generation configuration
            generation_config = {
                "max_output_tokens": kwargs.get("max_tokens", self.config.max_tokens),
                "temperature": kwargs.get("temperature", self.config.temperature),
                "top_p": kwargs.get("top_p", self.config.top_p),
                "top_k": kwargs.get("top_k", self.config.top_k),
                "stop_sequences": kwargs.get("stop_sequences", self.config.stop_sequences),
            }
            
            # Since Gemini API doesn't have a dedicated system prompt parameter,
            # we prepend it to the user's content as a workaround
            if system_prompt:
                # Method 1: Add system prompt at the beginning of content
                # This approach is common in models without dedicated system prompt support
                if content_parts and isinstance(content_parts[0], str):
                    # If first part is text, prepend system prompt to it
                    content_parts[0] = f"System: {system_prompt}\n\n{content_parts[0]}"
                else:
                    # Otherwise insert as new first element
                    content_parts.insert(0, f"System: {system_prompt}\n\n")
            
            # Use direct generation instead of chat session
            # This avoids the error with system_instruction parameter
            response = self.model.generate_content(
                content_parts,
                generation_config=generation_config
            )
            
            # Extract text response
            response_text = response.text
            
            # Calculate usage (token counts are estimated since Gemini doesn't provide them)
            text_content = "".join([part for part in content_parts if isinstance(part, str)])
            prompt_tokens_estimate = len(text_content) // 4  # Rough estimate
            completion_tokens_estimate = len(response_text) // 4  # Rough estimate
            
            return {
                "text": response_text,
                "usage": {
                    "prompt_tokens": prompt_tokens_estimate,
                    "completion_tokens": completion_tokens_estimate,
                    "total_tokens": prompt_tokens_estimate + completion_tokens_estimate
                },
                "finish_reason": "stop"  # Gemini doesn't provide this info directly
            }
            
        except Exception as e:
            logger.error(f"Gemini API error: {e}")
            raise
    
    def _convert_qwen_to_gemini_format(self, messages: List[Dict]) -> List[Any]:
        """
        Convert Qwen format messages to Gemini content parts.
        
        Qwen format: Text with <image> placeholders + separate multi_modal_data
        Gemini format: List of content parts (strings and images)
        
        Returns:
            List of content parts (texts and images)
        """
        # Final content parts for Gemini
        gemini_content = []
        
        # Process messages to build content
        for message in messages:
            role = message.get("role", "user")
            content = message.get("content", "")
            
            # Skip system messages (handled separately)
            if role == "system":
                continue
            
            # Handle multimodal content
            if "multi_modal_data" in message and "<image>" in content:
                # Extract images from multi_modal_data
                images = []
                for key, values in message["multi_modal_data"].items():
                    if key == "<image>" or "image" in key.lower():
                        images.extend(values)
                
                # Split content by <image> placeholders
                parts = content.split("<image>")
                
                # Build content array alternating text and images
                for i, part in enumerate(parts):
                    # Add text part if not empty
                    if part.strip():
                        gemini_content.append(part.strip())
                    
                    # Add image if available (except after last part)
                    if i < len(parts) - 1 and i < len(images):
                        processed_image = self._process_image_for_gemini(images[i])
                        gemini_content.append(processed_image)
            else:
                # Text-only message
                gemini_content.append(content)
        
        return gemini_content
    
    def _extract_system_prompt(self, messages: List[Dict]) -> Optional[str]:
        """Extract system prompt from messages if present."""
        for message in messages:
            if message.get("role") == "system":
                return message.get("content", "")
        return None
    
    def _process_image_for_gemini(self, image: Any) -> Image.Image:
        """
        Process image for Gemini API.
        Unlike Claude/OpenAI, Gemini directly accepts PIL Image objects.
        """
        if isinstance(image, Image.Image):
            # Ensure RGB mode
            if image.mode != "RGB":
                image = image.convert("RGB")
            
            # Resize if too large (to optimize token usage)
            max_size = 1568  # Recommended max for Gemini
            if max(image.size) > max_size:
                ratio = max_size / max(image.size)
                new_size = tuple(int(dim * ratio) for dim in image.size)
                image = image.resize(new_size, Image.Resampling.LANCZOS)
            
            return image
            
        elif isinstance(image, dict) and "__pil_image__" in image:
            # Handle serialized PIL image
            from vagen.server.serial import deserialize_pil_image
            pil_image = deserialize_pil_image(image)
            return self._process_image_for_gemini(pil_image)
        else:
            raise ValueError(f"Unsupported image type: {type(image)}")
    
    def format_prompt(self, messages: List[Dict[str, Any]]) -> str:
        """
        Format prompt for debugging/logging purposes.
        
        Since Gemini has a different structure, this provides a string
        representation that's easy to understand.
        """
        formatted = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            
            # Handle Qwen special tokens if present
            if role == "system":
                formatted.append(f"System: {content}")
            elif role == "user":
                formatted.append(f"User: {content}")
            elif role == "assistant":
                formatted.append(f"Assistant: {content}")
        
        return "\n".join(formatted)
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get detailed information about the model."""
        info = super().get_model_info()
        
        info.update({
            "name": self.config.model_name,
            "type": "multimodal", 
            "supports_images": True,
            "max_tokens": self.config.max_tokens,
            "temperature": self.config.temperature,
            "config_id": self.config.config_id()
        })
        
        return info