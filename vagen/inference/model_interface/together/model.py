import base64
import logging
import re
import json
import os
import sys
from typing import List, Dict, Any
from concurrent.futures import ThreadPoolExecutor
import requests
from PIL import Image
import io

from vagen.inference.model_interface.base_model import BaseModelInterface
from .model_config import TogetherModelConfig

logger = logging.getLogger(__name__)

class TogetherModelInterface(BaseModelInterface):
    """Model interface for Together AI API with Qwen format compatibility."""
    
    def __init__(self, config: TogetherModelConfig):
        super().__init__(config)
        self.config = config
        
        # Get API key from config or environment
        self.api_key = config.api_key or os.environ.get("TOGETHER_API_KEY")
        
        # Check if API key is available
        if not self.api_key:
            error_msg = "ERROR: Together API key not set. Please set the TOGETHER_API_KEY environment variable or provide api_key in config."
            logger.error(error_msg)
            print(error_msg, file=sys.stderr)
            sys.exit(1)
            
        # Base URL
        self.base_url = config.base_url
        
        # Thread pool for batch processing
        self.executor = ThreadPoolExecutor(max_workers=10)
        
        logger.info(f"Initialized Together AI interface with model {config.model_name}")
    
    def generate(self, prompts: List[Any], **kwargs) -> List[Dict[str, Any]]:
        """Generate responses using Together AI API."""
        # Make parallel API calls
        futures = []
        
        for prompt in prompts:
            # Keep original Qwen format for processing
            future = self.executor.submit(
                self._single_api_call,
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
                # Check if error is related to API key
                error_str = str(e)
                if "401" in error_str or "api_key" in error_str.lower() or "unauthorized" in error_str.lower():
                    error_msg = f"ERROR: API key invalid or unauthorized: {error_str}"
                    logger.error(error_msg)
                    print(error_msg, file=sys.stderr)
                    sys.exit(1)
                
                logger.error(f"API call failed: {e}")
                results.append({
                    "text": f"Error: {str(e)}",
                    "error": str(e)
                })
        
        return results
    
    def _prepare_together_request(self, prompt: List[Dict], **kwargs) -> Dict:
        """
        Convert Qwen format messages to Together AI request format.
        Together API supports OpenAI-compatible format.
        """
        # Convert Qwen messages to OpenAI format
        messages = self._convert_qwen_to_together_format(prompt)
        
        # Prepare request payload
        request_data = {
            "model": self.config.model_name,
            "messages": messages,
            "max_tokens": kwargs.get("max_tokens", self.config.max_tokens),
            "temperature": kwargs.get("temperature", self.config.temperature),
            "top_p": kwargs.get("top_p", self.config.top_p),
            "top_k": kwargs.get("top_k", self.config.top_k),
        }
        
        # Add optional parameters if provided
        if self.config.seed is not None:
            request_data["seed"] = kwargs.get("seed", self.config.seed)
        
        if self.config.presence_penalty != 0:
            request_data["presence_penalty"] = kwargs.get("presence_penalty", self.config.presence_penalty)
        
        if self.config.frequency_penalty != 0:
            request_data["frequency_penalty"] = kwargs.get("frequency_penalty", self.config.frequency_penalty)
        
        return request_data
    
    def _convert_qwen_to_together_format(self, prompt: List[Dict]) -> List[Dict]:
        """
        Convert Qwen format messages to Together AI format.
        
        Qwen format: Text with <image> placeholders + separate multi_modal_data
        Together format: For multimodal models, uses OpenAI-compatible format
        """
        together_messages = []
        
        for message in prompt:
            role = message.get("role", "user")
            content = message.get("content", "")
            
            # Create Together AI message structure (OpenAI compatible)
            together_msg = {
                "role": role,
            }
            
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
                content_array = []
                for i, part in enumerate(parts):
                    # Add text part if not empty
                    if part.strip():
                        content_array.append({
                            "type": "text",
                            "text": part
                        })
                    
                    # Add image if available (except for last part)
                    if i < len(parts) - 1 and i < len(images):
                        image_data = self._process_image_for_together(images[i])
                        content_array.append({
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{image_data}"
                            }
                        })
                
                together_msg["content"] = content_array
            else:
                # Text-only message - use simple content string 
                # (Together supports both formats)
                together_msg["content"] = content
            
            together_messages.append(together_msg)
        
        return together_messages
    
    def _process_image_for_together(self, image: Any) -> str:
        """Convert image to base64 for Together AI API."""
        if isinstance(image, Image.Image):
            # Ensure RGB mode
            if image.mode != "RGB":
                image = image.convert("RGB")
            
            # Resize if too large to save tokens
            max_size = 1024
            if max(image.size) > max_size:
                ratio = max_size / max(image.size)
                new_size = tuple(int(dim * ratio) for dim in image.size)
                image = image.resize(new_size, Image.Resampling.LANCZOS)
            
            buffered = io.BytesIO()
            image.save(buffered, format="JPEG", quality=85)
            return base64.b64encode(buffered.getvalue()).decode()
            
        elif isinstance(image, dict) and "__pil_image__" in image:
            from vagen.server.serial import deserialize_pil_image
            pil_image = deserialize_pil_image(image)
            return self._process_image_for_together(pil_image)
        else:
            raise ValueError(f"Unsupported image type: {type(image)}")
    
    def _single_api_call(self, prompt: List[Dict], **kwargs) -> Dict[str, Any]:
        """Make a single API call to Together AI."""
        try:
            # Prepare request data
            request_data = self._prepare_together_request(prompt, **kwargs)
            
            # Set headers with explicit API key
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            
            # Debug information
            logger.debug(f"API URL: {self.base_url}/v1/chat/completions")
            logger.debug(f"Headers: {json.dumps({k: '***' if k == 'Authorization' else v for k, v in headers.items()})}")
            logger.debug(f"Request data: {json.dumps(request_data)}")
            
            # Make the API call - ensure we're using POST
            response = requests.post(
                f"{self.base_url}/v1/chat/completions",
                headers=headers,
                json=request_data,
                timeout=self.config.timeout
            )
            
            # Debug response
            logger.debug(f"Status code: {response.status_code}")
            logger.debug(f"Response headers: {dict(response.headers)}")
            logger.debug(f"Response body: {response.text[:200]}...")
            
            # Check for API key errors specifically
            if response.status_code == 401:
                error_msg = f"ERROR: API key invalid or unauthorized. Status code: {response.status_code}, Response: {response.text}"
                logger.error(error_msg)
                print(error_msg, file=sys.stderr)
                sys.exit(1)
            
            # Check for other errors
            response.raise_for_status()
            
            # Parse the response
            response_data = response.json()
            
            # Extract text response
            response_text = response_data["choices"][0]["message"]["content"]
            
            # Return response in Qwen format
            return {
                "text": response_text,
                "usage": {
                    "prompt_tokens": response_data.get("usage", {}).get("prompt_tokens", 0),
                    "completion_tokens": response_data.get("usage", {}).get("completion_tokens", 0),
                    "total_tokens": response_data.get("usage", {}).get("total_tokens", 0)
                },
                "finish_reason": response_data["choices"][0].get("finish_reason", "unknown")
            }
            
        except requests.exceptions.HTTPError as e:
            if hasattr(e, 'response') and e.response.status_code == 401:
                error_msg = f"ERROR: API key invalid or unauthorized: {e}"
                logger.error(error_msg)
                print(error_msg, file=sys.stderr)
                sys.exit(1)
            logger.error(f"HTTP error: {e}")
            raise
        except Exception as e:
            error_str = str(e)
            # Check for API key related errors in the exception message
            if "401" in error_str or "api_key" in error_str.lower() or "unauthorized" in error_str.lower():
                error_msg = f"ERROR: API key invalid or unauthorized: {error_str}"
                logger.error(error_msg)
                print(error_msg, file=sys.stderr)
                sys.exit(1)
                
            logger.error(f"Together AI API error: {e}")
            if 'response' in locals():
                logger.error(f"Response status: {response.status_code}")
                logger.error(f"Response body: {response.text}")
            raise
    
    def format_prompt(self, messages: List[Dict[str, Any]]) -> str:
        """
        Format conversation messages into a prompt string.
        
        For Together AI, returns a string representation of the messages
        for logging/debugging purposes.
        """
        formatted = []
        
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            
            # Handle different roles
            if role == "system":
                formatted.append(f"System: {content}")
            elif role == "user":
                formatted.append(f"User: {content}")
            elif role == "assistant":
                formatted.append(f"Assistant: {content}")
            else:
                formatted.append(f"{role.capitalize()}: {content}")
        
        return "\n".join(formatted)
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get detailed information about the model."""
        info = super().get_model_info()
        
        # Determine multimodal support based on model name
        # This is a simplification - actual determination may require more logic
        multimodal_models = [
            "llava", "qwen-vl", "cogvlm", "bakllava", "internlm-xcomposer"
        ]
        is_multimodal = any(model_type in self.config.model_name.lower() for model_type in multimodal_models)
        
        info.update({
            "name": self.config.model_name,
            "type": "multimodal" if is_multimodal else "text",
            "supports_images": is_multimodal,
            "max_tokens": self.config.max_tokens,
            "temperature": self.config.temperature,
            "top_p": self.config.top_p,
            "top_k": self.config.top_k,
            "config_id": self.config.config_id()
        })
        
        return info