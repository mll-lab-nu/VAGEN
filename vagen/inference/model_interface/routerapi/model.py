import base64
import logging
import io
from typing import List, Dict, Any
from concurrent.futures import ThreadPoolExecutor
from openai import OpenAI
from PIL import Image

from vagen.mllm_agent.model_interface.base_model import BaseModelInterface
from .model_config import RouterAPIModelConfig

logger = logging.getLogger(__name__)

class RouterAPIModelInterface(BaseModelInterface):
    """Model interface for OpenRouter API with Qwen format compatibility."""
    
    def __init__(self, config: RouterAPIModelConfig):
        super().__init__(config)
        self.config = config
        
        # Initialize OpenAI client with OpenRouter base URL
        self.client = OpenAI(
            api_key=config.api_key,
            base_url=config.base_url
        )
        
        # Thread pool for batch processing
        self.executor = ThreadPoolExecutor(max_workers=10)
        
        logger.info(f"Initialized RouterAPI interface with model {config.model_name}")
    
    def generate(self, prompts: List[Any], **kwargs) -> List[Dict[str, Any]]:
        """Generate responses using OpenRouter API."""
        # Process prompts into OpenRouter message format
        formatted_requests = []
        
        for prompt in prompts:
            messages = self._convert_qwen_to_router_format(prompt)
            formatted_requests.append(messages)
        
        # Make parallel API calls
        futures = []
        for messages in formatted_requests:
            future = self.executor.submit(
                self._single_api_call,
                messages,
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
    
    def _convert_qwen_to_router_format(self, prompt: List[Dict]) -> List[Dict]:
        """
        Convert Qwen format messages to OpenRouter format.
        
        Qwen format: Text with <image> placeholders + separate multi_modal_data
        OpenRouter format: Structured content array with text and image objects
        """
        router_messages = []
        
        for message in prompt:
            role = message.get("role", "user")
            content = message.get("content", "")
            
            # Create OpenRouter message structure for content
            router_content = []
            
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
                        router_content.append({
                            "type": "text",
                            "text": part
                        })
                    
                    # Add image if available (except for last part)
                    if i < len(parts) - 1 and i < len(images):
                        image_data = self._process_image_for_router(images[i])
                        router_content.append({
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{image_data}"
                            }
                        })
            else:
                # Text-only message
                router_content.append({
                    "type": "text",
                    "text": content
                })
            
            # Create the final router message
            router_messages.append({
                "role": role,
                "content": router_content
            })
        
        return router_messages
    
    def _process_image_for_router(self, image: Any) -> str:
        """Convert image to base64 for OpenRouter API."""
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
            return self._process_image_for_router(pil_image)
        else:
            raise ValueError(f"Unsupported image type: {type(image)}")
    
    def _single_api_call(self, messages: List[Dict], **kwargs) -> Dict[str, Any]:
        """Make a single API call to OpenRouter."""
        try:
            # Create extra headers for OpenRouter
            extra_headers = {}
            if self.config.site_url:
                extra_headers["HTTP-Referer"] = self.config.site_url
            if self.config.site_name:
                extra_headers["X-Title"] = self.config.site_name
            
            response = self.client.chat.completions.create(
                model=self.config.model_name,
                messages=messages,
                max_tokens=kwargs.get("max_tokens", self.config.max_tokens),
                temperature=kwargs.get("temperature", self.config.temperature),
                presence_penalty=kwargs.get("presence_penalty", self.config.presence_penalty),
                frequency_penalty=kwargs.get("frequency_penalty", self.config.frequency_penalty),
                seed=kwargs.get("seed", self.config.seed),
                extra_headers=extra_headers,
                extra_body={}  # Can be extended for additional parameters
            )
            
            # Extract text response
            response_text = response.choices[0].message.content
            
            # Build response in Qwen compatible format
            result = {
                "text": response_text,
                "usage": {
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens
                },
                "finish_reason": response.choices[0].finish_reason
            }
            
            return result
            
        except Exception as e:
            logger.error(f"OpenRouter API error: {e}")
            raise
    
    def format_prompt(self, messages: List[Dict[str, Any]]) -> str:
        """
        Format prompt for compatibility.
        
        Since OpenRouter uses structured messages, this returns a string representation
        of the messages for logging/debugging purposes.
        """
        formatted = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            
            # Handle multiple content types if needed
            if isinstance(content, list):
                text_parts = []
                for item in content:
                    if item.get("type") == "text":
                        text_parts.append(item.get("text", ""))
                    elif item.get("type") == "image_url":
                        text_parts.append("[IMAGE]")
                content_str = " ".join(text_parts)
            else:
                content_str = content
            
            # Format based on role
            if role == "system":
                formatted.append(f"System: {content_str}")
            elif role == "user":
                formatted.append(f"User: {content_str}")
            elif role == "assistant":
                formatted.append(f"Assistant: {content_str}")
        
        return "\n".join(formatted)
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get detailed information about the model."""
        info = super().get_model_info()
        
        info.update({
            "name": self.config.model_name,
            "type": "multimodal",  # All specified Qwen models support vision
            "supports_images": True,
            "max_tokens": self.config.max_tokens,
            "temperature": self.config.temperature,
            "config_id": self.config.config_id(),
            "provider": "OpenRouter"
        })
        
        return info