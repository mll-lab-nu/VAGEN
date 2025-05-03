"""
vLLM model interface for local inference.
Provides a standardized interface to interact with Qwen models via vLLM.
"""

import base64
import io
import logging
from typing import List, Dict, Any
from PIL import Image

from vllm import LLM, SamplingParams

from ..base_model import BaseModelInterface

logger = logging.getLogger(__name__)

class VLLMModelInterface(BaseModelInterface):
    """
    Model interface for local inference using vLLM.
    Specifically designed for Qwen models with support for both text and multimodal inputs.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the vLLM model interface.
        
        Args:
            config: Configuration dictionary containing:
                - model_name: Model name or path (e.g. "Qwen/Qwen2.5-VL-3B-Instruct")
                - tensor_parallel_size: Number of GPUs for tensor parallelism
                - max_tokens: Maximum tokens to generate
                - temperature: Sampling temperature
                - top_p: Top-p sampling parameter
                - top_k: Top-k sampling parameter
                - seed: Random seed
        """
        super().__init__(config)
        
        # Extract config parameters with defaults
        self.model_name = config.get("model_name", "Qwen/Qwen2.5-0.5B-Instruct")
        tensor_parallel_size = config.get("tensor_parallel_size", 1)
        
        # Model loading parameters
        dtype = config.get("dtype", "bfloat16")
        trust_remote_code = config.get("trust_remote_code", True)  # Required for Qwen
        enforce_eager = config.get("enforce_eager", False)
        gpu_memory_utilization = config.get("gpu_memory_utilization", 0.9)
        
        # Default sampling parameters
        self.default_params = {
            "max_tokens": config.get("max_tokens", 1024),
            "temperature": config.get("temperature", 0.7),
            "top_p": config.get("top_p", 0.9),
            "top_k": config.get("top_k", 50),
            "seed": config.get("seed", None)
        }
        
        # Determine if model is multimodal based on name
        self.is_multimodal = any(mm_indicator in self.model_name.lower() 
                                for mm_indicator in ["vl", "vision"])
        
        # Load model
        logger.info(f"Loading model {self.model_name} with vLLM...")
        try:
            self.model = LLM(
                model=self.model_name,
                tensor_parallel_size=tensor_parallel_size,
                trust_remote_code=trust_remote_code,
                enforce_eager=enforce_eager,
                gpu_memory_utilization=gpu_memory_utilization,
                dtype=dtype
            )
            logger.info(f"Model successfully loaded with vLLM")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
    def generate(self, prompts: List[Any], **kwargs) -> List[Dict[str, Any]]:
        """
        Generate responses for the given prompts using vLLM.
        
        Args:
            prompts: List of prompts in message format
            **kwargs: Additional generation parameters to override defaults
            
        Returns:
            List of response dictionaries with fields:
            - text: Generated text
            - usage: Token usage information
            - finish_reason: Reason for finishing generation
        """
        # Format all prompts
        formatted_prompts = []
        for prompt in prompts:
            formatted_prompts.append(self.format_prompt(prompt))
        
        # Prepare sampling parameters
        sampling_params = SamplingParams(
            max_tokens=kwargs.get("max_tokens", self.default_params["max_tokens"]),
            temperature=kwargs.get("temperature", self.default_params["temperature"]),
            top_p=kwargs.get("top_p", self.default_params["top_p"]),
            top_k=kwargs.get("top_k", self.default_params["top_k"]),
            seed=kwargs.get("seed", self.default_params["seed"])
        )
        
        # Perform generation
        try:
            logger.debug(f"Generating responses for {len(formatted_prompts)} prompts")
            outputs = self.model.generate(formatted_prompts, sampling_params)
            
            # Format results
            results = []
            for output in outputs:
                prompt_tokens = len(output.prompt_token_ids)
                completion_tokens = len(output.outputs[0].token_ids)
                
                results.append({
                    "text": output.outputs[0].text,
                    "usage": {
                        "prompt_tokens": prompt_tokens,
                        "completion_tokens": completion_tokens,
                        "total_tokens": prompt_tokens + completion_tokens
                    },
                    "finish_reason": output.outputs[0].finish_reason
                })
            
            return results
            
        except Exception as e:
            logger.error(f"Generation failed: {e}")
            # Return error responses to maintain batch size
            return [{"text": f"Error: {str(e)}", "error": str(e)} for _ in range(len(formatted_prompts))]
    
    def format_prompt(self, messages: List[Dict[str, Any]]) -> str:
        """
        Format conversation messages into a Qwen-compatible prompt string.
        Handles multimodal content from environment observations.
        
        Args:
            messages: List of message dictionaries
            
        Returns:
            Formatted prompt string for Qwen
        """
        formatted_prompt = ""
        
        for message in messages:
            role = message.get("role", "").lower()
            content = message.get("content", "")
            
            # Process multimodal data if present
            if "multi_modal_data" in message and self.is_multimodal:
                content = self._process_multimodal_content(content, message["multi_modal_data"])
            
            # Format according to Qwen chat format
            if role == "system":
                formatted_prompt += f"<|im_start|>system\n{content}<|im_end|>\n"
            elif role == "user":
                formatted_prompt += f"<|im_start|>user\n{content}<|im_end|>\n"
            elif role == "assistant":
                formatted_prompt += f"<|im_start|>assistant\n{content}<|im_end|>\n"
        
        # Add assistant prefix for response generation
        formatted_prompt += "<|im_start|>assistant\n"
        
        return formatted_prompt
    
    def _process_multimodal_content(self, content: str, multi_modal_data: Dict[str, List]) -> str:
        """
        Process multimodal data and insert image tags.
        
        Args:
            content: Original text content with placeholders
            multi_modal_data: Dictionary of multimodal data with keys matching placeholders
            
        Returns:
            Content with image tags inserted
        """
        if not self.is_multimodal:
            return content
        
        processed_content = content
        
        # Process images
        if "<image>" in multi_modal_data:
            images = multi_modal_data["<image>"]
            
            for img in images:
                # Convert image to base64
                if isinstance(img, Image.Image):
                    buffer = io.BytesIO()
                    img.save(buffer, format="PNG")
                    img_b64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
                    img_tag = f"<img>{img_b64}</img>"
                    
                    # Replace the first occurrence of <image> placeholder
                    processed_content = processed_content.replace("<image>", img_tag, 1)
                elif isinstance(img, dict) and "__pil_image__" in img:
                    # Already serialized image
                    img_b64 = img["__pil_image__"]
                    img_tag = f"<img>{img_b64}</img>"
                    processed_content = processed_content.replace("<image>", img_tag, 1)
        
        # Process other modalities if needed
        # (Add support for other modalities like audio if necessary)
        
        return processed_content
    
    def process_images(self, images: List[Image.Image]) -> List[Image.Image]:
        """
        Process images for multimodal models.
        
        Args:
            images: List of PIL Image objects
            
        Returns:
            Processed images ready for the model
        """
        processed_images = []
        for img in images:
            # Ensure image is in RGB mode
            if img.mode != "RGB":
                img = img.convert("RGB")
            processed_images.append(img)
            
        return processed_images
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get detailed information about the model.
        
        Returns:
            Dictionary with model information
        """
        info = super().get_model_info()
        
        # Add vLLM-specific information
        info.update({
            "name": self.model_name,
            "type": "multimodal" if self.is_multimodal else "text",
            "supports_images": self.is_multimodal,
            "max_tokens": self.default_params["max_tokens"],
            "temperature": self.default_params["temperature"],
        })
        
        return info