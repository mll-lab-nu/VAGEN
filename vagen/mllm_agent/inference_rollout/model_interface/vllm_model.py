from typing import List, Dict, Union, Optional, Any
import os
import torch
from PIL import Image
import numpy as np

from vagen.mllm_agent.inference_rollout.model_interface.base_model import BaseModelInterface

class VLLMModelInterface(BaseModelInterface):
    """
    Simplified model interface for local models using vLLM.
    Focuses on the core functionality needed for inference.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the vLLM model interface with configuration.

        Args:
            config: Configuration dictionary with model parameters
        """
        super().__init__(config)
        
        # 初始化 _is_multimodal 属性为 False，稍后会更新
        self._is_multimodal = False
        
        # 设置模型路径
        self.model_path = self.config.get("path")
        if not self.model_path:
            raise ValueError("Model path is required")
        
        # 先检查模型是否是多模态的
        self._is_multimodal = self._check_if_multimodal()
        
        # 初始化 tokenizer 和 engine
        self._init_tokenizer()
        self._init_engine()
        
        # 如果模型是多模态的，初始化 processor
        if self._is_multimodal:
            self._init_processor()
        
        print(f"Initialized vLLM model: {self.model_path}")

    def _init_tokenizer(self):
        """Initialize the tokenizer from the model path."""
        from transformers import AutoTokenizer
        
        tokenizer_path = self.config.get("tokenizer_path") or self.model_path
        
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                tokenizer_path,
                trust_remote_code=self.config.get("trust_remote_code", True),
                use_fast=True
            )
            
            # Set padding token if not set
            if self.tokenizer.pad_token is None:
                if self.tokenizer.eos_token is not None:
                    self.tokenizer.pad_token = self.tokenizer.eos_token
                else:
                    self.tokenizer.pad_token = self.tokenizer.eos_token = "</s>"
                    
        except Exception as e:
            raise RuntimeError(f"Failed to initialize tokenizer: {e}")

    def _init_engine(self):
        """Initialize the vLLM engine."""
        try:
            from vllm import LLM, SamplingParams
        except ImportError:
            raise ImportError("vLLM is not installed. Please install it with 'pip install vllm'.")
        
        # Extract vLLM parameters
        vllm_kwargs = {
            "model": self.model_path,
            "tensor_parallel_size": self.config.get("tensor_parallel_size", 1),
            "trust_remote_code": self.config.get("trust_remote_code", True),
            "gpu_memory_utilization": self.config.get("gpu_memory_utilization", 0.9),
        }
        
        # Handle dtype
        dtype = self.config.get("dtype", "auto")
        if dtype != "auto":
            vllm_kwargs["dtype"] = dtype
            
        # Handle max_model_len
        max_model_len = self.config.get("max_model_len")
        if max_model_len is not None:
            vllm_kwargs["max_model_len"] = max_model_len
            
        # Initialize engine
        try:
            self.engine = LLM(**vllm_kwargs)
        except Exception as e:
            raise RuntimeError(f"Failed to initialize vLLM engine: {e}")

    def _check_if_multimodal(self):
        """
        Check if model is multimodal based on model name.
        This avoids accessing engine.model attributes which may not exist.
        """
        model_name = self.model_path.lower()
        multimodal_keywords = [
            "llava", "qwenvl", "qwen2vl", "qwen-vl", "qwen2-vl",
            "idefics", "fuyu", "persimmon-vl", "gemini-pro-vision",
            "flamingo", "blip", "instructblip", "cogvlm"
        ]
        
        return any(keyword in model_name for keyword in multimodal_keywords)

    def _init_processor(self):
        """Initialize processor for multimodal models if needed."""
        try:
            from transformers import AutoProcessor
            
            processor_path = self.config.get("processor_path") or self.model_path
            
            self.processor = AutoProcessor.from_pretrained(
                processor_path,
                trust_remote_code=self.config.get("trust_remote_code", True)
            )
        except Exception as e:
            print(f"Warning: Failed to initialize processor: {e}")
            self.processor = None
            # 如果无法初始化 processor，将多模态标志设为 False
            self._is_multimodal = False

    def generate(self, prompts: List[Any], **kwargs) -> List[Dict[str, Any]]:
        """
        Generate responses for the given prompts.

        Args:
            prompts: List of prompts (strings or dicts with messages/images)
            **kwargs: Additional generation parameters

        Returns:
            List of response dictionaries
        """
        # Prepare generation parameters
        from vllm import SamplingParams
        
        # Start with config defaults
        params = {
            "temperature": self.config.get("temperature", 0.7),
            "top_p": self.config.get("top_p", 0.95),
            "max_tokens": self.config.get("max_tokens", 256),
        }
        
        # Add optional parameters if in config
        for param in ["top_k", "presence_penalty", "frequency_penalty", "stop"]:
            if param in self.config:
                params[param] = self.config[param]
                
        # Override with kwargs
        params.update(kwargs)
        
        # Create SamplingParams
        sampling_params = SamplingParams(**params)
        
        # Process prompts based on type
        processed_prompts = []
        
        for prompt in prompts:
            if isinstance(prompt, str):
                # Text-only prompt
                processed_prompts.append(prompt)
            elif isinstance(prompt, dict) and "messages" in prompt:
                # Messages with possible image data
                if self.is_multimodal and "images" in prompt and prompt["images"]:
                    # For multimodal prompts, we need model-specific processing
                    # This is simplified - you may need to adapt to specific models
                    processed_prompts.append(self.format_prompt(prompt["messages"]))
                else:
                    # Text-only messages
                    processed_prompts.append(self.format_prompt(prompt["messages"]))
            else:
                # Default to string conversion
                processed_prompts.append(str(prompt))
                
        # Generate with vLLM
        try:
            outputs = self.engine.generate(processed_prompts, sampling_params=sampling_params)
        except Exception as e:
            raise RuntimeError(f"vLLM generation failed: {e}")
            
        # Format results
        results = []
        for output in outputs:
            result = {
                "text": output.outputs[0].text,
                "finish_reason": output.outputs[0].finish_reason or "stop",
                "usage": {
                    "prompt_tokens": len(output.prompt_token_ids),
                    "completion_tokens": len(output.outputs[0].token_ids),
                    "total_tokens": len(output.prompt_token_ids) + len(output.outputs[0].token_ids)
                }
            }
            results.append(result)
            
        return results

    def format_prompt(self, messages: List[Dict[str, Any]]) -> str:
        """
        Format conversation messages into a prompt string.

        Args:
            messages: List of message dictionaries

        Returns:
            Formatted prompt string
        """
        # Use tokenizer's chat template if available
        if hasattr(self.tokenizer, "apply_chat_template"):
            try:
                # Clean messages for text-only formatting
                clean_messages = []
                for msg in messages:
                    clean_msg = {
                        "role": msg["role"],
                        "content": msg["content"]
                    }
                    clean_messages.append(clean_msg)
                    
                prompt = self.tokenizer.apply_chat_template(
                    clean_messages,
                    add_generation_prompt=True,
                    tokenize=False
                )
                return prompt
            except Exception as e:
                print(f"Warning: Failed to apply chat template: {e}")
                
        # Fallback to manual formatting
        formatted_prompt = ""
        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            
            if role == "system":
                formatted_prompt += f"System: {content}\n\n"
            elif role == "user":
                formatted_prompt += f"User: {content}\n\n"
            elif role == "assistant":
                formatted_prompt += f"Assistant: {content}\n\n"
                
        formatted_prompt += "Assistant: "
        return formatted_prompt

    def process_images(self, images: List[Any]) -> List[Any]:
        """
        Process images for multi-modal models.

        Args:
            images: List of image data

        Returns:
            Processed image data
        """
        if not self.is_multimodal or not hasattr(self, 'processor') or self.processor is None:
            return []
            
        processed_images = []
        
        for image in images:
            # Convert to PIL Image if needed
            pil_image = self._ensure_pil_image(image)
            if pil_image:
                try:
                    processed = self.processor.image_processor(pil_image, return_tensors="pt")
                    processed_images.append(processed)
                except Exception as e:
                    print(f"Warning: Failed to process image: {e}")
                    
        return processed_images

    def _ensure_pil_image(self, image: Any) -> Optional[Image.Image]:
        """Convert various image formats to PIL Image."""
        if isinstance(image, Image.Image):
            return image
            
        try:
            # Handle bytes
            if isinstance(image, bytes):
                import io
                return Image.open(io.BytesIO(image))
                
            # Handle numpy array
            if isinstance(image, np.ndarray):
                return Image.fromarray(image)
                
            # Handle file path
            if isinstance(image, str) and os.path.isfile(image):
                return Image.open(image)
                
            # Handle base64
            if isinstance(image, str) and image.startswith(('data:image', 'base64:')):
                import base64
                import re
                
                base64_data = re.sub(r'^data:image/\w+;base64,', '', image)
                image_data = base64.b64decode(base64_data)
                return Image.open(io.BytesIO(image_data))
                
        except Exception as e:
            print(f"Warning: Failed to convert image: {e}")
            
        return None

    @property
    def is_multimodal(self) -> bool:
        """Whether the model supports images."""
        return self._is_multimodal
        
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the model.
        
        Returns:
            Dictionary with model information
        """
        return {
            "name": self.config.get("name", os.path.basename(self.model_path)),
            "type": "local",
            "is_multimodal": self.is_multimodal,
            "context_length": self.config.get("max_model_len", 4096),
            "token_limit": self.config.get("max_tokens", 256)
        }