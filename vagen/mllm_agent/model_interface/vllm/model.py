import base64
import io
import logging
from typing import List, Dict, Any
from PIL import Image

from vllm import LLM, SamplingParams

from vagen.mllm_agent.model_interface.base_model import BaseModelInterface
from .model_config import VLLMModelConfig

logger = logging.getLogger(__name__)

class VLLMModelInterface(BaseModelInterface):
    """
    Model interface for local inference using vLLM.
    Specifically designed for Qwen models with support for both text and multimodal inputs.
    """
    
    def __init__(self, config: VLLMModelConfig):
        """
        Initialize the vLLM model interface.
        
        Args:
            config: VLLMModelConfig instance
        """
        # Convert config to dict for base class
        super().__init__(config)
        
        self.config = config
        self.model_name = config.model_name
        
        # Determine if model is multimodal based on name
        self.is_multimodal = any(mm_indicator in self.model_name.lower() 
                                for mm_indicator in ["vl", "vision", "vlm"])
        
        # Base model parameters (work for all models)
        model_kwargs = {
            "model": self.model_name,
            "tensor_parallel_size": config.tensor_parallel_size,
            "trust_remote_code": config.trust_remote_code,
            "enforce_eager": config.enforce_eager,
            "gpu_memory_utilization": config.gpu_memory_utilization,
            "dtype": config.dtype
        }
        
        # If this is a multimodal model, try to handle according to vLLM VLM documentation
        if self.is_multimodal:
            logger.info(f"Detected multimodal model: {self.model_name}. Using VLM parameters.")
            
            # Create engine_args dict separately for VLM models
            # Note: using engine_args dict directly instead of adding to model_kwargs
            from vllm.engine.arg_utils import EngineArgs
            
            # Default engine arguments
            engine_args_dict = {
                "model": self.model_name,
                "tensor_parallel_size": config.tensor_parallel_size,
                "trust_remote_code": config.trust_remote_code,
                "enforce_eager": config.enforce_eager,
                "gpu_memory_utilization": config.gpu_memory_utilization,
                "dtype": config.dtype
            }
            
            # Add additional args for VLM if provided in config
            extra_vlm_args = {}
            if config.image_input_type:
                extra_vlm_args["image_input_type"] = config.image_input_type
            if config.image_token_id is not None:
                extra_vlm_args["image_token_id"] = config.image_token_id
            if config.image_input_shape:
                extra_vlm_args["image_input_shape"] = config.image_input_shape
            if config.image_feature_size:
                extra_vlm_args["image_feature_size"] = config.image_feature_size
            
            try:
                # Try to create engine args with VLM-specific parameters
                vlm_engine_args = EngineArgs(**engine_args_dict, **extra_vlm_args)
                model_kwargs["engine_args"] = vlm_engine_args
                logger.info("Successfully created VLM engine args with multimodal parameters")
            except TypeError as e:
                logger.warning(f"Failed to create VLM engine args: {e}")
                logger.info("Trying direct initialization approach...")
                
                # Fallback to direct parameter passing method
                try:
                    # For newer vLLM versions that directly support VLM parameters
                    model_kwargs.update(extra_vlm_args)
                    logger.info("Using direct VLM parameters instead of engine_args")
                except Exception as e:
                    logger.error(f"Both VLM initialization methods failed. Running in text-only mode: {e}")
                    # Remove all VLM-specific parameters to ensure text mode works
                    for key in ["image_input_type", "image_token_id", "image_input_shape", "image_feature_size"]:
                        if key in model_kwargs:
                            del model_kwargs[key]
                    # Mark as non-multimodal to prevent VLM-specific processing
                    self.is_multimodal = False
        
        # Load model
        logger.info(f"Loading {'multimodal' if self.is_multimodal else 'text'} model {self.model_name} with vLLM...")
        try:
            self.model = LLM(**model_kwargs)
            logger.info(f"Model successfully loaded with vLLM")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            # If using engine_args failed, try without engine_args
            if 'engine_args' in model_kwargs:
                logger.info("Retrying without engine_args")
                del model_kwargs['engine_args']
                
                # Try direct parameter approach again
                try:
                    self.model = LLM(**model_kwargs)
                    logger.info("Model loaded successfully with direct parameters")
                except Exception as e2:
                    logger.error(f"Failed with direct parameters: {e2}. Trying text-only mode.")
                    # Remove all VLM parameters
                    for key in ["image_input_type", "image_token_id", "image_input_shape", "image_feature_size"]:
                        if key in model_kwargs:
                            del model_kwargs[key]
                    
                    try:
                        self.model = LLM(**model_kwargs)
                        logger.info("Model loaded successfully in text-only mode")
                        self.is_multimodal = False
                    except Exception as e3:
                        logger.error(f"All loading attempts failed: {e3}")
                        raise
            else:
                # No engine_args, so this is already a direct attempt
                # Try again without multimodal parameters
                for key in ["image_input_type", "image_token_id", "image_input_shape", "image_feature_size"]:
                    if key in model_kwargs:
                        del model_kwargs[key]
                
                try:
                    self.model = LLM(**model_kwargs)
                    logger.info("Model loaded successfully in text-only mode")
                    self.is_multimodal = False
                except Exception as e2:
                    logger.error(f"Failed to load model even in text-only mode: {e2}")
                    raise
    
    def _check_vllm_multimodal_support(self) -> bool:
        """
        Check if the installed vLLM version supports multimodal parameters.
        
        Returns:
            True if vLLM supports multimodal, False otherwise
        """
        try:
            from vllm.engine.arg_utils import EngineArgs
            
            # Check if EngineArgs supports multimodal parameters
            # Create a test instance to see if image_input_type is a valid parameter
            supported_params = set(EngineArgs.__init__.__code__.co_varnames)
            
            # If image_input_type is in the parameters, multimodal is supported
            return "image_input_type" in supported_params
        except (ImportError, AttributeError):
            logger.warning("Could not inspect vLLM's EngineArgs class, assuming no multimodal support")
            return False
    
    def generate(self, prompts: List[Any], **kwargs) -> List[Dict[str, Any]]:
        """
        Generate responses for the given prompts using vLLM.
        
        Args:
            prompts: List of prompts which can be:
                     - List of message dicts for text-only  
                     - Message dicts with 'multi_modal_data' containing images
            **kwargs: Additional generation parameters to override defaults
            
        Returns:
            List of response dictionaries
        """
        # Process prompts and extract images if present
        formatted_prompts = []
        image_inputs = []
        
        for prompt in prompts:
            # Check if prompt is a list of messages with multimodal data
            has_multimodal = False
            images = []
            
            # Extract images from messages if they have multi_modal_data
            if isinstance(prompt, list):
                for message in prompt:
                    if "multi_modal_data" in message:
                        # Get image placeholder (default is "<image>")
                        for key, values in message["multi_modal_data"].items():
                            if key == "<image>" or "image" in key.lower():
                                images.extend(values)
                                has_multimodal = True
            
            # Format prompt text
            formatted_text = self.format_prompt(prompt)
            formatted_prompts.append(formatted_text)
            
            # Add images for vLLM multimodal
            if self.is_multimodal and has_multimodal:
                # Process images for vLLM
                processed_images = self.process_images(images)
                image_inputs.append(processed_images)
            else:
                image_inputs.append(None)
        
        # Prepare sampling parameters
        sampling_params = SamplingParams(
            max_tokens=kwargs.get("max_tokens", self.config.max_tokens),
            temperature=kwargs.get("temperature", self.config.temperature),
            top_p=kwargs.get("top_p", self.config.top_p),
            top_k=kwargs.get("top_k", self.config.top_k),
            seed=kwargs.get("seed", self.config.seed)
        )
        
        # Perform generation
        try:
            if self.is_multimodal and any(img is not None for img in image_inputs):
                # For multimodal generation with vLLM
                logger.debug(f"Generating multimodal responses for {len(formatted_prompts)} prompts")
                try:
                    outputs = self.model.generate(
                        prompts=formatted_prompts,
                        sampling_params=sampling_params,
                        multi_modal_data={"image": image_inputs}  # vLLM expects this format
                    )
                except Exception as e:
                    logger.warning(f"Multimodal generation failed: {e}. Falling back to text-only generation.")
                    # Fallback to text-only
                    outputs = self.model.generate(formatted_prompts, sampling_params)
            else:
                # Text-only generation
                logger.debug(f"Generating text responses for {len(formatted_prompts)} prompts")
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
        
        For Qwen-VL models in vLLM, follow the same format as training:
        - <|vision_start|><|image_pad|><|vision_end|> for images
        - Keep the original message structure
        
        Args:
            messages: List of message dictionaries
            
        Returns:
            Formatted prompt string for Qwen
        """
        formatted_prompt = ""
        
        for message in messages:
            role = message.get("role", "").lower()
            content = message.get("content", "")
            
            # Handle multimodal data like training rollout
            if "multi_modal_data" in message and self.is_multimodal:
                # Replace <image> placeholders with vLLM format
                # Training uses: <|vision_start|><|image_pad|><|vision_end|>
                for key, values in message["multi_modal_data"].items():
                    if key == "<image>" or "image" in key.lower():
                        # Count images and replace placeholders
                        image_count = len(values)
                        for _ in range(image_count):
                            content = content.replace(
                                "<image>", 
                                "<|vision_start|><|image_pad|><|vision_end|>", 
                                1
                            )
            
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
    
    def process_images(self, images: List[Any]) -> List[Any]:
        """
        Process images for multimodal models.
        Aligns with training rollout's image processing.
        
        Args:
            images: List of images (PIL Images or serialized)
            
        Returns:
            Processed images ready for vLLM
        """
        processed_images = []
        
        for img in images:
            # Handle different image formats (same as training)
            if isinstance(img, Image.Image):
                # Ensure image is in RGB mode
                if img.mode != "RGB":
                    img = img.convert("RGB")
                processed_images.append(img)
            elif isinstance(img, dict) and "__pil_image__" in img:
                # Handle serialized images from service
                from vagen.server.serial import deserialize_pil_image
                deserialized_img = deserialize_pil_image(img)
                if deserialized_img.mode != "RGB":
                    deserialized_img = deserialized_img.convert("RGB")
                processed_images.append(deserialized_img)
            else:
                # If it's already processed, just append
                processed_images.append(img)
        
        return processed_images
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get detailed information about the model.
        
        Returns:
            Dictionary with model information
        """
        info = super().get_model_info()
        
        info.update({
            "name": self.model_name,
            "type": "multimodal" if self.is_multimodal else "text",
            "supports_images": self.is_multimodal,
            "max_tokens": self.config.max_tokens,
            "temperature": self.config.temperature,
            "config_id": self.config.config_id()
        })
        
        return info