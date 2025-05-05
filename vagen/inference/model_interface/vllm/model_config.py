# vagen/mllm_agent/model_interface/vllm/model_config.py

from dataclasses import dataclass, field, fields
from typing import Optional, Dict, Any, List

from vagen.inference.model_interface.base_model_config import BaseModelConfig

@dataclass
class VLLMModelConfig(BaseModelConfig):
    """Configuration specific to vLLM model interface."""
    
    # vLLM specific parameters
    tensor_parallel_size: int = 1
    gpu_memory_utilization: float = 0.9
    dtype: str = "bfloat16"
    trust_remote_code: bool = True  # Required for Qwen models
    enforce_eager: bool = False
    top_p: float = 0.9
    top_k: int = 50
    
    # VLM specific settings
    image_input_type: str = "pixel_values"  # For Qwen-VL
    image_token_id: Optional[int] = None
    image_input_shape: Optional[str] = None  # e.g., "1,3,224,224"
    image_feature_size: Optional[int] = None
    
    # Special tokens that might be needed
    additional_special_tokens: Optional[List[str]] = field(default_factory=lambda: ["<think>", "</think>", "<answer>", "</answer>"])
    tokenizer_mode: str = "auto"  # Can be "auto", "slow", or "mistral"
    
    # interface specific parameter
    model_name: str = field(default="Qwen/Qwen2.5-0.5B-Instruct")
    provider: str = "vllm"
    
    def config_id(self) -> str:
        """Generate a unique identifier for this configuration."""
        id_fields = ["model_name", "max_tokens", "temperature", 
                     "tensor_parallel_size", "gpu_memory_utilization", "dtype"]
        id_str = ",".join([f"{field_name}={getattr(self, field_name)}" 
                          for field_name in id_fields 
                          if field_name in [f.name for f in fields(self)]])
        return f"VLLMModelConfig({id_str})"
    
    @staticmethod
    def get_provider_info() -> Dict[str, Any]:
        """Get information about the vLLM provider."""
        return {
            "description": "Local model inference using vLLM",
            "supports_multimodal": True,
            "supported_models": [
                "Qwen/Qwen2.5-0.5B-Instruct",
                "Qwen/Qwen2.5-VL-3B-Instruct",
                "Qwen/Qwen2.5-VL-7B-Instruct",
                "omlab/VLM-R1-Qwen2.5VL-3B-OVD-0321",
                "omlab/VLM-R1-Qwen2.5VL-3B-Math-0305",
                "omlab/Qwen2.5VL-3B-VLM-R1-REC-500steps"
            ],
            "default_model": "Qwen/Qwen2.5-0.5B-Instruct"
        }
        
if __name__ == "__main__":
    config = VLLMModelConfig()
    print(config)