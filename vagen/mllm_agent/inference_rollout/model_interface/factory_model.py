from typing import Dict, Any, Optional

from vagen.mllm_agent.inference_rollout.model_interface.base_model import BaseModelInterface

def create_model_interface(config: Dict[str, Any]) -> BaseModelInterface:
    """
    Factory function to create the appropriate model interface based on configuration.

    Args:
        config: Model configuration dictionary

    Returns:
        Initialized model interface instance

    Raises:
        ValueError: If model type is unknown or required config is missing
    """
    # Validate required config
    if "type" not in config:
        raise ValueError("Model configuration must include 'type' field")

    model_type = config["type"].lower()

    # Create appropriate model interface based on type
    if model_type == "vllm":
        from vagen.mllm_agent.inference_rollout.model_interface.vllm_model import VLLMModelInterface
        return VLLMModelInterface(config)

    elif model_type == "openai":
        from vagen.mllm_agent.inference_rollout.model_interface.openai_model import OpenAIModelInterface
        return OpenAIModelInterface(config)

    elif model_type == "claude":
        from vagen.mllm_agent.inference_rollout.model_interface.claude_model import ClaudeModelInterface
        return ClaudeModelInterface(config)

    elif model_type == "gemini":
        from vagen.mllm_agent.inference_rollout.model_interface.gemini_model import GeminiModelInterface
        return GeminiModelInterface(config)

    else:
        raise ValueError(f"Unknown model type: {model_type}")

def get_available_model_types() -> Dict[str, str]:
    """
    Get a dictionary of available model types and their descriptions.

    Returns:
        Dictionary mapping model types to descriptions
    """
    return {
        "vllm": "Local models using vLLM for efficient inference",
        "openai": "OpenAI API models (GPT-3.5, GPT-4, etc.)",
        "claude": "Anthropic Claude API models",
        "gemini": "Google Gemini API models"
    }

def get_model_type_requirements(model_type: str) -> Dict[str, Any]:
    """
    Get the required configuration fields for a specific model type.

    Args:
        model_type: Type of model interface

    Returns:
        Dictionary of required configuration fields and their descriptions

    Raises:
        ValueError: If model type is unknown
    """
    model_type = model_type.lower()

    if model_type == "vllm":
        return {
            "path": "Path to the model directory or Hugging Face model ID",
            "tokenizer_path": "Optional path to tokenizer (defaults to model path)",
            "tensor_parallel_size": "Number of GPUs for tensor parallelism (default: 1)",
            "dtype": "Model precision: float16, bfloat16, or float32 (default: auto)",
            "max_model_len": "Maximum sequence length (default: model's max_position_embeddings)"
        }

    elif model_type == "openai":
        return {
            "api_key": "OpenAI API key",
            "model": "Model name (e.g., gpt-4-turbo, gpt-3.5-turbo)",
            "api_base": "Optional API base URL for non-standard endpoints",
            "organization": "Optional organization ID"
        }

    elif model_type == "claude":
        return {
            "api_key": "Anthropic API key",
            "model": "Model name (e.g., claude-3-opus, claude-3-sonnet)",
            "api_base": "Optional API base URL for non-standard endpoints"
        }

    elif model_type == "gemini":
        return {
            "api_key": "Google API key",
            "model": "Model name (e.g., gemini-pro, gemini-pro-vision)",
            "project_id": "Optional Google Cloud project ID"
        }

    else:
        raise ValueError(f"Unknown model type: {model_type}")

def create_default_config(model_type: str) -> Dict[str, Any]:
    """
    Create a default configuration for a model type.

    Args:
        model_type: Type of model interface

    Returns:
        Default configuration dictionary with placeholder values

    Raises:
        ValueError: If model type is unknown
    """
    model_type = model_type.lower()

    if model_type == "vllm":
        return {
            "type": "vllm",
            "name": "Local vLLM Model",
            "path": "your/model/path/or/hf/model_id",
            "tokenizer_path": None,  # Will default to model path
            "tensor_parallel_size": 1,
            "dtype": "auto",
            "max_model_len": None,  # Will use model default
            "gpu_memory_utilization": 0.9,
            "trust_remote_code": True,
            "temperature": 0.7,
            "top_p": 0.95,
            "max_tokens": 256
        }

    elif model_type == "openai":
        return {
            "type": "openai",
            "name": "OpenAI API",
            "api_key": "YOUR_API_KEY_HERE",
            "model": "gpt-4-turbo",
            "api_base": None,  # Will use default
            "organization": None,  # Will use default
            "temperature": 0.7,
            "top_p": 0.95,
            "max_tokens": 256,
            "timeout": 30,
            "retry_count": 3
        }

    elif model_type == "claude":
        return {
            "type": "claude",
            "name": "Anthropic Claude API",
            "api_key": "YOUR_API_KEY_HERE",
            "model": "claude-3-opus-20240229",
            "api_base": None,  # Will use default
            "temperature": 0.7,
            "top_p": 0.95,
            "max_tokens": 1024,
            "timeout": 30,
            "retry_count": 3
        }

    elif model_type == "gemini":
        return {
            "type": "gemini",
            "name": "Google Gemini API",
            "api_key": "YOUR_API_KEY_HERE",
            "model": "gemini-pro",
            "project_id": None,  # Will use default
            "temperature": 0.7,
            "top_p": 0.95,
            "max_tokens": 256,
            "timeout": 30,
            "retry_count": 3
        }

    else:
        raise ValueError(f"Unknown model type: {model_type}")