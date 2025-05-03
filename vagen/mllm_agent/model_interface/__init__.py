from .vllm import VLLMModelInterface, VLLMModelConfig
from .openai import OpenAIModelInterface, OpenAIModelConfig

REGISTERED_MODEL = {
    "vllm": {
        "model_cls": VLLMModelInterface,
        "config_cls": VLLMModelConfig,
    },
    "openai": {
        "model_cls": OpenAIModelInterface,
        "config_cls": OpenAIModelConfig
    }
}