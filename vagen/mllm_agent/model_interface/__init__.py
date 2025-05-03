from .vllm import VLLMModelInterface, VLLMModelConfig

REGISTERED_MODEL = {
    "vllm": {
        "model_cls": VLLMModelInterface,
        "config_cls": VLLMModelConfig,
    },
}