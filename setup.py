from setuptools import setup, find_packages

setup(
    name="vagen",
    version="26.2.5",
    packages=find_packages(),
    install_requires=[
        "gym-sokoban",
        "gymnasium",
        "gymnasium[toy-text]",
        "uvicorn<0.41",
        # Qwen3-VL. Older versions raise KeyError('qwen3_vl') from AutoConfig, which
        # names the model type and not the reason.
        "transformers>=4.57.0",
        # Not a direct dependency -- it arrives with something else -- but peft's
        # is_torchao_available() *raises* below 0.16 instead of returning False, so a
        # stale copy breaks LoRA even though nothing here quantises. Absent is fine;
        # present and old is not, which is why this is a floor rather than a package
        # anything actually imports.
        "torchao>=0.16.0",
    ],
    python_requires=">=3.10",
)
