from setuptools import setup, find_packages

setup(
    name="vagen",
    version="26.2.5",
    description="Reinforcing world model reasoning for multi-turn VLM agents",
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/mll-lab-nu/VAGEN",
    license="Apache-2.0",
    packages=find_packages(exclude=["tests", "tests.*", "vagen.tests", "vagen.tests.*"]),
    # find_packages() alone ships no data files, and `vagen/configs/` is not a package --
    # it holds no .py, so nothing can reach it. envs/registry.py resolves
    # configs/env_registry.yaml relative to __file__ and raises FileNotFoundError when it
    # is absent, so without this every environment dies on a non-editable install. It
    # never showed up because `pip install -e .` maps back to the source tree.
    package_data={"vagen": [
        "configs/*.yaml",
        "configs/*.flags",
        "envs/navigation/assets/*.json",
    ]},
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
        # The per-env servers and every `python -m vagen.envs.<env>.<env>_env` smoke test
        # use it. Nothing else in the stack pulls it in, so those all died with
        # ModuleNotFoundError on a clean machine.
        "fire",
    ],
    extras_require={"test": ["pytest", "pytest-asyncio"]},
    # 3.12, not 3.10: the source parses under 3.10, but verl's installer fetches a
    # cp312-only flash-attn wheel, so an older interpreter fails partway through the
    # install rather than here.
    python_requires=">=3.12",
)
