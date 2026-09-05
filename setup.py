from setuptools import setup, find_packages

setup(
    name="vagen",
    version="26.8.14",
    description="Reinforcing world model reasoning for multi-turn VLM agents",
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/mll-lab-nu/VAGEN",
    license="MIT",
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
        # The per-environment extras the docs tell you to install. Absent from a
        # non-editable install, the README points at a file that is not there.
        "envs/*/requirements.txt",
    ]},
    install_requires=[
        # vLLM compiles kernels through ninja at engine startup even with --enforce-eager,
        # and neither engine extra pulls it in. Missing, the server dies with a bare
        # FileNotFoundError several frames below anything that mentions vLLM.
        "ninja",
        "gym-sokoban",
        "gymnasium",
        "gymnasium[toy-text]",
        "uvicorn<0.41",
        # The per-env servers and every `python -m vagen.envs.<env>.<env>_env` smoke test
        # use it. Nothing else in the stack pulls it in, so those all died with
        # ModuleNotFoundError on a clean machine.
        "fire",
    ],
    # ------------------------------------------------------------- the two rollout engines
    #
    # Pick exactly one. scripts/install.sh is the supported entry point because the
    # SGLang stack needs an ordered second pass that package metadata cannot express.
    #
    # They are mutually exclusive, and not by preference -- every (vllm, sglang) pair in
    # this torch tier pins a different flashinfer patch version, so pip refuses the two
    # together:
    #     vllm   0.22.0  -> flashinfer 0.6.11.post2
    #     sglang 0.5.13  -> flashinfer 0.6.12
    # verl models them the same way, as separate extras. Installing both into one
    # environment is not a supported configuration; use two environments.
    #
    extras_require={
        "test": ["pytest", "pytest-asyncio"],
        # Supported alternative for launchers that explicitly select vLLM.
        "vllm": [
            "torch==2.11.0",
            "vllm==0.22.0",          # verl main needs >=0.18.0 for vllm.entrypoints.openai.parser
            "transformers[kernels]==5.12.1",
            "torchao>=0.16.0",
            "trl>=0.27,<0.29",
        ],
        # Metadata for the default stack. Use scripts/install.sh rather than installing
        # this extra directly: causal-conv1d and trl require the ordered second pass in
        # scripts/install_sglang.sh.
        "sglang": [
            "torch==2.11.0",
            "sglang==0.5.13",
            "transformers==5.8.1",
            "flashinfer-python[cu13]==0.6.12",
            "fla-core[cuda]==0.5.2",
            "flash-linear-attention==0.5.2",
            "causal-conv1d==1.7.0",
            "peft==0.19.1",
        ],
    },
    # 3.12, not 3.10: the source parses under 3.10, but verl's installer fetches a
    # cp312-only flash-attn wheel, so an older interpreter fails partway through the
    # install rather than here.
    # ★ Exactly 3.12, matching scripts/install.sh, which refuses anything else. They
    # disagreed: setup.py said >=3.12 while the installer rejected 3.13, so following the
    # documented prerequisite and then the documented install command failed.
    python_requires=">=3.12,<3.13",
)
