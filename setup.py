from setuptools import setup, find_packages

setup(
    name="vagen",
    version="26.8.14",
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
        # 5.2 is where `qwen3_5` lands; below it AutoConfig raises KeyError('qwen3_5'),
        # and below 4.57 it raises the same for 'qwen3_vl' -- naming the model type and
        # not the reason. The engine extras pin this exactly; the floor is for anyone
        # installing VAGEN without one.
        #
        # [kernels] is how flash attention gets here at all. verl asks for
        # attn_implementation="flash_attention_2" -- hardcoded, in the critic's value-head
        # path -- and the flash-attn package has no wheel for this torch: upstream builds
        # up to torch 2.9 and we are on 2.11, so installing it means a source build.
        # transformers avoids that by pulling a prebuilt kernels-community/flash-attn2
        # from the Hub, but only when `kernels` is importable; otherwise it raises
        # "FlashAttention2 has been toggled on, but ... doesn't seem to be installed"
        # and blames the missing package rather than the missing fallback. Written as an
        # extra so the version range stays transformers' to declare -- it is currently
        # kernels<0.13,>=0.12.0, and a plain `pip install kernels` picks up 0.16, whose
        # get_kernel() requires an explicit revision and fails the fallback.
        "transformers[kernels]>=5.2.0",
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
        # verl's critic loads its value head from trl, so every GAE estimator needs it.
        # verl installs with --no-deps, so it has to be named here or the run dies at
        # critic construction with "is not a value head model, please install trl".
        #
        # Both bounds are load-bearing:
        #   >=0.27  below it, trl's __init__ monkey-patches vllm behind a bare
        #           `if is_vllm_available()`, importing vllm.transformers_utils.tokenizer,
        #           which vllm moved to vllm/tokenizers/hf.py. `import trl` then raises
        #           ModuleNotFoundError just because vllm is installed.
        #   <0.29   0.29 drops the top-level AutoModelForCausalLMWithValueHead, which
        #           verl's utils/model.py imports unqualified. Its monkey_patch.py already
        #           prefers trl.experimental.ppo; utils/model.py was not updated to match,
        #           so raising this ceiling means patching verl too.
        "trl>=0.27,<0.29",
    ],
    # ------------------------------------------------------------- the two rollout engines
    #
    # Pick exactly one:  pip install -e ".[vllm]"   or   pip install -e ".[sglang]"
    #
    # They are mutually exclusive, and not by preference -- every (vllm, sglang) pair in
    # this torch tier pins a different flashinfer patch version, so pip refuses the two
    # together:
    #     vllm   0.22.0  -> flashinfer 0.6.11.post2
    #     sglang 0.5.15  -> flashinfer 0.6.12
    # verl models them the same way, as separate extras. Installing both into one
    # environment is not a supported configuration; use two environments.
    #
    # torch and transformers are pinned identically on both sides so the two environments
    # differ only in the engine, and transformers 5.12.1 is what gives Qwen3.5.
    extras_require={
        "test": ["pytest", "pytest-asyncio"],
        # The verified default: the whole VAGEN suite and the sokoban concat / no_concat /
        # compact scripts run on this.
        "vllm": [
            "torch==2.11.0",
            "vllm==0.22.0",          # verl main needs >=0.18.0 for vllm.entrypoints.openai.parser
            "transformers[kernels]==5.12.1",
        ],
        # sglang 0.5.15, not the 0.5.8 verl main declares: 0.5.8 pins torch==2.9.1, which
        # would make the two extras disagree on torch for no benefit. 0.5.15 still exports
        # `ContinueGenerationReqInput`, the symbol verl's sglang rollout imports and the
        # one whose absence broke sglang on the old pin.
        "sglang": [
            "torch==2.11.0",
            "sglang[srt,openai]==0.5.15",
            "transformers[kernels]==5.12.1",
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
