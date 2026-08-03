"""Shared base for VAGEN's gym agent loops.

verl 0.8 constructs an agent loop per rollout via hydra and dropped the ``init_class``
classmethod that used to carry one-off setup. Anything expensive therefore has to be
cached explicitly rather than parked on the class, which is what this module provides:
``__init__`` only reads config, and the two costs worth avoiding per rollout -- importing
an environment module and rendering a chat template -- go through module-level caches.

The caches hold a reference to the key object so its ``id`` cannot be reused by a later
allocation after a garbage collection.
"""

from __future__ import annotations

import importlib
from typing import Any

from verl.experimental.agent_loop.agent_loop import AgentLoopBase

# env name -> class, shared by every agent loop instance in this worker process.
_ENV_CLASS_CACHE: dict[str, type] = {}

# (id(processing_class), chat-template kwargs) -> (processing_class, prefix token ids)
_SYSTEM_PREFIX_CACHE: dict[tuple[int, str], tuple[Any, list[int]]] = {}


class VagenGymAgentLoopBase(AgentLoopBase):
    """Config plumbing common to the concat and no-concat gym loops."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Module paths only; the classes themselves are imported on first use, so a
        # worker never pays for environments this run does not touch.
        self.env_registry_paths = dict(self.config.env_registry.items())
        self.prompt_length = self.rollout_config.prompt_length
        self.response_length = self.rollout_config.response_length

    def resolve_env_class(self, env_name: str) -> type:
        """Import and cache the environment class registered under ``env_name``."""
        if env_name not in _ENV_CLASS_CACHE:
            if env_name not in self.env_registry_paths:
                raise KeyError(f"Unknown env: {env_name}. Available: {list(self.env_registry_paths.keys())}")
            module_path, class_name = self.env_registry_paths[env_name].rsplit(".", 1)
            _ENV_CLASS_CACHE[env_name] = getattr(importlib.import_module(module_path), class_name)
        return _ENV_CLASS_CACHE[env_name]

    def system_prompt_prefix_ids(self) -> list[int]:
        """Token ids a system message contributes before its own content.

        Used to strip the system turn back off a rendered prompt. Distinct from the
        base class's ``system_prompt``, which is the prefix a template inserts on its
        own; this one is the wrapper around an explicit system-role message.
        """
        processing_class = self.processor if self.processor is not None else self.tokenizer
        key = (id(processing_class), repr(sorted(self.apply_chat_template_kwargs.items())))
        if key not in _SYSTEM_PREFIX_CACHE:
            _SYSTEM_PREFIX_CACHE[key] = (processing_class, self._render_system_prompt_prefix())
        return _SYSTEM_PREFIX_CACHE[key][1]

    def _render_system_prompt_prefix(self) -> list[int]:
        placeholder = [{"role": "system", "content": "placeholder"}]
        if self.processor is not None:
            text = self.processor.apply_chat_template(
                placeholder, add_generation_prompt=False, tokenize=False, **self.apply_chat_template_kwargs
            )
            return self.processor(text=[text], return_tensors="pt")["input_ids"].squeeze(0).tolist()
        return self.tokenizer.apply_chat_template(
            placeholder,
            add_generation_prompt=False,
            tokenize=True,
            return_dict=False,
            **self.apply_chat_template_kwargs,
        )
