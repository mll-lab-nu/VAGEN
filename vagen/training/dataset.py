# -*- coding: utf-8 -*-
# Training dataset construction; code comments are kept in English.

import hashlib
import random
from typing import List, Optional, Sequence

from torch.utils.data import Dataset

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from omegaconf import OmegaConf
import torch

@dataclass
class EnvSpec:
    """One logical environment family to expand into N data points."""
    # Key in REGISTERED_ENVS, e.g., "GymProxyNoTool"
    name: str
    # How many concrete instances to materialize from this spec
    n_envs: int
    data_source: str = "default"
    # Environment-specific configuration passed through untouched
    config: Dict[str, Any] = field(default_factory=dict)
    # Seed directive: [base] | [min, max] | [min, max, limit]
    # 1-element list: fixed base seed
    # 2-element list: for each env, uniformly random sample a seed in [min, max]
    # 3-element list: as above, but each seed occur at most 'limit' times
    seed: List[int] = field(default_factory=lambda: [0])
    # Optional explicit per-instance seeds; must be longer than n_envs
    seed_list: Optional[List[int]] = None
    max_turns: int = 1
    response_length_per_turn: Optional[int] = None
    # The largest a single observation from this environment may be, in tokens, after the
    # processor has expanded any images. An observation over it is CUT -- its text is
    # trimmed before rendering -- so this is a real ceiling, not advice. Unset it is
    # vagen/harness/budget.py:DEFAULT_MAX_ENV_RESPONSE, a flat 2048.
    #
    # The environment-side twin of ``response_length_per_turn``: one bounds what the model
    # may write in a turn, this bounds what the environment may hand back. It feeds exactly
    # two things -- that cut, and compaction's "can a conversation buy a turn" refusal. It
    # deliberately does NOT size max_response_length or max_model_len; see budget.py.
    max_env_response_per_turn: Optional[int] = None
    #: Deprecated spelling of ``max_env_response_per_turn``. Kept because it is what the
    #: oversized-observation error told people to set.
    env_response_length: Optional[int] = None
    # How many tokens the model may spend *inside* its reasoning block before the engine
    # closes it for it. Unset, nothing is passed and behaviour is unchanged.
    #
    # ★ Not the same lever as ``response_length_per_turn``, and this is the whole point.
    # ``response_length_per_turn`` is ``max_tokens`` -- a guillotine. The model is not told
    # how much room it has, so it plans as if unbounded and is cut mid-sentence; measured
    # on Qwen3.5, that lands the cut before ``</think>`` on 92% of turns at a 2048 cap, and
    # a turn that never closed its reasoning has no answer to score. This instead makes the
    # engine *force* the closing token when the budget runs out, so the model stops
    # reasoning and answers. A bounded turn rather than a truncated one.
    #
    # Model-agnostic here: this is a token count, and nothing in VAGEN knows what a
    # reasoning block looks like. The delimiters live in engine config
    # (``rollout.engine_kwargs.vllm.reasoning_config``: either a registered
    # ``reasoning_parser`` name or an explicit start/end string pair), which is where
    # per-family knowledge belongs. vLLM refuses the request if this is set and that is
    # not configured.
    thinking_token_budget: Optional[int] = None

    def __post_init__(self):
        if self.max_env_response_per_turn is None:
            self.max_env_response_per_turn = self.env_response_length
        elif self.env_response_length not in (None, self.max_env_response_per_turn):
            raise ValueError(
                f"env spec {self.name!r} sets both max_env_response_per_turn="
                f"{self.max_env_response_per_turn} and its old name env_response_length="
                f"{self.env_response_length}. They are one quantity; drop the old one."
            )

@dataclass
class EnvSpecs:
    specs: List[EnvSpec] = field(default_factory=list)


def load_envspecs(yaml_path: str) -> EnvSpecs:
    cfg = OmegaConf.load(yaml_path)
    specs = [EnvSpec(**OmegaConf.to_container(s, resolve=True)) for s in cfg.get("envs", [])]
    return EnvSpecs(specs=specs)

# Upper bound used for RNG sampling when only a base seed is provided
MAX_INT32 = 2 ** 31 - 1


def _make_rng_seed(base_seed: int, spec: EnvSpec, spec_idx: int, hint: str) -> int:
    """Expand the global seed into a deterministic per-spec RNG seed."""
    payload = f"{base_seed}|{spec_idx}|{spec.name}|{hint}"
    h = hashlib.blake2b(payload.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(h, "little")


def _coerce_to_int_list(values: Optional[Sequence]) -> Optional[List[int]]:
    """Return a list of ints if `values` is a proper sequence, otherwise None."""
    if values is None:
        return None
    if isinstance(values, (str, bytes)):
        raise TypeError("seed_list must be a sequence of integers, not string")
    coerced = [int(v) for v in values]
    return coerced


def _normalize_seed_directive(seed_field) -> List[int]:
    """Normalise the seed directive into a list of ints, defaulting to [0]."""
    if seed_field is None:
        return [0]
    if isinstance(seed_field, (int, float)):
        return [int(seed_field)]
    if isinstance(seed_field, Sequence) and not isinstance(seed_field, (str, bytes)):
        coerced = [int(v) for v in seed_field]
        return coerced if coerced else [0]
    raise TypeError("seed must be an integer or a sequence of integers")


def _generate_from_len_one(
    rng: random.Random,
    n_envs: int,
) -> List[int]:
    """len(seed)==1 → sample uniformly from the full 32-bit range."""
    return [rng.randrange(0, MAX_INT32 + 1) for _ in range(n_envs)]


def _generate_from_len_two(
    rng: random.Random,
    n_envs: int,
    minimum: int,
    maximum: int,
) -> List[int]:
    """len(seed)==2 → sample from the inclusive [min, max] range."""
    if maximum < minimum:
        raise ValueError("seed[1] must be >= seed[0] when len(seed) == 2")
    if minimum == maximum:
        return [minimum] * n_envs
    return [rng.randrange(minimum, maximum + 1) for _ in range(n_envs)]


def _generate_from_len_three(
    rng: random.Random,
    n_envs: int,
    minimum: int,
    maximum: int,
    limit: int,
) -> List[int]:
    """len(seed)==3 → sample from [min, max] but cap occurrences per value."""
    if maximum < minimum:
        raise ValueError("seed[1] must be >= seed[0] when len(seed) == 3")
    if limit <= 0:
        raise ValueError("seed[2] must be a positive integer when len(seed) == 3")
    range_size = maximum - minimum + 1
    if range_size <= 0:
        raise ValueError("seed range must contain at least one value")
    if range_size * limit < n_envs:
        raise ValueError(
            "seed range with given limit cannot supply enough unique seeds for n_envs"
        )

    if limit == 1:
        population = range(minimum, maximum + 1)
        return rng.sample(population, n_envs)

    counts = {}
    seeds = []
    while len(seeds) < n_envs:
        candidate = rng.randint(minimum, maximum)
        count = counts.get(candidate, 0)
        if count >= limit:
            continue
        counts[candidate] = count + 1
        seeds.append(candidate)
    return seeds


def _generate_seeds_for_spec(
    spec: EnvSpec,
    base_seed: int,
    spec_idx: int,
) -> List[int]:
    """Generate `n_envs` seeds using either seed_list or seed directive rules."""
    explicit_list = _coerce_to_int_list(spec.seed_list)
    if explicit_list is not None:
        # `<`, not `<=`: a list of exactly n_envs values is used in full, so demanding
        # "more than" rejected the obvious way to name every seed you want. The eval side
        # has always required only `>= n_envs`; this was the one that disagreed.
        if len(explicit_list) < spec.n_envs:
            raise ValueError(
                f"seed_list for env '{spec.name}' has {len(explicit_list)} values for "
                f"n_envs={spec.n_envs}; it must contain at least n_envs values"
            )
        return explicit_list[: spec.n_envs]

    directive = _normalize_seed_directive(spec.seed)
    length = len(directive)
    if length == 0:
        directive = [0]
        length = 1

    rng_seed = _make_rng_seed(base_seed, spec, spec_idx, f"seed-{directive}")
    rng = random.Random(rng_seed)

    if length == 1:
        return _generate_from_len_one(rng, spec.n_envs)
    if length == 2:
        return _generate_from_len_two(rng, spec.n_envs, directive[0], directive[1])
    if length == 3:
        return _generate_from_len_three(rng, spec.n_envs, directive[0], directive[1], directive[2])

    raise ValueError(
        "seed directive must be of length 1, 2, or 3 when seed_list is not provided"
    )


class AgenticDataset(Dataset):
    """
    Expand a list of EnvSpec into individual environment instances with deterministic seeds.
    **No sharding is performed here**; let the DataLoader/sampler handle rank/world_size.
    """

    def __init__(
        self,
        data_files: str, # path of data files yaml
        config: Dict[str, Any],
        **kwargs,
    ):
        # load yaml
        env_specs = load_envspecs(data_files).specs
        base_seed = config.get("base_seed", 0)
        self.items = []

        for spec_idx, spec in enumerate(env_specs):
            seeds = _generate_seeds_for_spec(
                spec,
                base_seed,
                spec_idx,
            )
            for env_seed in seeds:
                # Each record contains env metadata and the resolved RNG seed
                data_source = getattr(spec, "data_source", "default")
                self.items.append(
                    {
                        "env_name": spec.name,
                        "seed": env_seed,
                        "config": spec.config,
                        "max_turns": spec.max_turns,
                        "response_length_per_turn": spec.response_length_per_turn,
                        "max_env_response_per_turn": spec.max_env_response_per_turn,
                        "thinking_token_budget": spec.thinking_token_budget,
                        "data_source": data_source,
                        "agent_name":"gym_agent",
                        # verl's agent loop stores raw_prompt on every output and reads
                        # it unconditionally. VAGEN builds the prompt inside the loop
                        # from environment observations, so there is no dataset-side
                        # chat to carry -- an empty list keeps the type consistent with
                        # RLHFDataset, which puts the chat messages here.
                        "raw_prompt": [],
                        "input_ids": torch.tensor([0]),  # dummy
                        "attention_mask": torch.tensor([0]),  # dummy
                        "position_ids": torch.tensor([0]),  # dummy
                    }
                )

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        return self.items[idx]
