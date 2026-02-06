# All comments are in English.
from __future__ import annotations

import hashlib
import random
from typing import Any, List, Optional, Sequence

MAX_INT32 = 2**31 - 1


def _coerce_to_int_list(values: Optional[Sequence]) -> Optional[List[int]]:
    """Return a list of ints if `values` is a proper sequence, otherwise None."""
    if values is None:
        return None
    if isinstance(values, (str, bytes)):
        raise TypeError("seed_list must be a sequence of integers, not string")
    return [int(v) for v in values]


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


def _make_rng_seed(base_seed: int, spec: Any, spec_idx: int, hint: str) -> int:
    """Expand the global seed into a deterministic per-spec RNG seed."""
    name = getattr(spec, "name", "unknown")
    split = getattr(spec, "split", "default")
    payload = f"{base_seed}|{spec_idx}|{name}|{split}|{hint}"
    digest = hashlib.blake2b(payload.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "little")


def _generate_from_len_one(rng: random.Random, n_envs: int) -> List[int]:
    """len(seed)==1 → sample uniformly from the full 32-bit range."""
    return [rng.randrange(0, MAX_INT32 + 1) for _ in range(n_envs)]


def _generate_from_len_two(rng: random.Random, n_envs: int, minimum: int, maximum: int) -> List[int]:
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
        raise ValueError("seed range with given limit cannot supply enough unique seeds for n_envs")

    if limit == 1:
        population = range(minimum, maximum + 1)
        return rng.sample(population, n_envs)

    counts: dict[int, int] = {}
    seeds: List[int] = []
    while len(seeds) < n_envs:
        candidate = rng.randint(minimum, maximum)
        count = counts.get(candidate, 0)
        if count >= limit:
            continue
        counts[candidate] = count + 1
        seeds.append(candidate)
    return seeds


def generate_seeds_for_spec(spec: Any, base_seed: int, spec_idx: int) -> List[int]:
    """Generate `n_envs` seeds using either seed_list or seed directive rules."""
    n_envs = int(getattr(spec, "n_envs"))
    explicit_list = _coerce_to_int_list(getattr(spec, "seed_list", None))
    if explicit_list is not None:
        if len(explicit_list) < n_envs:
            raise ValueError(f"seed_list for env '{getattr(spec, 'name', 'unknown')}' must contain at least n_envs values")
        return explicit_list[:n_envs]

    directive = _normalize_seed_directive(getattr(spec, "seed", None))
    if not directive:
        directive = [0]

    rng_seed = _make_rng_seed(base_seed, spec, spec_idx, f"seed-{directive}")
    rng = random.Random(rng_seed)

    length = len(directive)
    if length == 1:
        return _generate_from_len_one(rng, n_envs)
    if length == 2:
        return _generate_from_len_two(rng, n_envs, directive[0], directive[1])
    if length == 3:
        return _generate_from_len_three(rng, n_envs, directive[0], directive[1], directive[2])

    raise ValueError("seed directive must be of length 1, 2, or 3 when seed_list is not provided")


__all__ = ["MAX_INT32", "generate_seeds_for_spec"]
