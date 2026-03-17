"""
Benchmark: async multi-client stress test for the Navigation remote service.

Tests that the service handles N concurrent clients each running M steps,
repeated over R rounds, with timing and success stats.

Usage:
    python -m vagen.envs.navigation.benchmark \
        --base_url http://localhost:8000 \
        --num_rounds 10 \
        --num_clients 64 \
        --num_steps 5

    # Or with fire:
    python -m vagen.envs.navigation.benchmark --help
"""

from __future__ import annotations

import asyncio
import logging
import random
import time
from dataclasses import dataclass, field
from typing import List

import fire

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
LOGGER = logging.getLogger("benchmark")

# ---------------------------------------------------------------------------
# Valid actions in free_think format (guaranteed to be parseable)
# ---------------------------------------------------------------------------
VALID_ACTIONS = [
    "moveahead", "moveback", "moveright", "moveleft",
    "rotateright", "rotateleft", "lookup", "lookdown",
]


def random_action_str(max_actions: int = 3, action_sep: str = ",") -> str:
    """Generate a random valid free_think format action string."""
    n = random.randint(1, max_actions)
    actions = [random.choice(VALID_ACTIONS) for _ in range(n)]
    answer = action_sep.join(actions)
    think = f"I will {actions[0]} to explore the environment."
    return f"<think>{think}</think><answer>{answer}</answer>"


# ---------------------------------------------------------------------------
# Per-client episode runner
# ---------------------------------------------------------------------------

@dataclass
class ClientResult:
    client_id: int
    seed: int
    steps_completed: int = 0
    reset_time: float = 0.0
    step_times: List[float] = field(default_factory=list)
    close_time: float = 0.0
    total_time: float = 0.0
    error: str = ""


async def run_single_client(
    client_id: int,
    base_url: str,
    num_steps: int,
    seed: int,
    timeout: float,
    env_config_overrides: dict,
) -> ClientResult:
    """Run one client: connect → reset → N steps → close."""
    from vagen.envs_remote import GymImageEnvClient

    result = ClientResult(client_id=client_id, seed=seed)
    t0 = time.perf_counter()

    env_config = {
        "base_urls": [base_url],
        "timeout": timeout,
        "retries": 3,
        "backoff": 1.0,
        "eval_set": "base",
        "prompt_format": "free_think",
        **env_config_overrides,
    }

    env = GymImageEnvClient(env_config)
    try:
        # Reset
        t_reset = time.perf_counter()
        obs, info = await env.reset(seed)
        result.reset_time = time.perf_counter() - t_reset

        # Steps
        for step_i in range(num_steps):
            action = random_action_str()
            t_step = time.perf_counter()
            obs, reward, done, info = await env.step(action)
            result.step_times.append(time.perf_counter() - t_step)
            result.steps_completed += 1
            if done:
                break

        # Close
        t_close = time.perf_counter()
        await env.close()
        result.close_time = time.perf_counter() - t_close

    except Exception as e:
        result.error = str(e)
        LOGGER.error(f"[Client {client_id}] Error: {e}")
        # Best-effort close
        try:
            await env.close()
        except Exception:
            pass

    result.total_time = time.perf_counter() - t0
    return result


# ---------------------------------------------------------------------------
# Main benchmark
# ---------------------------------------------------------------------------

async def _run_benchmark(
    base_url: str = "http://localhost:8000",
    num_rounds: int = 10,
    num_clients: int = 64,
    num_steps: int = 5,
    timeout: float = 300.0,
    env_config_overrides: dict | None = None,
):
    """Core async benchmark loop."""
    overrides = env_config_overrides or {}
    all_round_stats = []

    LOGGER.info(
        f"Benchmark: {num_rounds} rounds × {num_clients} clients × {num_steps} steps"
    )
    LOGGER.info(f"Server: {base_url}")

    # Warm-up: quick health check
    import httpx
    async with httpx.AsyncClient(timeout=10) as hc:
        try:
            resp = await hc.get(f"{base_url}/health")
            LOGGER.info(f"Health check: {resp.json()}")
        except Exception as e:
            LOGGER.error(f"Health check failed: {e}. Is the server running?")
            return

    benchmark_start = time.perf_counter()

    for round_idx in range(num_rounds):
        LOGGER.info(f"--- Round {round_idx + 1}/{num_rounds} ---")
        round_start = time.perf_counter()

        # Launch all clients concurrently
        tasks = [
            run_single_client(
                client_id=i,
                base_url=base_url,
                num_steps=num_steps,
                seed=round_idx * num_clients + i,
                timeout=timeout,
                env_config_overrides=overrides,
            )
            for i in range(num_clients)
        ]
        results: List[ClientResult] = await asyncio.gather(*tasks)

        round_time = time.perf_counter() - round_start

        # Compute stats
        errors = [r for r in results if r.error]
        successful = [r for r in results if not r.error]
        reset_times = [r.reset_time for r in successful]
        all_step_times = [t for r in successful for t in r.step_times]
        close_times = [r.close_time for r in successful]
        total_steps = sum(r.steps_completed for r in successful)

        stats = {
            "round": round_idx + 1,
            "round_time": round_time,
            "num_successful": len(successful),
            "num_errors": len(errors),
            "total_steps": total_steps,
            "throughput_steps_per_sec": total_steps / round_time if round_time > 0 else 0,
            "avg_reset_time": sum(reset_times) / len(reset_times) if reset_times else 0,
            "avg_step_time": sum(all_step_times) / len(all_step_times) if all_step_times else 0,
            "max_step_time": max(all_step_times) if all_step_times else 0,
            "avg_close_time": sum(close_times) / len(close_times) if close_times else 0,
        }
        all_round_stats.append(stats)

        LOGGER.info(
            f"  Round time:   {stats['round_time']:.2f}s"
        )
        LOGGER.info(
            f"  Successful:   {stats['num_successful']}/{num_clients}, "
            f"Errors: {stats['num_errors']}"
        )
        LOGGER.info(
            f"  Steps:        {stats['total_steps']} total, "
            f"{stats['throughput_steps_per_sec']:.1f} steps/s"
        )
        LOGGER.info(
            f"  Avg reset:    {stats['avg_reset_time']:.3f}s, "
            f"Avg step: {stats['avg_step_time']:.3f}s, "
            f"Max step: {stats['max_step_time']:.3f}s"
        )

        if errors:
            for r in errors[:3]:
                LOGGER.warning(f"  Error client {r.client_id}: {r.error}")

    benchmark_total = time.perf_counter() - benchmark_start

    # Summary
    LOGGER.info("=" * 60)
    LOGGER.info("BENCHMARK SUMMARY")
    LOGGER.info("=" * 60)
    LOGGER.info(f"  Total time:       {benchmark_total:.2f}s")
    LOGGER.info(f"  Rounds:           {num_rounds}")
    LOGGER.info(f"  Clients/round:    {num_clients}")
    LOGGER.info(f"  Steps/client:     {num_steps}")

    avg_round = sum(s["round_time"] for s in all_round_stats) / len(all_round_stats)
    avg_throughput = sum(s["throughput_steps_per_sec"] for s in all_round_stats) / len(all_round_stats)
    total_errors = sum(s["num_errors"] for s in all_round_stats)
    total_steps_all = sum(s["total_steps"] for s in all_round_stats)

    LOGGER.info(f"  Avg round time:   {avg_round:.2f}s")
    LOGGER.info(f"  Avg throughput:   {avg_throughput:.1f} steps/s")
    LOGGER.info(f"  Total steps:      {total_steps_all}")
    LOGGER.info(f"  Total errors:     {total_errors}")
    LOGGER.info("=" * 60)


def main(
    base_url: str = "http://localhost:8000",
    num_rounds: int = 10,
    num_clients: int = 64,
    num_steps: int = 5,
    timeout: float = 300.0,
    eval_set: str = "base",
    prompt_format: str = "free_think",
):
    """
    Run async multi-client benchmark against the Navigation service.

    Args:
        base_url: Server URL.
        num_rounds: Number of benchmark rounds.
        num_clients: Number of concurrent clients per round.
        num_steps: Number of steps each client takes per round.
        timeout: HTTP request timeout in seconds.
        eval_set: Navigation eval set to use.
        prompt_format: Prompt format for the environment.
    """
    asyncio.run(
        _run_benchmark(
            base_url=base_url,
            num_rounds=num_rounds,
            num_clients=num_clients,
            num_steps=num_steps,
            timeout=timeout,
            env_config_overrides={
                "eval_set": eval_set,
                "prompt_format": prompt_format,
            },
        )
    )


if __name__ == "__main__":
    fire.Fire(main)
