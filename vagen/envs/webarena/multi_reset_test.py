"""
Test: multiple resets on the same env to verify browser reuse.

Measures reset time across multiple episodes to confirm that
the first reset (cold start) is slower than subsequent resets (warm, reusing browser).

Usage:
    python -m vagen.envs.webarena.multi_reset_test
    python -m vagen.envs.webarena.multi_reset_test --num_resets 10 --steps_per_reset 3
"""

import random
import time

import fire


RANDOM_ACTIONS = [
    "scroll [down]",
    "scroll [up]",
    "press [Enter]",
    "press [Escape]",
    "click [1]",
    "click [2]",
    "click [5]",
    "go_back",
]


def test_multi_reset(
    config_file: str = "/work/nvme/bgig/ryu4/VAGEN/vagen/envs/webarena/config_files/0.json",
    num_resets: int = 5,
    steps_per_reset: int = 3,
    headless: bool = True,
):
    from vagen.envs.webarena.browser_env import ScriptBrowserEnv, create_id_based_action

    print(f"=== Multi-reset test: {num_resets} resets x {steps_per_reset} steps ===")
    print(f"Config: {config_file}\n")

    env = ScriptBrowserEnv(
        headless=headless,
        observation_type="accessibility_tree",
        current_viewport_only=True,
        viewport_size={"width": 1280, "height": 720},
        sleep_after_execution=0.0,
    )

    reset_times = []
    step_times = []

    for episode in range(num_resets):
        # --- reset ---
        t0 = time.perf_counter()
        try:
            obs, info = env.reset(options={"config_file": config_file})
            dt_reset = time.perf_counter() - t0
            reset_times.append(dt_reset)
            print(f"  [episode {episode}] reset done in {dt_reset:.3f}s")
        except Exception as e:
            dt_reset = time.perf_counter() - t0
            reset_times.append(dt_reset)
            print(f"  [episode {episode}] reset FAILED in {dt_reset:.3f}s: {e}")
            continue

        # --- steps ---
        for step_i in range(steps_per_reset):
            action_str = random.choice(RANDOM_ACTIONS)
            t0 = time.perf_counter()
            try:
                action = create_id_based_action(action_str)
                obs, reward, terminated, truncated, info = env.step(action)
                dt_step = time.perf_counter() - t0
                step_times.append(dt_step)
                print(f"  [episode {episode}] step {step_i} done in {dt_step:.3f}s ({action_str})")
                if terminated:
                    break
            except Exception as e:
                dt_step = time.perf_counter() - t0
                step_times.append(dt_step)
                print(f"  [episode {episode}] step {step_i} FAILED in {dt_step:.3f}s ({action_str}): {e}")

    env.close()

    # --- Report ---
    print("\n" + "=" * 50)
    print("RESULTS")
    print("=" * 50)
    if reset_times:
        print(f"  Reset times: {['%.3f' % t for t in reset_times]}")
        print(f"  First reset (cold):  {reset_times[0]:.3f}s")
        if len(reset_times) > 1:
            warm_resets = reset_times[1:]
            print(f"  Avg warm reset:      {sum(warm_resets)/len(warm_resets):.3f}s")
        print(f"  Avg all resets:      {sum(reset_times)/len(reset_times):.3f}s")
    if step_times:
        print(f"  Avg step time:       {sum(step_times)/len(step_times):.3f}s")
    print()


if __name__ == "__main__":
    fire.Fire(test_multi_reset)
