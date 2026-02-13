"""
Simple example demonstrating the remote gym environment framework.

This example shows:
1. How to implement a custom handler
2. How to start a server
3. How to use the client

Run server:
    python simple_example.py server

Run client:
    python simple_example.py client
"""

import asyncio
import sys
from typing import Any, Dict

from vagen.envs.remote import BaseGymHandler, GymImageEnvClient, build_gym_service
from vagen.envs.gym_image_env import GymImageEnv


# ============================================================================
# Step 1: Implement a simple dummy environment (this would be your real env)
# ============================================================================
class DummyEnv(GymImageEnv):
    """A simple dummy environment for demonstration."""

    def __init__(self, env_config: Dict[str, Any]):
        super().__init__(env_config)
        self.step_count = 0
        self.max_steps = env_config.get("max_steps", 10)

    async def system_prompt(self) -> Dict[str, Any]:
        return {"obs_str": "This is a dummy environment. Type anything to step."}

    async def reset(self, seed: int):
        self.step_count = 0
        obs = {"obs_str": f"Environment reset with seed={seed}. Step 0/{self.max_steps}"}
        info = {"seed": seed}
        return obs, info

    async def step(self, action_str: str):
        self.step_count += 1
        done = self.step_count >= self.max_steps

        obs = {
            "obs_str": f"You said: '{action_str}'. Step {self.step_count}/{self.max_steps}"
        }
        reward = 1.0 if not done else 10.0
        info = {"step": self.step_count}

        return obs, reward, done, info

    async def close(self):
        print(f"[DummyEnv] Closed after {self.step_count} steps")


# ============================================================================
# Step 2: Implement handler (only this needs to be customized per environment)
# ============================================================================
class DummyEnvHandler(BaseGymHandler):
    """Handler for DummyEnv. This is the ONLY custom code needed."""

    async def create_env(self, env_config: Dict[str, Any]) -> Any:
        """Create environment instance. This is the only method you need to implement!"""
        print(f"[Handler] Creating DummyEnv with config: {env_config}")
        return DummyEnv(env_config)


# ============================================================================
# Step 3: Server code (completely generic, just plug in your handler)
# ============================================================================
def run_server(port: int = 8000):
    """Start the server."""
    import uvicorn

    # Create handler
    handler = DummyEnvHandler(session_timeout=600.0)

    # Build service (generic, reusable)
    app = build_gym_service(handler)

    print(f"Starting server on port {port}...")
    print(f"Health check: http://localhost:{port}/health")
    print(f"Ready to accept connections!")

    uvicorn.run(app, host="0.0.0.0", port=port)


# ============================================================================
# Step 4: Client code (completely generic, works with any environment)
# ============================================================================
async def run_client(urls: list = None):
    """Run client test."""
    if urls is None:
        urls = ["http://localhost:8000"]

    print(f"Connecting to servers: {urls}")

    # Create client (generic, reusable)
    client = GymImageEnvClient(
        env_config={
            # Client config
            "base_urls": urls,
            "timeout": 30.0,
            "retries": 3,
            "backoff": 1.0,
            # Environment config (passed to remote handler)
            "max_steps": 5,
        }
    )

    try:
        # Use like any GymImageEnv
        print("\n=== Getting system prompt ===")
        system_prompt = await client.system_prompt()
        print(f"System: {system_prompt['obs_str']}")

        print("\n=== Resetting environment ===")
        obs, info = await client.reset(seed=42)
        print(f"Obs: {obs['obs_str']}")
        print(f"Info: {info}")

        print("\n=== Running episode ===")
        for i in range(5):
            action = f"Action #{i + 1}"
            print(f"\nAction: {action}")

            obs, reward, done, info = await client.step(action)
            print(f"Obs: {obs['obs_str']}")
            print(f"Reward: {reward}, Done: {done}")

            if done:
                print("\nEpisode finished!")
                break

    finally:
        print("\n=== Closing ===")
        await client.close()
        print("Done!")


# ============================================================================
# Main
# ============================================================================
def main():
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python simple_example.py server [--port PORT]")
        print("  python simple_example.py client [--urls URL1 URL2 ...]")
        sys.exit(1)

    mode = sys.argv[1]

    if mode == "server":
        port = 8000
        if "--port" in sys.argv:
            port = int(sys.argv[sys.argv.index("--port") + 1])
        run_server(port)

    elif mode == "client":
        urls = ["http://localhost:8000"]
        if "--urls" in sys.argv:
            idx = sys.argv.index("--urls")
            urls = sys.argv[idx + 1 :]
        asyncio.run(run_client(urls))

    else:
        print(f"Unknown mode: {mode}")
        sys.exit(1)


if __name__ == "__main__":
    main()
