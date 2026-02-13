"""
Remote Gym Environment Framework

A reusable HTTP-based client-server framework for remote gym environments.

Architecture:
- client: Generic HTTP client (GymImageEnvClient) - handles transport only
- server: Generic FastAPI service - handles routing and session management only
- handler: Customizable business logic - implements environment-specific behavior

Usage:

    # Client side
    from vagen.envs.remote import GymImageEnvClient

    client = GymImageEnvClient(env_config={
        "base_urls": ["http://localhost:8000", "http://localhost:8001"],
        "timeout": 120.0,
        "retries": 8,
        # ... other env_config for the remote environment
    })

    # Server side
    from vagen.envs.remote import build_gym_service, BaseGymHandler

    class MyEnvHandler(BaseGymHandler):
        async def create_env(self, env_config):
            return MyGymEnv(env_config)

    app = build_gym_service(MyEnvHandler())
"""

from .gym_image_env_client import GymImageEnvClient
from .service import build_gym_service
from .handler import BaseGymHandler, HandlerResult

__all__ = [
    "GymImageEnvClient",
    "build_gym_service",
    "BaseGymHandler",
    "HandlerResult",
]
