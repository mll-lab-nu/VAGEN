# Remote Gym Environment Framework

Some environments can't run in the same process as training — they may need dedicated GPUs, have incompatible dependencies, or use blocking C++/Unity backends that would stall the training loop. This framework provides a **stateful** HTTP server/client layer so the environment runs as a separate process and the training side talks to it via a drop-in client.

Stateful means each client gets a persistent session: the server keeps the environment instance alive across multiple `reset()` / `step()` / `close()` calls within the same session, rather than recreating it on every request.

**What it provides:**

- A generic HTTP client (`GymImageEnvClient`) that implements the standard `GymImageEnv` interface — swap it in with zero code changes on the training side.
- A generic FastAPI server (`GymService`) that handles routing, auth, and concurrency.
- A handler base class (`BaseGymHandler`) that manages sessions and timeout cleanup.
- Multipart encoding for passing images + JSON over HTTP.
- Retry, failover, and URL pool support on the client side.

**What you write:**

A handler with one method — `create_env()` — that returns your environment instance. Everything else (session management, HTTP transport, image serialization) is handled by the framework.

## Usage

### Handler

```python
from vagen.envs_remote import BaseGymHandler

class MyHandler(BaseGymHandler):
    async def create_env(self, env_config):
        return MyEnv(env_config)
```

For more control (GPU assignment, env caching, capacity limits), override `connect()` and `_handle_close()`. See `vagen/envs/navigation/handler.py`.

### Server

```python
from vagen.envs_remote import GymService

handler = MyHandler(session_timeout=3600.0)
app = GymService(handler).build()
uvicorn.run(app, host="0.0.0.0", port=8000, workers=1)
```

### Client

```python
from vagen.envs_remote import GymImageEnvClient

env = GymImageEnvClient({
    "base_urls": "http://server:8000",
    "timeout": 120,
    "retries": 3,
    # remaining keys forwarded as env_config
})

obs, info = await env.reset(seed=42)
obs, reward, done, info = await env.step(action)
await env.close()
```

Or via eval config (`RemoteEnv` is registered in `env_registry.yaml`):

```yaml
envs:
  - name: RemoteEnv
    config:
      base_urls: "http://localhost:8000"
      timeout: 120
      retries: 3
```

## File structure

```
envs_remote/
├── __init__.py               # Public API
├── gym_image_env_client.py   # Client (drop-in GymImageEnv replacement)
├── service.py                # Server (FastAPI)
├── handler.py                # Handler base class (sessions, cleanup)
├── multipart_codec.py        # Image + JSON encoding over HTTP
└── README.md
```
