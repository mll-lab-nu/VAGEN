# Remote Gym Environment Framework

**Generic, reusable** HTTP-based client-server framework for remote gym environments.

## Core Design

```
Client (Generic)  → Handles HTTP transport, retry, session management
Server (Generic)  → Handles routing, session ID management
Handler (Custom)  → Only component to customize: implement create_env()
```

**Principle**: Client and Server are 100% reusable. Only implement new Handler for different environments.

## Quick Start

### 1. Implement Handler (Server Side)

```python
from vagen.envs_remote import BaseGymHandler

class MyHandler(BaseGymHandler):
    async def create_env(self, env_config):
        return MyGymEnv(env_config)  # That's it!
```

### 2. Start Server

```python
from vagen.envs_remote import build_gym_service
import uvicorn

# Configure session management
handler = MyHandler(
    session_timeout=1800.0,  # 30 min timeout
    max_sessions=100,        # Max 100 concurrent sessions
)

app = build_gym_service(handler)
uvicorn.run(app, host="0.0.0.0", port=8000)
```

### 3. Use Client

**Option 1: Via Config (Recommended)**

`RemoteEnv` is registered in [`env_registry.yaml`](../../configs/env_registry.yaml):

```yaml
# your_config.yaml
env_name: RemoteEnv  # Use registered remote env

env_config:
  # Client config
  base_urls: "http://server:8000"
  timeout: 120.0
  retries: 8

  # Environment config (passed to server)
  max_steps: 100
  # ... other config ...
```

Full example: [`configs/examples/remote_env_example.yaml`](../../configs/examples/remote_env_example.yaml)

**Option 2: Direct Code Usage**

```python
from vagen.envs_remote import GymImageEnvClient

# Create (synchronous, no connection)
env = GymImageEnvClient(env_config={
    "base_urls": ["http://server1:8000", "http://server2:8000"],
    "timeout": 120.0,
    "retries": 8,
    # ... environment config ...
})

# First reset establishes connection (efficient, 1 round-trip)
obs, info = await env.reset(seed=42)  # → Send {config, seed}, receive {session_id, obs, info}

# Normal usage
obs, reward, done, info = await env.step("action")
await env.close()
```

## Compatibility

### 100% Compatible with gym_agent_loop.py

```python
# gym_agent_loop.py usage (no modification needed)
env = env_cls(env_config)              # Sync init ✓
init_obs, info = await env.reset(seed) # First reset establishes connection ✓
sys_obs = await env.system_prompt()    # Use session ✓
obs, reward, done, info = await env.step(action) # Use session ✓
await env.close()                      # Cleanup session ✓
```

Just modify config:
```yaml
# Use registered RemoteEnv
env_name: RemoteEnv  # Change here (already registered in env_registry.yaml)

env_config:
  base_urls: "http://your-server:8000"
  # ... other config unchanged ...
```

## Core Features

### Client Features
- ✅ URL Pool + Failover
- ✅ Retry with exponential backoff (configurable jitter)
- ✅ Lazy connection (connect on reset)
- ✅ Session locking (one env = one session)

### Server Features
- ✅ Session management (unique session_id)
- ✅ Concurrency control (configurable)
- ✅ API Key authentication (optional)
- ✅ Timeout cleanup (automatic)
- ✅ Max session limit (prevent resource exhaustion)

### Protocol Optimization
**First reset optimization**: Merge connect + reset into 1 round-trip
```
Client → Server: {env_config, seed}
Client ← Server: {session_id, obs, info}
```

## Session Management

### Complete Session Lifecycle

```
1. Client Connect
   └→ Server checks: sessions < max_sessions?
      ├─ Yes → Create session, return session_id
      └─ No  → Return 503 "Max sessions limit reached"

2. Client Interaction (reset/step)
   └→ Update last_access on each call

3. Timeout Cleanup (automatic background)
   └→ Check every minute: (now - last_access) > timeout?
      └─ Yes → env.close() + delete session

4. Explicit Close
   └→ Client: await env.close()
      └→ Server: env.close() + immediately delete session

5. Server Shutdown
   └→ Close all sessions + cleanup resources
```

### Session Tracking

```python
# Handler maintains session state internally
self._sessions: Dict[str, SessionContext] = {
    "session_id_1": SessionContext(
        session_id="...",
        env=env_instance,
        created_at=1234567890.0,
        last_access=1234567895.0,  # Updated on each call
    ),
}
```

**Tracking features**:
- Each client gets unique `session_id` (UUID)
- `last_access` updated on every `reset/step/system_prompt`
- Query active sessions via `GET /sessions` API

### Query Session Status

```bash
# Query all active sessions
curl http://localhost:8000/sessions

# Response
{
  "num_sessions": 3,
  "max_sessions": 100,
  "session_timeout": 1800.0,
  "sessions": [
    {
      "session_id": "abc123...",
      "idle_seconds": 5.0,
      "will_timeout_in": 1795.0
    }
  ]
}
```

## Configuration

### Client Config (env_config)

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `base_urls` | str/list | required | Server URL(s) |
| `timeout` | float | 120.0 | Request timeout (seconds) |
| `retries` | int | 8 | Number of retries |
| `backoff` | float | 2.0 | Backoff multiplier |
| `backoff_jitter_min` | float | 0.7 | Minimum jitter factor |
| `backoff_jitter_range` | float | 0.6 | Jitter range |
| `token` | str | None | API key |
| `failover_after_failures` | int | 4 | Failover after N failures |

### Server Config (Environment Variables)

| Variable | Default | Description |
|----------|---------|-------------|
| `GYM_API_KEY` | "" | API key (empty = no auth) |
| `GYM_MAX_INFLIGHT` | 0 | Max concurrent requests (0 = unlimited) |
| `GYM_ADMIT_TIMEOUT` | 5.0 | Queue timeout (seconds) |

### Handler Config (Constructor)

```python
handler = MyHandler(
    session_timeout=1800.0,  # 30 min idle → auto cleanup
    max_sessions=100,        # Max 100 concurrent sessions (0 = unlimited)
)
```

## Advanced Usage

### Multi-Process + GPU Allocation

Handler can return proxy objects instead of real environments:

```python
# Example 1: GPU Round-Robin (simple)
class GPUHandler(BaseGymHandler):
    def __init__(self, gpus=[0, 1, 2, 3], session_timeout=3600.0):
        super().__init__(session_timeout=session_timeout)
        self.gpus = gpus
        self.next_gpu = 0

    async def create_env(self, env_config):
        gpu_id = self.gpus[self.next_gpu]
        self.next_gpu = (self.next_gpu + 1) % len(self.gpus)

        # Pass gpu_id to environment
        return MyEnv({**env_config, "device": f"cuda:{gpu_id}"})

# Example 2: Multi-Process Isolation (see examples/ for full implementation)
class MultiProcessHandler(BaseGymHandler):
    async def create_env(self, env_config):
        # Return proxy object, real env runs in worker process
        return ProcessEnvProxy(worker_pool, env_config)
```

Detailed examples:
- [`examples/gpu_round_robin_handler.py`](examples/gpu_round_robin_handler.py) - GPU allocation
- [`examples/multiprocess_handler.py`](examples/multiprocess_handler.py) - Multi-process isolation

### Custom Handler

```python
class CustomHandler(BaseGymHandler):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Initialize resource pool (process pool, GPU manager, etc.)
        self.resource_pool = ResourcePool()

    async def create_env(self, env_config):
        # Custom resource allocation logic
        resource = await self.resource_pool.acquire()

        # Can return:
        # - Real environment object
        # - Proxy object (forward to worker process)
        # - Remote environment reference
        # As long as it implements GymImageEnv interface
        return CustomEnvProxy(resource, env_config)

    async def aclose(self):
        # Cleanup resources
        await self.resource_pool.close()
        await super().aclose()
```

## API Reference

### BaseGymHandler

```python
class BaseGymHandler:
    def __init__(self, session_timeout=3600.0, max_sessions=0):
        """
        Args:
            session_timeout: Max idle time before cleanup (seconds)
            max_sessions: Max concurrent sessions (0 = unlimited)
        """

    async def create_env(self, env_config) -> GymImageEnv:
        """Create environment instance (must implement)"""

    async def connect(self, env_config, seed=None) -> HandlerResult:
        """Handle connect request (automatically calls create_env)"""

    async def call(self, session_id, method, params, images) -> HandlerResult:
        """Execute method call"""

    def get_session_stats(self) -> Dict[str, Any]:
        """Get statistics about current sessions"""

    async def aclose(self):
        """Cleanup resources"""
```

### GymImageEnvClient

```python
class GymImageEnvClient(GymImageEnv):
    def __init__(self, env_config):
        """Synchronous initialization (no connection)"""

    async def reset(self, seed) -> (obs, info):
        """Establish connection on first call"""

    async def step(self, action) -> (obs, reward, done, info):
        """Use established session"""

    async def close():
        """Close session"""
```

## Troubleshooting

### Q: First reset is slow?
A: Normal, needs to establish connection + create environment. Already optimized to 1 round-trip.

### Q: How to handle server disconnection?
A: Client automatically retries + fails over to next URL.

### Q: Can I run multiple environments in parallel?
A: Yes! Each env instance has independent session_id.

### Q: How to implement GPU allocation?
A: Implement in Handler's `create_env()`, see [examples/](examples/).

### Q: Connection rejected with "Max sessions limit reached"?
A: Wait for other sessions to timeout, or contact admin to increase `max_sessions`.

### Q: Session unexpectedly timed out?
A: Increase `session_timeout` or ensure client has activity before timeout.

## Monitoring

### Health Check

```bash
curl http://localhost:8000/health

{
  "ok": true,
  "service": "gym-env-service",
  "max_inflight": 50
}
```

### Session Monitoring

```bash
# Check session count
watch -n 10 'curl -s http://localhost:8000/sessions | jq ".num_sessions"'

# View all session details
curl http://localhost:8000/sessions | jq
```

### Logs

```bash
# Handler logs
[Handler] Created session abc123 (1/100)
[Handler] Session def456 timed out after 1805.3s idle
[Handler] Closed session abc123 (0/100 remaining)

# Service logs
[Service] Connect rejected: Max sessions limit reached (100)
```

## File Structure

```
envs_remote/
├── __init__.py                   # Export interface
├── gym_image_env_client.py       # Client implementation
├── service.py                    # FastAPI service
├── handler.py                    # Handler base class
├── multipart_codec.py            # Encoding/decoding utilities
├── README.md                     # This document
└── examples/
    ├── simple_example.py         # Basic example
    ├── gpu_round_robin_handler.py    # GPU allocation
    └── multiprocess_handler.py       # Multi-process isolation
```

## Summary

✅ **100% Compatible** with gym_agent_loop.py
✅ **Zero Code Changes** - Just modify config
✅ **Performance Optimized** - First reset merges connection (1 round-trip)
✅ **Highly Extensible** - Handler supports arbitrary resource management
✅ **Production Ready** - Complete session management, retry, failover, timeout cleanup
✅ **Resource Safe** - Automatic cleanup, max session limits, explicit close

Ready for production use! 🚀
