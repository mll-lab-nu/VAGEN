# Remote Gym Environment Framework

**通用、可复用**的HTTP-based client-server框架，用于远程gym环境。

## 核心设计

```
Client (通用)  → 只负责HTTP传输、retry、session管理
Server (通用)  → 只负责路由、session ID管理
Handler (定制) → 唯一需要定制的部分：实现 create_env()
```

**原则**：Client和Server 100%复用，只需实现新的Handler。

## 快速开始

### 1. 实现Handler (Server端)

```python
from vagen.envs_remote import BaseGymHandler

class MyHandler(BaseGymHandler):
    async def create_env(self, env_config):
        return MyGymEnv(env_config)  # 仅此而已！
```

### 2. 启动Server

```python
from vagen.envs_remote import build_gym_service
import uvicorn

app = build_gym_service(MyHandler())
uvicorn.run(app, host="0.0.0.0", port=8000)
```

### 3. 使用Client

```python
from vagen.envs_remote import GymImageEnvClient

# 创建（同步，不连接）
env = GymImageEnvClient(env_config={
    "base_urls": ["http://server1:8000", "http://server2:8000"],
    "timeout": 120.0,
    "retries": 8,
    # ... 环境配置 ...
})

# 第一次reset时建立连接（高效，1次往返）
obs, info = await env.reset(seed=42)  # → 发送 {config, seed}, 收到 {session_id, obs, info}

# 正常使用
obs, reward, done, info = await env.step("action")
await env.close()
```

## 兼容性

### 与 gym_agent_loop.py 完全兼容

```python
# gym_agent_loop.py 的使用方式（无需修改）
env = env_cls(env_config)              # 同步初始化 ✓
init_obs, info = await env.reset(seed) # 第一次reset建立连接 ✓
sys_obs = await env.system_prompt()    # 使用session ✓
obs, reward, done, info = await env.step(action) # 使用session ✓
await env.close()                      # 清理session ✓
```

只需修改配置：
```yaml
env_registry:
  my_task: "vagen.envs_remote.GymImageEnvClient"  # 改这里
env_config:
  base_urls: "http://your-server:8000"
  # ... 其他配置不变 ...
```

## 核心特性

### Client特性
- ✅ URL Pool + Failover
- ✅ Retry with exponential backoff (可配置jitter)
- ✅ Lazy connection (reset时才连接)
- ✅ Session locking (一个env = 一个session)

### Server特性
- ✅ Session管理 (unique session_id)
- ✅ 并发控制 (可配置)
- ✅ API Key认证 (可选)
- ✅ 超时清理 (自动)

### Protocol优化
**第一次reset优化**：合并connect + reset为1次往返
```
Client → Server: {env_config, seed}
Client ← Server: {session_id, obs, info}
```

## 配置参数

### Client配置 (env_config)

| 参数 | 类型 | 默认值 | 说明 |
|-----|------|--------|-----|
| `base_urls` | str/list | required | 服务器URL(s) |
| `timeout` | float | 120.0 | 请求超时(秒) |
| `retries` | int | 8 | 重试次数 |
| `backoff` | float | 2.0 | 退避乘数 |
| `backoff_jitter_min` | float | 0.7 | Jitter最小值 |
| `backoff_jitter_range` | float | 0.6 | Jitter范围 |
| `token` | str | None | API密钥 |
| `failover_after_failures` | int | 4 | N次失败后切换URL |

### Server配置 (环境变量)

| 变量 | 默认值 | 说明 |
|------|--------|-----|
| `GYM_API_KEY` | "" | API密钥 (空=无认证) |
| `GYM_MAX_INFLIGHT` | 0 | 最大并发数 (0=无限) |
| `GYM_ADMIT_TIMEOUT` | 5.0 | 队列超时(秒) |

## 高级用法

### 多进程 + GPU分配

Handler可以返回代理对象而非真实环境：

```python
# 示例 1: GPU Round-Robin (简单)
class GPUHandler(BaseGymHandler):
    def __init__(self, gpus=[0, 1, 2, 3]):
        super().__init__()
        self.gpus = gpus
        self.next_gpu = 0

    async def create_env(self, env_config):
        gpu_id = self.gpus[self.next_gpu]
        self.next_gpu = (self.next_gpu + 1) % len(self.gpus)

        # 传递gpu_id给环境
        return MyEnv({**env_config, "device": f"cuda:{gpu_id}"})

# 示例 2: 多进程隔离 (完整示例见 examples/)
class MultiProcessHandler(BaseGymHandler):
    async def create_env(self, env_config):
        # 返回代理对象，实际环境在worker进程中
        return ProcessEnvProxy(worker_pool, env_config)
```

详细示例：
- [`examples/gpu_round_robin_handler.py`](examples/gpu_round_robin_handler.py) - GPU分配
- [`examples/multiprocess_handler.py`](examples/multiprocess_handler.py) - 多进程隔离

### 自定义Handler

```python
class CustomHandler(BaseGymHandler):
    def __init__(self, **kwargs):
        super().__init__()
        # 初始化资源池（进程池、GPU管理器等）
        self.resource_pool = ResourcePool()

    async def create_env(self, env_config):
        # 自定义资源分配逻辑
        resource = await self.resource_pool.acquire()

        # 可以返回：
        # - 真实环境对象
        # - 代理对象（转发到worker进程）
        # - 远程环境引用
        # 只要实现GymImageEnv接口即可
        return CustomEnvProxy(resource, env_config)

    async def aclose(self):
        # 清理资源
        await self.resource_pool.close()
        await super().aclose()
```

## API

### BaseGymHandler

```python
class BaseGymHandler:
    async def create_env(self, env_config) -> GymImageEnv:
        """创建环境实例 (必须实现)"""

    async def connect(self, env_config, seed=None) -> HandlerResult:
        """处理连接请求 (自动调用 create_env)"""

    async def call(self, session_id, method, params, images) -> HandlerResult:
        """执行方法调用"""

    async def aclose(self):
        """清理资源"""
```

### GymImageEnvClient

```python
class GymImageEnvClient(GymImageEnv):
    def __init__(self, env_config):
        """同步初始化 (不连接)"""

    async def reset(self, seed) -> (obs, info):
        """第一次调用时建立连接"""

    async def step(self, action) -> (obs, reward, done, info):
        """使用已建立的session"""

    async def close():
        """关闭session"""
```

## 故障排查

### Q: 第一次reset很慢？
A: 正常，需要建立连接+创建环境。已优化到1次往返。

### Q: 如何处理服务器断连？
A: Client自动retry + failover到下一个URL。

### Q: 能否并行多个环境？
A: 可以！每个env实例都有独立session_id。

### Q: 如何实现GPU分配？
A: 在Handler的`create_env()`中实现，见[examples/](examples/)。

## 文件结构

```
envs_remote/
├── __init__.py                   # 导出接口
├── gym_image_env_client.py       # Client实现
├── service.py                    # FastAPI服务
├── handler.py                    # Handler基类
├── multipart_codec.py            # 编解码工具
├── README.md                     # 本文档
└── examples/
    ├── simple_example.py         # 基础示例
    ├── gpu_round_robin_handler.py    # GPU分配
    └── multiprocess_handler.py       # 多进程隔离
```

## 总结

✅ **100%兼容** gym_agent_loop.py
✅ **零代码修改** - 只需改配置
✅ **性能优化** - 第一次reset合并连接(1次往返)
✅ **高度可扩展** - Handler支持任意资源管理策略
✅ **生产就绪** - Retry, failover, 超时清理完备

可以放心使用！🚀
