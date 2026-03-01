# GraphRL 深度解析（中文）

> 以添加**图像观测环境**为主线，讲解 GraphRL 的核心数据流与扩展点

---

## 目录

1. [框架概述](#1-框架概述)
2. [核心数据结构：图](#2-核心数据结构图)
3. [traj_to_transitions 详解](#3-traj_to_transitions-详解)
4. [节点/边的去重机制](#4-节点边的去重机制)
5. [如何添加图像环境](#5-如何添加图像环境)
6. [配置与运行](#6-配置与运行)

---

## 1. 框架概述

GraphRL 是「**RL 采样 → 轨迹建图 → 图采样 SFT → SFT 训练**」的迭代训练框架。

```
初始模型
  │
  ▼
① RL 训练（VAGEN/verl）→ 游戏轨迹 .jsonl + RL checkpoint
  │
  ▼  [后台线程实时转换]
② 轨迹 → 图（traj_to_transitions）→ graph.json
  │
  ▼
③ 图 → SFT 数据集（generate_datasets）→ sft_data/
  │
  ▼
④ SFT 训练（LLaMA-Factory）→ 更强模型 → 进入下一轮
```

**每轮迭代目录：**
```
iter_000/
├── rl/
│   ├── rollout_data/       ← VAGEN 原始轨迹 .jsonl（转换后删除）
│   ├── verl_checkpoints/   ← 训练 checkpoint（完成后删除）
│   └── graph/
│       ├── graph.json      ← 状态转移图
│       └── images/         ← 节点/边图片（多模态环境）
├── rl_model/               ← 最终 RL 模型（HuggingFace 格式）
├── sft_data/               ← SFT 数据集
└── sft_model/              ← SFT 训练后的模型（进入下一轮）
```

---

## 2. 核心数据结构：图

图用 NetworkX `MultiDiGraph` 实现，保存为 `graph.json`。

**节点**代表一个环境状态，**边**代表一次动作/转移。

```json
{
  "nodes": {
    "abc123def456": {
      "state": "###\n_P_\n###",        // 用于去重的原始状态（任意 JSON 对象）
      "obs_str": "###\n_P_\n###",      // 给模型看的文本描述
      "image_paths": ["images/abc123_0.jpg"],  // 相对路径，空列表则无图
      "extra": {}                       // 自定义元数据
    }
  },
  "edges": [
    {
      "from": "abc123def456",
      "to": "789ghijkl012",
      "obs_str": "Right",              // 动作字符串
      "image_paths": [],
      "extra": {}
    }
  ]
}
```

---

## 3. `traj_to_transitions` 详解

这是**你唯一必须实现的方法**。框架负责读取 JSONL、多进程调用、去重、写入图。

### 3.1 函数签名

```python
def traj_to_transitions(
    self,
    messages: List[Dict[str, str]],
    rollout_dir: Path,
    step_idx: int,
    line_idx: int,
) -> List[Tuple[NodeData, EdgeData, NodeData]]:
    """
    输入：一个完整游戏回合的对话消息
    输出：(起始节点, 边, 目标节点) 三元组列表
    """
```

### 3.2 每个参数详解

---

#### `messages: List[Dict[str, str]]`

**来源：** VAGEN 产出的 `.jsonl` 文件。每行是一个回合，格式为：
```json
{"input": "<|im_start|>user\n...<|im_end|>", "output": "<|im_start|>assistant\n...<|im_end|>..."}
```
框架将 `input + output` 拼接后，用正则按 ChatML 格式（`<|im_start|>role\ncontent<|im_end|>`）解析，得到：

```python
messages = [
    {"role": "user",      "content": "[Initial Observation]:\n###\n_P_\n###\nDecide your next action"},
    {"role": "assistant", "content": "<answer>Right</answer>"},
    {"role": "user",      "content": "After that, the observation is:\n###\n__P\n###\nDecide your next action"},
    {"role": "assistant", "content": "<answer>Down</answer>"},
    # ...
]
```

**对于图像环境（如 ViewSuite）：**
```python
messages = [
    {"role": "user",      "content": "You're in scene0353_02. [tx=1.0, ty=2.0, tz=3.0, rx=-90°, ry=0°, rz=45°] <image>"},
    {"role": "assistant", "content": "<action>turn_left</action>"},
    {"role": "user",      "content": "[tx=1.5, ty=2.0, tz=3.0, rx=-90°, ry=0°, rz=45°] <image>"},
    # ...
]
```
> 注意：消息里的 `<image>` 是**占位符**，实际图片文件需要通过 `rollout_dir`/`step_idx`/`line_idx` 定位（见下文）。

---

#### `rollout_dir: Path`

rollout 数据的根目录，即 `iter_000/rl/rollout_data/`。

**图像文件的组织方式：**
```
rollout_dir/
├── 0.jsonl              ← step_idx=0 的轨迹
├── 1.jsonl              ← step_idx=1 的轨迹
├── image_0/             ← step_idx=0 对应的图片目录
│   ├── images_0/        ← line_idx=0（第0行轨迹）的图片
│   │   ├── 0.png        ← 第0张图（global_img_idx=0）
│   │   ├── 1.png        ← 第1张图（global_img_idx=1）
│   │   └── ...
│   ├── images_1/        ← line_idx=1（第1行轨迹）的图片
│   └── ...
└── image_1/             ← step_idx=1 对应的图片目录
```

---

#### `step_idx: int`

从 JSONL **文件名**（去后缀）解析的整数，对应上面目录结构中的 `image_{step_idx}/`。
- 文件名 `5.jsonl` → `step_idx = 5`
- 文件名非数字 → `step_idx = 0`

---

#### `line_idx: int`

JSONL 文件中的**行号**（0-indexed），对应 `images_{line_idx}/` 子目录。
每个 JSONL 文件的每一行（每个回合）对应一个独立的图片目录。

---

#### 图像路径构造示例

```python
# 构建图像基础路径
image_base = rollout_dir / f"image_{step_idx}" / f"images_{line_idx}"
# e.g., rollout_data/image_5/images_3/

# 遍历 messages 时，用 global_img_idx 追踪第几张图
global_img_idx = 0
for msg in messages:
    if msg["role"] == "user":
        num_images = msg["content"].count("<image>")
        if num_images > 0:
            img_path = image_base / f"{global_img_idx}.png"  # 尝试 .png 或 .jpg
            global_img_idx += num_images
```

---

#### 返回值

```python
List[Tuple[NodeData, EdgeData, NodeData]]
```

每个三元组 `(src, edge, dst)` 代表「从状态 src，执行 edge 中的动作，到达状态 dst」。

框架拿到列表后，自动：
1. 对每个 NodeData 做去重（详见第4节）
2. 将 `source_images` 里的图片从 rollout 目录复制到 `graph/images/`
3. 更新 graph.json

---

### 3.3 返回值数据结构：`VagenNodeData` 与 `VagenEdgeData`

#### `VagenNodeData` 字段

```python
VagenNodeData(
    state=...,           # 【必须】用于去重的原始状态（任意 JSON 可序列化对象）
    obs_str=...,         # 【可选】给模型看的文本描述（默认 = str(state)）
    source_images=[...], # 【图像环境用】rollout 目录里的图片绝对路径列表（框架会自动复制并删除）
    image_paths=[...],   # 【通常留空】graph/images/ 里的相对路径（框架填充）
    extra={...},         # 【可选】任意元数据（scene_id、任务类型等）
)
```

**关键区分：`state` vs `obs_str`**

| 字段 | 用途 | 典型值 |
|------|------|--------|
| `state` | **去重键的原始输入**，用于计算 unique_key（hash 输入） | 文本环境：棋盘字符串；图像环境：`{"scene_id": "xxx", "pose": {...}}` |
| `obs_str` | **模型可读的描述**，写入 graph.json 并用于 SFT prompt | 文本环境：同 state；图像环境：`"[tx=1.00, ty=2.00, ...]"` |

> **视觉环境要点：** 如果观测是图片，`state` 应该是**可以精确去重的符号表示**（如位姿坐标、哈希值），而不是图片本身。图片通过 `source_images` 传入。

**`source_images` vs `image_paths`：**

| 字段 | 时机 | 来源 |
|------|------|------|
| `source_images` | 你在 `traj_to_transitions` 中填入 | rollout 目录里的绝对路径 |
| `image_paths` | 框架自动填入 | 复制到 `graph/images/` 后的相对路径 |

你只需要填 `source_images`，`image_paths` 会自动处理。

#### `VagenEdgeData` 字段

```python
VagenEdgeData(
    obs_str="turn_left",   # 动作字符串（必须）
    image_paths=[],        # 边关联的图片（通常为空）
    extra={},
)
```

---

## 4. 节点/边的去重机制

每个 `NodeData` 需实现三个方法，控制去重行为：

| 方法 | 返回值 | 作用 |
|------|--------|------|
| `bucket_key()` | `str` | **粗粒度分桶**：只与同桶内的节点比较，减少比较次数 |
| `unique_key()` | `str` | **精确 ID**：存入 graph.json 作为节点 key |
| `is_similar_to(other)` | `bool` | **同桶内相似度**：返回 True 则复用已有节点 |

**去重流程：**
```
新节点到来
  ↓
计算 bucket_key()
  ↓
遍历同桶内所有已有节点
  ↓
对每个已有节点调用 is_similar_to()
  ├── 返回 True → 复用该节点的 unique_key，丢弃新节点
  └── 所有都返回 False → 用 unique_key() 创建新节点，加入桶
```

### 4.1 默认实现（文本环境）

`VagenNodeData` 的默认实现适用于**文本状态**：

```python
def unique_key(self) -> str:
    # SHA256(state) 前 16 位，确保相同文本 → 相同 ID
    return hashlib.sha256(json.dumps(self.state).encode()).hexdigest()[:16]

def bucket_key(self) -> str:
    return self.unique_key()  # bucket = 精确 ID → 每桶只有 1 个节点

def is_similar_to(self, other) -> bool:
    return True  # 同桶必然是同一状态
```

效果：O(1) 哈希去重，相同文本状态只存一次。

### 4.2 图像/位姿环境（ViewSuite 示例）

当状态含有浮点坐标时，需要容差匹配：

```python
class ViewSuiteNodeData(VagenNodeData):

    def unique_key(self) -> str:
        # 场景 ID + 4位小数位姿 → MD5 → 精确 ID
        p = self.state["pose"]
        pose_str = f"{p['tx']:.4f}_{p['ty']:.4f}_{p['tz']:.4f}_{p['rx']:.4f}_{p['ry']:.4f}_{p['rz']:.4f}"
        h = hashlib.md5(f"{self.state['scene_id']}|{pose_str}".encode()).hexdigest()[:12]
        return f"{self.state['scene_id']}_{h}"

    def bucket_key(self) -> str:
        return self.state["scene_id"]  # 同场景内才比较，跨场景不比较

    def is_similar_to(self, other) -> bool:
        # 位置距离 < 5cm 且每个角度偏差 < 5°，视为同一状态
        pa, pb = self.state["pose"], other.state["pose"]
        dist = math.sqrt((pa["tx"]-pb["tx"])**2 + (pa["ty"]-pb["ty"])**2 + (pa["tz"]-pb["tz"])**2)
        if dist > 0.05:
            return False
        return all(abs(pa[k]-pb[k]) <= 5.0 for k in ("rx", "ry", "rz"))
```

**举例：**

假设同一场景 `scene001` 有 3 个来自不同回合的位姿节点：
```
节点A: tx=1.000, ty=2.000, tz=3.000, rx=-90°, ry=0°, rz=45°
节点B: tx=1.003, ty=2.001, tz=2.999, rx=-90°, ry=0°, rz=45°  ← 差 3mm
节点C: tx=2.000, ty=2.000, tz=3.000, rx=-90°, ry=0°, rz=45°  ← 差 1m
```

去重结果：
- 节点A、B：`bucket_key = "scene001"`，`is_similar_to` → True（dist=0.003m < 5cm）→ **合并为同一节点**
- 节点C：`is_similar_to` → False（dist=1m > 5cm）→ **独立节点**

---

## 5. 如何添加图像环境

**你需要实现的文件：**

```
graphrl/envs/my_image_env/
├── __init__.py              ← 触发注册
├── env.py                   ← VAGEN gym 环境
├── vagen_graph_builder.py   ← 【核心】实现 traj_to_transitions + 可选自定义 NodeData
└── traj_to_sft.py           ← 实现 generate_datasets
```

### Step 1: 弄清楚你的环境产出什么

首先搞清楚 VAGEN 对话里的消息格式：
- 用户消息里的**状态**长什么样？是纯文本？还是含 `<image>` 占位符？
- 还有没有可解析的**符号表示**（位姿/坐标/标签）？→ 这将作为 `state` 字段用于去重
- 图片文件在 `rollout_dir/image_{step_idx}/images_{line_idx}/` 下的命名规则
- 助手消息里的**动作**用什么格式（`<answer>`/`<action>` 等）？

### Step 2: 定义自定义 NodeData（如需要）

**什么时候需要自定义 NodeData？**

| 状态类型 | 是否需要自定义 |
|---------|---------------|
| 文本/离散状态（棋盘、符号） | 不需要，直接用 `VagenNodeData` |
| 浮点位姿/坐标（需要容差匹配） | 需要，重写三个去重方法 |
| 纯图像（无符号标注） | 需要，`state` 用图片哈希，`is_similar_to` 用感知哈希 |

```python
# vagen_graph_builder.py

from graphrl.modules.rl.vagen.base.vagen_graph_builder_networkx import (
    VagenGraphBuilderNetworkx, VagenNodeData, VagenEdgeData,
)
from graphrl.modules.rl.vagen.base.vagen_graph_builder_base import graph_builder_registry

# 示例：带位姿的图像环境
class MyImageNodeData(VagenNodeData):
    """state = {"scene_id": str, "pose": {"x": float, "y": float, "heading": float}}"""

    def unique_key(self) -> str:
        p = self.state["pose"]
        raw = f"{self.state['scene_id']}|{p['x']:.4f}_{p['y']:.4f}_{p['heading']:.4f}"
        return hashlib.md5(raw.encode()).hexdigest()[:16]

    def bucket_key(self) -> str:
        return self.state["scene_id"]  # 同场景内比较

    def is_similar_to(self, other) -> bool:
        if not isinstance(other, MyImageNodeData):
            return False
        pa, pb = self.state["pose"], other.state["pose"]
        dist = math.sqrt((pa["x"]-pb["x"])**2 + (pa["y"]-pb["y"])**2)
        angle_diff = abs(pa["heading"] - pb["heading"])
        return dist < 0.1 and angle_diff < 5.0
```

### Step 3: 实现 `traj_to_transitions`

```python
@graph_builder_registry.register("my_image_env")
class MyImageEnvGraphBuilder(VagenGraphBuilderNetworkx):

    def traj_to_transitions(
        self,
        messages: List[Dict[str, str]],
        rollout_dir: Path,
        step_idx: int,
        line_idx: int,
    ) -> List[Tuple[NodeData, EdgeData, NodeData]]:

        image_base = rollout_dir / f"image_{step_idx}" / f"images_{line_idx}"
        global_img_idx = 0
        states = []   # List of (pose_dict, abs_image_path)
        actions = []  # List of action_str, aligned with states

        for msg in messages:
            role, content = msg["role"], msg["content"]

            if role == "user":
                # 1. 解析符号状态（用于去重和 obs_str）
                pose = _parse_pose(content)  # 返回 {"x": ..., "y": ..., "heading": ...}
                scene_id = _parse_scene_id(content)

                # 2. 定位图片文件
                num_images = content.count("<image>")
                img_path = None
                if num_images > 0:
                    for ext in (".png", ".jpg"):
                        candidate = image_base / f"{global_img_idx}{ext}"
                        if candidate.exists():
                            img_path = str(candidate)
                            break
                    global_img_idx += num_images

                if pose:
                    states.append((scene_id, pose, img_path))
                    actions.append(None)

            elif role == "assistant":
                action = _parse_action(content)
                if action and actions:
                    actions[-1] = action

        # 构建转移列表
        transitions = []
        for i in range(len(states) - 1):
            scene_id, pose_src, img_src = states[i]
            _, pose_dst, img_dst = states[i + 1]
            action = actions[i]
            if action is None:
                continue

            src = MyImageNodeData(
                state={"scene_id": scene_id, "pose": pose_src},
                obs_str=f"[x={pose_src['x']:.2f}, y={pose_src['y']:.2f}]",
                source_images=[img_src] if img_src else [],  # 框架自动复制
                extra={"scene_id": scene_id},
            )
            dst = MyImageNodeData(
                state={"scene_id": scene_id, "pose": pose_dst},
                obs_str=f"[x={pose_dst['x']:.2f}, y={pose_dst['y']:.2f}]",
                source_images=[img_dst] if img_dst else [],
                extra={"scene_id": scene_id},
            )
            transitions.append((src, VagenEdgeData(obs_str=action), dst))

        return transitions
```

### Step 4: 实现 `generate_datasets`

```python
# traj_to_sft.py
from graphrl.modules.traj_to_sft.base.base_graph_sft_generator import BaseGraphSFTGenerator, DatasetMap
from graphrl.modules.traj_to_sft.base import sft_generator_registry

@sft_generator_registry.register("my_image_env")
class MyImageEnvSFTGenerator(BaseGraphSFTGenerator):
    ENV_CONFIG_KEY = "my_image_env"

    def generate_datasets(self, graph, images_dir, output_dir, config) -> DatasetMap:
        records = []

        # 遍历所有边生成逆向动力学样本
        for u, v, edge_data in graph.G.edges(data=True):
            src_node = graph.G.nodes[u]
            dst_node = graph.G.nodes[v]

            # 图片路径（相对 graph/，SFT 样本可直接引用）
            src_images = src_node.get("image_paths", [])
            dst_images = dst_node.get("image_paths", [])

            records.append({
                "messages": [
                    {"role": "system", "content": "You are a navigation model..."},
                    {
                        "role": "user",
                        "content": [
                            # 图片内容（路径转为实际内容由 LLaMA-Factory 处理）
                            *[{"type": "image", "image": p} for p in src_images],
                            {"type": "text", "text": f"From: {src_node['obs_str']}\nTo: {dst_node['obs_str']}\nWhat action was taken?"}
                        ]
                    },
                    {"role": "assistant", "content": edge_data["obs_str"]}
                ]
            })

        return {"my_image_nav": (records, "sharegpt")}
```

### Step 5: 注册

```python
# __init__.py
from graphrl.envs.my_image_env.vagen_graph_builder import MyImageEnvGraphBuilder
from graphrl.envs.my_image_env.traj_to_sft import MyImageEnvSFTGenerator
```

`graphrl/configs/rl/vagen/env_registry.yaml`：
```yaml
env_registry:
  MyImageEnv: graphrl.envs.my_image_env.env.MyImageEnv
```

`examples/my_image_env/pipeline.yaml`：
```yaml
env_module: graphrl.envs.my_image_env
general_overrides:
  rl:
    graph_builder: my_image_env
  traj_to_sft:
    generator: my_image_env
```

---

### 5.x 图像环境实现 Checklist

添加图像环境时，需要确认的关键问题：

**关于状态表示：**
- [ ] 消息里有没有可以解析的**符号状态**（位姿、坐标、场景 ID）？→ 用它作 `state`
- [ ] 如果完全没有符号标注，用图片文件哈希作 `state`（去重时按哈希）
- [ ] `obs_str` 应包含什么信息（文本描述？位姿字符串？）

**关于去重策略：**
- [ ] 状态是离散的（完全匹配）还是连续的（需要容差）？
- [ ] 不同回合的相同位置应该合并吗？→ 定义 `is_similar_to` 的容差
- [ ] 不同场景的节点需要隔离吗？→ `bucket_key` 返回场景 ID

**关于图片文件：**
- [ ] 每条消息的图片数量？（数 `<image>` 个数）
- [ ] 图片在 rollout 目录的路径格式？（默认 `image_{step_idx}/images_{line_idx}/{idx}.png`）
- [ ] 需要图片质量过滤吗？（参考 ViewSuite 的 `_filter_graph`）

**关于 SFT 数据：**
- [ ] 要生成哪些训练任务？（正向/逆向动力学、导航、分类等）
- [ ] SFT 样本如何引用图片？（`image_paths` 是相对于 `graph/` 的路径）

---

## 6. 配置与运行

### 三层配置合并

```
默认配置（configs/*.yaml）
  + general_overrides（对所有轮次）
  + iteration_overrides[N]（第 N 轮特定配置）
  + 控制器注入路径（model_path, output_dirs）← 最高优先级
```

### pipeline.yaml 关键字段

```yaml
experiment_dir: /path/to/experiments
initial_model_path: Qwen/Qwen2.5-7B-Instruct
iterations: 3

env_module: graphrl.envs.my_image_env  # 触发 __init__.py 注册

rl_module: vagen
sft_module: llama_factory
traj_to_sft_module: default

general_overrides:
  rl:
    graph_builder: my_image_env    # 对应 @graph_builder_registry.register("my_image_env")
    hydra_overrides:
      data:
        train_files: /abs/path/to/train.yaml
  traj_to_sft:
    generator: my_image_env        # 对应 @sft_generator_registry.register("my_image_env")
```

### 断点续跑

控制器自动检测已完成阶段：
- `iter_XXX/rl_model/config.json` 存在 → RL 阶段完成，跳过
- `iter_XXX/sft_data/dataset_info.json` 存在 → 轨迹转换完成，跳过
- `iter_XXX/sft_model/config.json` 存在 → SFT 阶段完成，跳过

---

## 参考实现

| 环境 | 路径 | 特点 |
|------|------|------|
| Sokoban（文本） | [graphrl/envs/sokoban_text/](graphrl/envs/sokoban_text/) | 最简单，文本状态，默认 VagenNodeData |
| ViewSuite（图像+位姿） | [graphrl/envs/viewsuite_active_explore/](graphrl/envs/viewsuite_active_explore/) | 图像状态，自定义 NodeData，图片质量过滤 |

**核心代码位置：**
- `VagenNodeData`/`VagenEdgeData` + `traj_to_transitions` 抽象：[graphrl/modules/rl/vagen/base/vagen_graph_builder_networkx.py](graphrl/modules/rl/vagen/base/vagen_graph_builder_networkx.py)
- 图数据结构 `BaseGraph`：[graphrl/modules/rl/vagen/base/base_graph.py](graphrl/modules/rl/vagen/base/base_graph.py)
- SFT 生成器基类：[graphrl/modules/traj_to_sft/base/base_graph_sft_generator.py](graphrl/modules/traj_to_sft/base/base_graph_sft_generator.py)
