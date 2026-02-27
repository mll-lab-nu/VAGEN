# WebArena Environment for VAGEN

This environment integrates the [WebArena](https://webarena.dev/) benchmark into VAGEN, enabling multi-turn RL training of VLM agents on realistic web navigation tasks.

## Prerequisites

### 1. Install Python Dependencies

```bash
pip install playwright
playwright install chromium
```

### 2. Launch WebArena Docker Containers

WebArena requires several web services running as Docker containers. Pull and start them:

```bash
# Shopping site (Magento)
docker run -d -p 7770:80 --name shopping ghcr.io/web-arena-x/shopping-final:latest

# Shopping admin panel
docker run -d -p 7780:80 --name shopping-admin ghcr.io/web-arena-x/shopping-admin-final:latest

# Reddit (Postmill)
docker run -d -p 9999:80 --name reddit ghcr.io/web-arena-x/postmill-final:latest

# GitLab
docker run -d -p 8023:8023 --name gitlab ghcr.io/web-arena-x/gitlab-final:latest

# Wikipedia
docker run -d -p 8888:80 --name wiki ghcr.io/web-arena-x/wikipedia-final:latest

# Map (OpenStreetMap)
docker run -d -p 443:443 --name map ghcr.io/web-arena-x/map-final:latest

# Homepage (task hub)
docker run -d -p 4399:4399 --name homepage ghcr.io/web-arena-x/homepage-final:latest
```

Verify containers are running:
```bash
docker ps --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"
```

### 3. Download Task Configs

Clone the WebArena repo and get the task definition files:

```bash
git clone https://github.com/web-arena-x/webarena.git
# Task files are at: webarena/config_files/test.raw.json
```

The task JSON contains ~800 tasks across different websites.

## Configuration

In your training YAML, configure the environment:

```yaml
envs:
  - name: WebArena
    n_envs: 500
    data_source: webarena
    seed: [0, 500, 1]
    max_turns: 15
    response_length_per_turn: 1024
    config:
      task_file: "path/to/webarena/config_files/test.raw.json"
      render_mode: vision    # or "text" for accessibility tree
      max_steps: 15
      viewport_width: 1280
      viewport_height: 720
      # Server URLs (if not using default ports)
      shopping_url: "http://localhost:7770"
      reddit_url: "http://localhost:9999"
      gitlab_url: "http://localhost:8023"
```

### Key Config Options

| Parameter | Default | Description |
|-----------|---------|-------------|
| `task_file` | `""` | Path to WebArena task JSON |
| `task_ids` | `None` | Filter to specific task IDs |
| `render_mode` | `"vision"` | `"vision"` (screenshot) or `"text"` (accessibility tree) |
| `max_steps` | `15` | Max browser actions per episode |
| `format_reward` | `0.01` | Reward for valid action format |
| `success_reward` | `1.0` | Reward for task completion |
| `step_penalty` | `0.0` | Per-step penalty (set negative to encourage efficiency) |
| `headless` | `True` | Run browser headlessly |

## Action Format

The agent uses XML tags:

```xml
<think>Reasoning about what to do next.</think>
<action>click[42]</action>
```

Available actions:
- `click[element_id]` - Click an element
- `type[element_id][text]` - Type into an input field
- `scroll[up|down]` - Scroll the page
- `goto[url]` - Navigate to URL
- `go_back` - Browser back button
- `stop[answer]` - Complete the task

## Quick Test

```bash
# Make sure Docker containers are running, then:
python -m vagen.envs.webarena.webarena_env
```

This launches an interactive session where you can manually test actions.
