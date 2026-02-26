# SearchR1 Environment

Train search agents using dense retrieval on Wikipedia with pre-built E5 embeddings. The agent learns to iteratively search a corpus and produce a final answer via multi-turn RL (PPO).

## Architecture

```text
Agent (VLM)                        Retrieval Server
    │                                     │
    ├── <search>query</search>  ────────► │ E5 Encoder → FAISS Index
    │                                     │      │
    │ ◄──── retrieved documents ──────────┘      │
    │                                        Wikipedia
    ├── <search>query</search>  ────────►    Corpus
    │ ◄──── retrieved documents ──────────┘
    │
    └── <final>answer</final>
```

## Quick Setup

### 1. Download Data

```bash
cd vagen/envs/search
python download_search_data.py --data_dir ./search_data
```

Then concatenate the index shards:

```bash
cd search_data/prebuilt_indices
cat part_aa part_ab > e5_Flat.index
```

This downloads:

- **Wikipedia corpus** from [PeterJinGo/wiki-18-corpus](https://huggingface.co/datasets/PeterJinGo/wiki-18-corpus) → `search_data/wikipedia/wiki-18.jsonl`
- **Pre-built E5 dense index** from [PeterJinGo/wiki-18-e5-index](https://huggingface.co/datasets/PeterJinGo/wiki-18-e5-index) → `search_data/prebuilt_indices/`

### 2. Set Up Retrieval Server Environment

```bash
conda create -n retriever python=3.10
conda activate retriever
conda install pytorch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 pytorch-cuda=12.1 -c pytorch -c nvidia
pip install transformers datasets pyserini
conda install -c pytorch -c nvidia faiss-gpu=1.8.0
pip install uvicorn fastapi
```

### 3. Launch Retrieval Server

```bash
cd vagen/envs/search
bash retrieval/launch_server.sh
```

### 4. Train

TBD

### 5. Evaluate

```bash
cd examples/evaluate/search && bash eval_qwen25_vl_3b.sh
```

## Environment Details

### Action Format

The agent produces one action per turn using XML tags:

```xml
<search>your search query</search>    <!-- retrieve documents -->
<final>your answer</final>            <!-- submit final answer -->
```

JSON format is also accepted:

```json
{"action": "search", "query": "your search query"}
{"action": "final", "answer": "your answer"}
```

### Observation Format

Each turn, the agent sees:

```text
Question:
Who wrote The Old Man and the Sea?

Budgets: remaining_searches=3 remaining_steps=7

Retrieved evidence (top snippets so far):
[1] id=doc_42 score=0.892
The Old Man and the Sea is a short novel written by Ernest Hemingway...

Respond with either <search>...</search> or <final>...</final>.
```

### Configuration

| Parameter | Default | Description |
| --------- | ------- | ----------- |
| `dataset` | — | Dataset name (`"hotpotqa"`) or list of episodes |
| `mode` | `"test"` | `"train"` or `"test"` split |
| `retrieval_server_url` | `http://127.0.0.1:8000` | Retrieval server endpoint |
| `max_steps` | `8` | Maximum interaction turns |
| `max_searches` | `4` | Maximum search queries per episode |
| `top_k` | `5` | Documents returned per search |
| `max_evidence_items` | `15` | Maximum evidence snippets in prompt |
| `format_reward` | `0.01` | Reward for valid action format |
| `search_penalty` | `-0.01` | Penalty per search (encourages efficiency) |
| `correct_reward` | `1.0` | Reward for correct final answer |
| `wrong_reward` | `0.0` | Reward for wrong final answer |
| `strict_format` | `False` | End episode on format error |

### Reward Structure

| Event | Reward |
| ----- | ------ |
| Valid format | +0.01 |
| Search query | -0.01 |
| Correct answer | +1.0 |
| Wrong answer | 0.0 |

Answer correctness is determined by normalized exact match (lowercased, whitespace-collapsed, punctuation-removed).

### Dataset

The default dataset is [HotpotQA](https://hotpotqa.github.io/) (distractor setting). Each episode contains:

```python
{"question": "...", "ground_truth": "...", "data_source": "hotpotqa"}
```

You can also pass a custom dataset as a list of dicts with `question` and `answer` (or `ground_truth`) fields.
