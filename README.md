<h1 align="center">VAGEN: Reinforcing World Model Reasoning for Multi-Turn VLM Agents</h1>
<!-- <p align="center" style="font-size: 30px;">
  <b>Training VLM agents with multi-turn reinforcement learning</b>
</p>
<p align="center" style="font-size: 10px;">
  <b>NeurIPS 2025</b>
</p> -->
<h3 align="center"><b>Training VLM agents with multi-turn reinforcement learning</b></h3>
<h4 align="center"><b>🔥 NeurIPS 2025 🔥</b></h4>

<p align="center" style="font-size: 16px;">
  Kangrui Wang*, Pingyue Zhang*, Zihan Wang*, Yaning Gao*, Linjie Li*, Qineng Wang, Hanyang Chen, Chi Wan, Yiping Lu, Zhengyuan Yang, Lijuan Wang, Ranjay Krishna, Jiajun Wu, Li Fei-Fei, Yejin Choi, Manling Li
</p>
<p align="center" style="font-size: 12px;"><i>(* equal contribution)</i></p>

<p align="center">
  <a href="https://arxiv.org/abs/2510.16907"><img src="https://img.shields.io/badge/📜_Paper-B31B1B?style=for-the-badge&logo=arXiv&logoColor=white" alt="Paper"></a>
  <a href="https://vagen.readthedocs.io/en/latest"><img src="https://img.shields.io/badge/📚_Documentation-4285F4?style=for-the-badge&logoColor=white" alt="Documentation"></a>
  <a href="https://mll-lab.notion.site/vagen"><img src="https://img.shields.io/badge/📝_Blog-FF5722?style=for-the-badge&logoColor=white" alt="Blog"></a>
  <a href="https://wandb.ai/ragen-V/vagen-final/reports/VAGEN-Experimental-Results--VmlldzoxMzM2NzczNA?accessToken=c9539vj7s3yxh8qu4rykmgi1kz47935mu9pvkind70m2tt6bdin6tx263ec7yqei"><img src="https://img.shields.io/badge/📊_Experiment_Log-FB8C00?style=for-the-badge&logoColor=white" alt="Experiment Log"></a>
  <a href="https://vagen-ai.github.io/"><img src="https://img.shields.io/badge/🌐_Website-00C851?style=for-the-badge&logoColor=white" alt="Website"></a>
</p>

**VAGEN** is a **reinforcement learning (RL) framework that trains multi-turn VLM agents** (vision-language model agents) to build an internal **world model** through explicit visual state reasoning. Instead of rewarding only task success, VAGEN reinforces the agent's world model reasoning itself, decomposed into **StateEstimation** ("what is the current state?") and **TransitionModeling** ("what comes next?"), with a turn-level **WorldModeling Reward** (LLM-as-Judge) and **Bi-Level GAE** for turn-aware credit assignment. Combining world models with reinforcement learning, a 3B VLM trained with VAGEN scores **0.82** across five visual agent benchmarks, a 3x improvement over its untrained backbone (0.21), outperforming GPT-5 (0.75), Gemini 2.5 Pro (0.67), and Claude 4.5 (0.62).

<div style="width:100%; overflow-x:auto;">
  <table style="width:100%;">
    <tr>
      <td align="center" style="width:20%;"><br>
        <img src="https://github.com/user-attachments/assets/6d72800a-9b4d-45ec-b528-ac81efb93966" style="width:72%;"/><br>
        <img src="https://github.com/user-attachments/assets/6f283f99-fa15-4e26-9f99-6649a7d72374" style="width:72%;"/><br>
        <b>FrozenLake</b>
      </td>
      <td align="center" style="width:20%;"><br>
        <img src="https://github.com/user-attachments/assets/b364e6c9-4c2c-46d0-afca-ee42f271c59c" style="width:75%;"/><br>
        <img src="https://github.com/user-attachments/assets/65662eb0-9440-4555-9436-8b9272791ac4" style="width:75%;"/><br>
        <b>Navigation</b>
      </td>
      <td align="center" style="width:20%;"><br>
        <img src="https://github.com/user-attachments/assets/145352b5-3a9e-4248-bb94-d3fa46e6c493" style="width:80%;"/><br>
        <img src="https://github.com/user-attachments/assets/676de052-37d6-4c99-a7eb-200a58d11ed4" style="width:80%;"/><br>
        <b>Sokoban</b>
      </td>
      <td align="center" style="width:20%;"><br>
        <img src="https://github.com/user-attachments/assets/c597f17d-5c62-4319-bdaa-b7fa8e4564e1" style="width:80%;"/><br>
        <img src="https://github.com/user-attachments/assets/f61ea55c-ea79-4ead-9345-45be06d24e81" style="width:80%;"/><br>
        <b>ManiSkill</b>
      </td>
      <td align="center" style="width:20%;"><br>
        <img src="https://github.com/user-attachments/assets/8646da5f-69be-4283-a078-969f9b8f3f3b" style="width:92%;"/><br>
        <img src="https://github.com/user-attachments/assets/691b896a-ce30-4acc-ac49-af2d89452bdd" style="width:92%;"/><br>
        <b>SVG</b>
      </td>
    </tr>
  </table>
</div>

We introduce **VAGEN**, a multi-turn reinforcement learning framework designed specifically for training vision-language model (VLM) agents. Built upon this framework, we propose **World Modeling RL**, a novel reinforcement learning approach that significantly improves the multi-turn performance of VLMs by explicitly supervising their worldmodel reasoning process, as shown in **Figure&nbsp;1**.

We frame multi-turn VLM agentic tasks as a Partially Observable Markov Decision Process (POMDP), shown in **Figure&nbsp;2**.
| <img src="https://github.com/user-attachments/assets/834b32fa-9bfc-4e0f-a148-99cd6fc3141e" alt="Framework Overview" height="260"> | <img src="https://github.com/user-attachments/assets/d99ee757-ecd1-433c-8a6d-981bf383748e" alt="POMDP Formulation" height="260"> |
|:--:|:--:|
| <sub><b>Figure 1.</b> Overview of the VAGEN framework.</sub> | <sub><b>Figure 2.</b> POMDP formulation of multi-turn VLM agentic tasks.</sub> |




## News
**[2026/08]**: Added support for Verl 0.9.0, introduced compaction RL, and decoupled the harness layer.
- **Compaction RL** — a multi-turn paradigm alongside concat and no-concat. Turns accumulate until a token budget is reached, then the conversation is summarised and reopened from that summary, so an episode longer than the context window still trains as one trajectory. See [Multi-turn Compacted Training](#multi-turn-compacted-training). Reference: ([CompactionRL](https://arxiv.org/abs/2607.05378))
- **Environment, harness, training backend and algorithm are decoupled.** Anything subclassing `BaseEnv` or `BaseHarness` plugs into both training and evaluation without either being modified — see [Custom Harness](#custom-harness) and [Custom Environment](#custom-environment).

**[2026/02]** We have migrated the `main` branch to VAGEN-Lite, a lightweight and clean reimplementation built on VERL agent-loop for easy customization and stable performance. For the previous full-featured release, please visit the [vagen-legacy](https://github.com/mll-lab-nu/VAGEN/tree/vagen-legacy) branch.

**[2025/12]** Introducing [VAGEN-Lite](https://github.com/mll-lab-nu/VAGEN/tree/vagen-lite): a lightweight and clean reimplementation of VAGEN, built on the VERL agent-loop for easy customization and stable performance.

**[2025/09]** VAGEN is accepted by Neurips 2025

**[2025/04]** We've introduced a new modular design for environments and services in VAGEN:
- Enhanced environment framework for easier creation of custom environments
- New service architecture for efficient distributed training
- Check out our new guides:
  - [Creating Environments](./docs/custom-environment.md): New environment protocol.
  - [Creating Services](./vagen/envs_remote/README.md): We now support hosting environments in a separate process

**[2025/03]** We release VAGEN, a multi-turn reinforcement learning framework for training VLM Agents!

## Installation

```bash
conda create -n vagen python=3.12 -y
conda activate vagen

git clone --recursive https://github.com/mll-lab-nu/VAGEN.git
cd VAGEN
bash scripts/install.sh
```

`scripts/install.sh` fetches the pinned verl submodule, installs VAGEN with a rollout
engine, then verl, and checks the result. It is idempotent, so it is safe to re-run.
`SKIP_ENGINE=1` installs VAGEN without an engine if you already have one.

vLLM is the default and the verified path. For SGLang:

```bash
BACKEND=sglang bash scripts/install.sh
```

**Use one engine per environment.** They are mutually exclusive, and not by preference:
each pins a different `flashinfer` patch version, so pip refuses to install them together.
Use two conda environments if you want both.

<details>
<summary>Doing it by hand</summary>

```bash
git submodule update --init --recursive   # verl, pinned; the scripts will not run without it

pip install -e ".[vllm]"                  # or ".[sglang]" -- pick one, never both
pip install --no-deps -e ./verl           # --no-deps: verl's pins would undo the line above
pip install accelerate codetiming datasets dill hydra-core numpy pandas peft pyarrow \
            pybind11 pylatexenc ray tensordict torchdata wandb
```

The engine, `torch` and `transformers` versions all live in `setup.py`'s
`extras_require`, so there is one place that says which versions go together.

No `flash-attn` step: it publishes no wheel past torch 2.9, so on a newer torch installing
it means a source build. `transformers[kernels]`, which the extras pull in, instead fetches
a prebuilt `kernels-community/flash-attn2` from the Hub on first use.

verl is imported from the checkout rather than from PyPI, and the training scripts find
it at `VAGEN/verl` (the submodule) or `../verl` (a sibling checkout), in that order. Set
`VERL=/path/to/verl` to override.
</details>

**Environments in this repository** — `vagen/configs/env_registry.yaml` is the list that
matters: `Sokoban`, `FrozenLake`, `SpatialGym`, `PrimitiveSkill` (ManiSkill), and
`RemoteEnv`, which is how `Navigation` runs. The five benchmarks pictured above are the
paper's; SVG is not part of this release.

Some need their own setup: [spatial_gym](vagen/envs/spatial_gym/README.md) (dataset
download, plus `matplotlib` and `scipy` from its requirements.txt — without them the
registry drops the environment and you get `KeyError: Unknown env name: SpatialGym`),
[navigation](vagen/envs/navigation/README.md) (AI2-THOR),
[primitive_skill](vagen/envs/primitive_skill/README.md) (ManiSkill).


## Quick Start

`trainer.logger` defaults to `["console", "wandb"]` and no example script overrides it, so
log in first — or turn it off:

```bash
wandb login                       # or: export WANDB_MODE=offline
                                  # or append: trainer.logger=[console]
```

### Training
VAGEN currently supports PPO / GRPO with three multi-turn training paradigms. An episode is
many turns; a training row is one conversation. The **harness** decides how the first maps
onto the second, and it is the central choice:

| `trainer.harness` | one episode becomes | use when |
|---|---|---|
| `concat` | **one** row holding every turn | the default; the episode fits the response region |
| `no_concat` | **one row per turn**, each a fresh conversation | history is not needed, or will not fit |
| `compact` | a row per conversation, summarised and reopened when full | long episodes that must keep context ([CompactionRL](https://arxiv.org/abs/2607.05378)) |


#### Multi-turn Concatenated Training

All turns in a trajectory are concatenated into a single training instance.

```bash
# Qwen/Qwen2.5-VL-3B-Instruct
cd VAGEN
bash examples/train/sokoban/train_default_gae_qwen25vl3b.sh
```

```bash
# Qwen/Qwen3-VL-4B-Instruct
# needs transformers>=5.2.0 (pinned in setup.py; the engine extras pin ==5.12.1)
cd VAGEN
bash examples/train/sokoban/train_grpo_qwen3vl4b.sh
```

```bash
# Enable reward variance based top-p filtering
cd VAGEN
bash examples/train/frozenlake/train_grpo_qwen25vl3b_filtertopp_vision.sh
```


#### Multi-turn Non-Concatenated Training

Each trajectory is split into multiple turn-level training instances.

```bash
cd VAGEN
bash examples/train/sokoban/train_ppo_no_concat_qwen25vl3b.sh
```

#### Multi-turn Compacted Training

Turns are concatenated until a token budget is reached, then summarised so the next conversation starts from the summary. One training instance per conversation.

```bash
cd VAGEN
bash examples/train/sokoban/train_default_gae_compact_qwen25vl3b.sh
```

The paradigm is chosen with `trainer.harness=concat|no_concat|compact`, and it is independent of `algorithm.adv_estimator` — the harness decides how an episode is laid out in rows and the estimator stitches those rows back into one trajectory. Note that verl's own `gae`/`grpo` score a row at a time, so they are only correct under `concat`; the trainer refuses the other two rather than training on truncated trajectories.

```bash
# LoRA. peft raises on an outdated torchao rather than skipping it, so
# a too-old version breaks LoRA even though nothing here quantises:
# pip install "torchao>=0.16.0"   (or uninstall torchao entirely)
cd VAGEN
bash examples/train/frozenlake/train_ppo_no_concat_lora_qwen25vl3b.sh
```
### Evaluation

VAGEN supports evaluation using different backends (OpenAI, Claude, Gemini, sglang, vLLM). For details, see [vagen/evaluate/README.md](vagen/evaluate/README.md).

```bash
cd VAGEN
# Sokoban with a local vLLM server -- starts one, evaluates, and shuts it down.
# vLLM is the engine `scripts/install.sh` gives you by default.
MODEL_PATH=Qwen/Qwen2.5-VL-3B-Instruct \
  bash examples/evaluate/sokoban/vllm/eval_qwen25_vl_3b.sh
```

The context policy is a config key, the same one training uses, so comparing them is one
override:

```bash
bash examples/evaluate/sokoban/vllm/eval_qwen25_vl_3b.sh 'envs.0.harness=no_concat'
```

Every environment ships a vLLM launcher:

| environment | launcher |
|---|---|
| Sokoban | `examples/evaluate/sokoban/vllm/eval_qwen25_vl_3b.sh` |
| FrozenLake | `examples/evaluate/frozenlake/vllm/eval_qwen25_vl_3b.sh` |
| Navigation | `examples/evaluate/navigation/vllm/eval_qwen25_vl_7b.sh` |
| PrimitiveSkill | `examples/evaluate/primitive_skill/vllm/eval_qwen25_vl_3b.sh` |
| SpatialGym | `examples/evaluate/spatial_gym/vllm/eval_qwen25_vl_3b.sh` |

Navigation and PrimitiveSkill also need their own environment server running
(`python -m vagen.envs.<env>.serve --port 8000`); the launcher checks and says so.
SpatialGym needs its room dataset — same, see `vagen/envs/spatial_gym/README.md`.

```bash
cd VAGEN
# Against any OpenAI-compatible endpoint already running (set OPENAI_API_KEY first --
# the shipped config defaults to gpt-4o-mini, which is a paid remote call).
bash examples/evaluate/sokoban/run_eval.sh
```

<details>
<summary>With sglang instead</summary>

Requires the sglang extra, which is mutually exclusive with vLLM — see Installation.

```bash
bash examples/evaluate/frozenlake/sglang/eval_qwen25_vl_3b.sh
```
</details>

## Customizing Your Environment

To train on your own environment, follow the steps below.

> A ★ in a comment, in the code or in a config, marks something that was got wrong once —
> the note says what the failure looked like rather than what the line does. Worth reading
> before changing the line it sits on.

### 1. Create Your Environment Class

* Use `GymImageEnv` as the base class:

  * [`vagen/envs/gym_image_env.py`](vagen/envs/gym_image_env.py)
* Refer to Sokoban for a full implementation example:

  * [`vagen/envs/sokoban/sokoban_env.py`](vagen/envs/sokoban/sokoban_env.py)


### 2. Register the Environment

Add your environment entry to:

```yaml
vagen/configs/env_registry.yaml
```

### 3. Create Configuration Files

Prepare training and validation configs:

* `train.yaml`
* `val.yaml`

You can follow the Sokoban examples as templates:

* [`examples/train/sokoban/train_sokoban_vision.yaml`](examples/train/sokoban/train_sokoban_vision.yaml)
* [`examples/train/sokoban/val_sokoban_vision.yaml`](examples/train/sokoban/val_sokoban_vision.yaml)


### 4. Create a Training Script

Write your training script based on:

* [`examples/train/sokoban/train_default_gae_qwen25vl3b.sh`](examples/train/sokoban/train_default_gae_qwen25vl3b.sh)


## Custom Harness

A **harness** is the context policy: each turn it answers one question — does the next
model call continue the current conversation, or start a new one, and seeded with what?
`concat`, `no_concat` and `compact` are three answers to that question, not three
mechanisms.

The design VAGEN is built around: **anything that subclasses `BaseEnv` or `BaseHarness`
plugs into training and evaluation without either being modified.** A harness holds no
tokenizer, no client and no environment, and never sees a token, a mask or a reward — so the
same object drives a training rollout and a closed-API evaluation. Nothing in it assumes the
conversation lives client-side either, so a backend with server-side continuation
(`previous_response_id`, an SGLang session, a vLLM prefix cache) can be added without
touching the policy. Today every shipped backend re-sends the messages.

```python
from vagen.core.harness import BaseHarness, Call
from vagen.harness import register_harness

@register_harness("mine")
class MyHarness(BaseHarness):
    #: Whether one episode can end up in more than one row. The trainer asks the harness
    #: rather than keeping a list of the ones it knows, and pairs the estimator accordingly.
    splits_episode_across_rows = True

    def next_call(self) -> Call: ...       # the only required method
```

Both places read the same key, and both accept either a registered name or an import path
— but only if something has imported your module first, which differs between them:

```yaml
# training -- vagen/configs/vagen_multiturn.yaml, or a -o override
trainer:
  harness: mine
  # verl builds a registry per worker process, so the module has to be imported inside each
  # one. That is what this is for; without it the decorator never runs and the name is
  # unknown:
  #   actor_rollout_ref.model.external_lib=mypkg.harnesses
```

```yaml
# evaluation -- examples/evaluate/<env>/config.yaml
envs:
  - name: Sokoban
    harness: mypkg.harnesses:MyHarness   # an import path, not a bare name
```

★ Use the import path for evaluation. `run_eval` has no `external_lib` hook, so nothing
imports your module and a bare `harness: mine` fails with
`unknown harness 'mine'; choose from ['compact', 'concat', 'no_concat']`. This is also why
the import path exists at all: a new policy is usually tried in evaluation first, where the
config is a yaml and there is nowhere a decorator would have run.

Full contract and the budget hooks: [`vagen/core/harness.py`](vagen/core/harness.py); the
three implementations: [`vagen/harness/`](vagen/harness/).

## More Customization

See the [Documentation](https://vagen.readthedocs.io/) for more customization options:

- [Custom Filter](https://vagen.readthedocs.io/en/latest/custom-filter/) — Trajectory filtering (e.g., Reward Variance (RV) filter in [RAGEN](https://github.com/RAGEN-AI/RAGEN))
- [Custom Metric](https://vagen.readthedocs.io/en/latest/custom-metric/) - Add W&B logging metrics
- [Configuration](https://vagen.readthedocs.io/en/latest/configuration/) - Training configuration reference

## Useful Configs
refer to `vagen/configs/vagen_multiturn.yaml`

### No Concat Mode
```yaml
# Enable no concat mode: input is system prompt + current step observation
trainer:
  harness: no_concat        # concat | no_concat | compact

# no_concat and compact put one episode in several rows, so the advantage estimator has
# to be one that stitches them back together. verl's own `gae`/`grpo` score a row at a
# time and would drop every turn's credit at the row boundary; the trainer refuses that
# pairing at startup rather than training on it.
algorithm:
  adv_estimator: default_gae       # or bi_level_gae | turn_level_gae | token_level_gae
                                   #    | trajectory_grpo
  # default_gae is the vanilla baseline: the episode's whole reward lumped onto its
  # last token, which is what single-turn RLHF does. It stitches rows like the others,
  # so it stays comparable under no_concat and compact where verl's `gae` would not.

```

### Image Logging
```yaml
# Warning:
# - If you set a training-data rollout dir AND enable image logging, training images will also be dumped to disk.
#   This can consume a large amount of storage very quickly. Monitor disk usage and consider cleanup/limits.
trainer:
  log_image:
    enable: false      # true can enable saving rollout/validation images to disk
    max_pending: 2     # max concurrent async image dump tasks
    png_compress_level: 0  # PNG compression (0 = fastest, 9 = smallest)
```

### HuggingFace Hub Upload
```yaml
# export HF_TOKEN=xxx
huggingface_hub:
  hf_save_freq: null   # upload every N steps (must be a multiple of trainer.save_freq); null = disabled
  repo_id: vagen-training   # the shipped default; enabling upload with it unchanged
                          # pushes to a repo of that name under your account        
  private: false        
```

### Training Data Filtering
```yaml

filter:
  name: reward_variance_top_p # refer to vagen/custom_filter
  filter_kwargs: 
    top_p: 0.9 
  enable: False # set to true to enable filtering, recommended for grpo trainining
```


## Known Issues & Fixes
See [docs/issues.md](docs/issues.md)

## Citation

If you find our framework and paper useful, we appreciate it if you could cite our work:

```bibtex
@inproceedings{wang2025vagen,
  title={VAGEN: Reinforcing World Model Reasoning for Multi-Turn VLM Agents},
  author={Kangrui Wang and Pingyue Zhang and Zihan Wang and Yaning Gao and Linjie Li and Qineng Wang and Hanyang Chen and Chi Wan and Yiping Lu and Zhengyuan Yang and Lijuan Wang and Ranjay Krishna and Jiajun Wu and Li Fei-Fei and Yejin Choi and Manling Li},
  booktitle={The Thirty-ninth Annual Conference on Neural Information Processing Systems},
  year={2025},
  url={https://arxiv.org/abs/2510.16907}
}
```
