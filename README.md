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
  Kangrui Wang*, Pingyue Zhang*, Zihan Wang*, Yaning Gao*, Linjie Li*, Qineng Wang, Hanyang Chen, Chi Wan, Yiping Lu, Zhengyuan Yang, Lijuan Wang, Ranjay Krishna, Jiajun Wu, Li Fei-Fei, Yejin Choi, <a href="https://limanling.github.io/">Manling Li</a>
</p>
<p align="center" style="font-size: 12px;"><i>(* equal contribution)</i></p>

<p align="center">
  <a href="https://arxiv.org/abs/2510.16907"><img src="https://img.shields.io/badge/📜_Paper-B31B1B?style=for-the-badge&logo=arXiv&logoColor=white" alt="Paper"></a>
  <a href="https://vagen.readthedocs.io/en/latest"><img src="https://img.shields.io/badge/📚_Documentation-4285F4?style=for-the-badge&logoColor=white" alt="Documentation"></a>
  <a href="https://mll-lab.notion.site/vagen"><img src="https://img.shields.io/badge/📝_Blog-FF5722?style=for-the-badge&logoColor=white" alt="Blog"></a>
  <a href="https://wandb.ai/ragen-V/vagen-final/reports/VAGEN-Experimental-Results--VmlldzoxMzM2NzczNA"><img src="https://img.shields.io/badge/📊_Experiment_Log-FB8C00?style=for-the-badge&logoColor=white" alt="Experiment Log"></a>
  <a href="https://vagen-ai.github.io/"><img src="https://img.shields.io/badge/🌐_Website-00C851?style=for-the-badge&logoColor=white" alt="Website"></a>
</p>

**VAGEN** is a **reinforcement learning (RL) framework that trains multi-turn VLM agents** (vision-language model agents) to build an internal **world model** through explicit visual state reasoning. In addition to task success, VAGEN can reinforce **StateEstimation** ("what is the current state?") and **TransitionModeling** ("what comes next?") with an LLM-as-Judge world-modeling reward. The reusable harness, environment, model, rollout, and evaluation boundaries keep those signals independent from the selected training algorithm.

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
**[2026/08]**: Added support for Verl 0.9.0, decoupled the environment and harness layer. This is a major update. The last commit before these changes is [here](https://github.com/mll-lab-nu/VAGEN/tree/vagen-lite-verl-v0.6.1-final); please check out the tag if you need the previous codebase.

**[2026/02]** We have migrated the `main` branch to VAGEN-Lite, a lightweight and clean reimplementation built on VERL agent-loop for easy customization and stable performance. For the previous full-featured release, please visit the [vagen-legacy](https://github.com/mll-lab-nu/VAGEN/tree/vagen-legacy) branch.

**[2025/12]** Introducing [VAGEN-Lite](https://github.com/mll-lab-nu/VAGEN/tree/vagen-lite): a lightweight and clean reimplementation of VAGEN, built on the VERL agent-loop for easy customization and stable performance.

**[2025/09]** VAGEN is accepted by Neurips 2025

**[2025/04]** We've introduced a new modular design for environments and services in VAGEN:
- Enhanced environment framework for easier creation of custom environments
- New service architecture for efficient distributed training
- Check out our new guides:
  - [Creating Environments](./docs/custom-environment.md): New environment protocol.
  - [Creating Services](./vagen/envs/_common/remote/README.md): We support hosting environments in a separate process

**[2025/03]** We release VAGEN, a multi-turn reinforcement learning framework for training VLM Agents!

## Installation

```bash
conda create -n vagen python=3.12 -y
conda activate vagen

git clone --recursive --branch release-ready https://github.com/JamesKrW/VAGEN.git
cd VAGEN
bash scripts/install.sh
```

`scripts/install.sh` fetches the pinned verl submodule, installs VAGEN with a rollout
engine, then verl, and checks the result. It is idempotent, so it is safe to re-run.
`SKIP_ENGINE=1` installs VAGEN without an engine if you already have one.

vLLM is the default and the verified training path. For SGLang evaluation and local
serving:

```bash
BACKEND=sglang bash scripts/install.sh
```

Installing the SGLang extra does **not** switch the shipped training launchers: they source
`baseline_vllm.flags` and still select vLLM. Use the SGLang evaluation launchers where one
is provided; a training launcher needs explicit SGLang rollout configuration and its own
model-level validation.

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

Environment responses use one shared protocol. Structured world-model output is
`<perception>...</perception><reasoning>...</reasoning><prediction>...</prediction><answer>...</answer>`;
the compact reasoning form is `<think>...</think><answer>...</answer>`. Native-thinking
models can use Sokoban's `wm_think` mode, which permits their model-owned thinking block
before the same structured suffix. See [Configuration](docs/configuration.md#prompt_format).


## Quick Start

```bash
wandb login
# Self-hosted W&B:
# WANDB_BASE_URL=https://your-wandb-host wandb login --host https://your-wandb-host
```

### Training

```bash
cd VAGEN

# Qwen2.5-VL: concat / no-concat / compact
bash examples/train/sokoban/train_default_gae_qwen25vl3b.sh
bash examples/train/sokoban/train_ppo_no_concat_qwen25vl3b.sh
bash examples/train/sokoban/train_default_gae_compact_qwen25vl3b.sh

# Qwen2.5-VL: default GAE with state reward
bash examples/train/sokoban/train_default_gae_sr_qwen25vl3b.sh

# Qwen3-VL, Qwen3.5, InternVL3.5 and GLM-4.6V-Flash all accept
# HARNESS=concat|no_concat|compact (concat is the default).
HARNESS=concat bash examples/train/sokoban/train_default_gae_qwen3vl4b.sh
HARNESS=concat bash examples/train/sokoban/train_default_gae_qwen35_4b.sh
HARNESS=concat bash examples/train/sokoban/train_default_gae_internvl35_2b.sh
HARNESS=concat bash examples/train/sokoban/train_default_gae_glm46v_flash.sh

# Validation only, without starting training
bash examples/train/sokoban/train_default_gae_internvl35_2b.sh \
  trainer.val_only=true trainer.save_freq=-1 trainer.test_freq=-1
```

See [Configuration](docs/configuration.md) for harnesses, estimators, model-specific flags,
and state-reward settings.

### Evaluation

```bash
cd VAGEN

# Local vLLM
MODEL_PATH=Qwen/Qwen2.5-VL-3B-Instruct \
  bash examples/evaluate/sokoban/vllm/eval_qwen25_vl_3b.sh

# OpenAI-compatible endpoint
bash examples/evaluate/sokoban/run_eval.sh
```

See [Evaluation](vagen/evaluation/README.md) for other environments and backends.

<details>
<summary>With sglang instead</summary>

Requires the sglang extra, which is mutually exclusive with vLLM — see Installation.

```bash
bash examples/evaluate/frozenlake/sglang/eval_qwen25_vl_3b.sh
```
</details>

## Repository Layout

VAGEN separates stable orchestration from selectable implementations:

```text
vagen/
├── algorithms/              # advantage algorithms and registry
├── envs/                    # environment facade, shared contracts, implementations
├── harness/                 # context-policy facade and implementations
├── models/                  # model-family adaptation
├── rollout/                 # framework-independent episode execution
├── evaluation/              # standalone evaluation and backend plugins
└── training/                # VERL agent-loop and trainer integration
```

Extension axes use `axis/_common/` for shared contracts and one directory per concrete
implementation. Import shared APIs from the axis facade, such as `vagen.envs`,
`vagen.harness`, or `vagen.algorithms`. Training-only integration lives under
`vagen.training`; framework-independent rollout code lives under `vagen.rollout`.
Shared environment reward machinery lives under `vagen/envs/_common/rewards/`, while
an environment-specific reward specification stays beside that environment.
Keep `_common` limited to contracts and helpers genuinely reused by multiple
implementations. A selectable implementation owns its actual control flow in its own
directory, and cross-package consumers import through the axis facade.

## Custom Environment

To train on your own environment, follow the steps below.


### 1. Create Your Environment Class

* Import `GymImageEnv` from the environment facade:

  ```python
  from vagen.envs import GymImageEnv
  ```

  The shared contract lives under [`vagen/envs/_common`](vagen/envs/_common/).
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

## Custom Advantage Estimator

Add an estimator under `vagen/algorithms/<name>/` and import it from the
[`vagen.algorithms`](vagen/algorithms/__init__.py) facade:

```python
from vagen.algorithms import AdvantageInputs, AdvantageOutputs, advantage_estimator

@advantage_estimator("my_estimator", needs_critic=True)
def my_estimator(inputs: AdvantageInputs):
    returns = inputs.rewards
    advantages = (returns - inputs.values) * inputs.response_mask
    return AdvantageOutputs(advantages=advantages, returns=returns)
```

Select it in a training command with `algorithm.adv_estimator=my_estimator`. See
[`inputs.py`](vagen/algorithms/_common/inputs.py) for the available inputs and the
implementation directories under [`vagen/algorithms`](vagen/algorithms/) for examples.

## Custom Harness

A **harness** is the reusable agent loop. It owns the message list, calls
`client.create(messages)`, steps the environment with the returned response, and stops
when the environment terminates or truncates. `concat`, `no_concat`, and `compact` differ
only in how they build that message list.

**Anything that subclasses `BaseEnv` or `BaseHarness` works in both training and evaluation
without changing either.** A harness uses the small `create`/`size` client surface and the
environment step contract. It never builds token masks, VERL rows, or reward mappings:
those belong to the inference client and rollout scoring seam.

A harness also doesn't assume the conversation is kept on the client side. That leaves room
to add a backend that keeps it on the server instead (OpenAI's `previous_response_id`, an
SGLang session, a vLLM prefix cache) without rewriting any harness. No backend shipped today
does that — they all re-send the whole message list every turn.

```python
from vagen.harness import BaseHarness, register_harness

@register_harness("mine")
class MyHarness(BaseHarness):
    #: Whether one episode can end up in more than one row. The trainer asks the harness
    #: rather than keeping a list of the ones it knows, and pairs the estimator accordingly.
    splits_episode_across_rows = True

    async def run_episode(self, client, env):
        observation, _ = await env.reset()
        messages = [await env.system_prompt(), observation]
        while True:
            response = await client.create(messages)
            observation, reward, terminated, truncated, info = await env.step(response)
            if terminated or truncated:
                return
            messages.extend([
                {"role": "assistant", "content": response.text},
                observation,
            ])
```

Training and evaluation both read a `harness` key, and both accept either a registered name
or an import path. What differs is who imports your module:

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

In evaluation, use the import path. The `@register_harness` decorator only runs if
something imports your module, and `run_eval` has no `external_lib` setting to make that
happen — so a bare `harness: mine` fails with:

```
unknown harness 'mine'; choose from ['compact', 'concat', 'no_concat']
```

Give the import path instead and VAGEN imports the module itself. This is why the import
path is supported at all: a new harness is usually tried in evaluation first, where
everything is configured in yaml and no decorator has had a chance to run.

The public contract and registry are exported by [`vagen/harness`](vagen/harness/), with
shared internals under `_common/` and one directory per implementation.

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
  model_adapter: auto       # detects qwen | internvl | glm; VERL remains the transport

# no_concat and compact put one episode in several rows, so the advantage estimator has
# to be one that stitches them back together. verl's own `gae`/`grpo` score a row at a
# time and would drop every turn's credit at the row boundary; the trainer refuses that
# pairing at startup rather than training on it.
algorithm:
  adv_estimator: default_gae       # or turn_level_gae | token_level_gae
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
  name: reward_variance_top_p # registered by vagen.training.filters
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
