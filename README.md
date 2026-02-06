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
**[2026/02]** We have migrated the `main` branch to VAGEN-Lite, a lightweight and clean reimplementation built on VERL agent-loop for easy customization and stable performance. For the previous full-featured release, please visit the [vagen-legacy](https://github.com/mll-lab-nu/VAGEN/tree/vagen-legacy) branch.

## Installation

```bash
conda create -n vagen python=3.12 -y
conda activate vagen

git clone https://github.com/mll-lab-nu/VAGEN.git
cd VAGEN
git submodule update --init --recursive

pip install -e .

cd verl
USE_MEGATRON=0 bash scripts/install_vllm_sglang_mcore.sh
pip install --no-deps -e .

pip install "trl==0.26.2"
```


## Quick Start

### Training
VAGEN currently supports PPO / GRPO with two multi-turn training paradigms:

**Multi-turn Concatenated Training**: All turns in a trajectory are concatenated into a single training instance.

```bash
cd VAGEN
bash examples/sokoban/train_ppo_qwen25vl3b.sh
```

**Multi-turn Non-Concatenated Training**: Each trajectory is split into multiple turn-level training instances.

```bash
cd VAGEN
bash examples/sokoban/train_ppo_no_concat_qwen25vl3b.sh
```
### Evaluation (supported by [ViewSuite](https://github.com/viewsuite/ViewSuite))

VAGEN supports evaluation using different backends (OpenAI, Claude, Gemini, sglang, vLLM). For details, see [vagen/evaluate/adapters/README.md](vagen/evaluate/adapters/README.md).

```bash
cd VAGEN

# FrozenLake evaluation with sglang
bash examples/evaluate/frozenlake/eval_qwen25_vl_3b.sh

# Sokoban evaluation
bash examples/evaluate/sokoban/run_eval.sh

```

## Customizing Your Environment

To train on your own environment, follow the steps below.

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

* [`examples/sokoban/train_sokoban_vision.yaml`](examples/sokoban/train_sokoban_vision.yaml)
* [`examples/sokoban/val_sokoban_vision.yaml`](examples/sokoban/val_sokoban_vision.yaml)


### 4. Create a Training Script

Write your training script based on:

* [`examples/sokoban/train_ppo_qwen25vl3b.sh`](examples/sokoban/train_ppo_qwen25vl3b.sh)


## More Customization

See the [Documentation](https://vagen.readthedocs.io/) for more customization options:

- [Custom Filter](https://vagen.readthedocs.io/en/latest/custom-filter/) - Preprocess training data (supported by [RAGEN](https://github.com/RAGEN-AI/RAGEN))
- [Custom Metric](https://vagen.readthedocs.io/en/latest/custom-metric/) - Add W&B logging metrics
- [Configuration](https://vagen.readthedocs.io/en/latest/configuration/) - Training configuration reference


## Citation

If you find our framework and paper useful, we appreciate it if you could cite our work:

```bibtex
@misc{wang2025vagen,
  title={VAGEN:Reinforcing World Model Reasoning for Multi-Turn VLM Agents},
  author={Kangrui Wang* and Pingyue Zhang* and Zihan Wang* and Yaning Gao* and Linjie Li* and Qineng Wang and Hanyang Chen and Chi Wan and Yiping Lu and Zhengyuan Yang and Lijuan Wang and Ranjay Krishna and Jiajun Wu and Li Fei-Fei and Yejin Choi and Manling Li},
  year={2025},
  url={https://arxiv.org/abs/2510.16907}
}
```
