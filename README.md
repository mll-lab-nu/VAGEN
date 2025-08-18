<h1 align="center">VAGEN: Training VLM agents with multi-turn reinforcement learning</h1>
<p align="center" style="font-size: 20px;">
  <b>Reinforcing Visual State Reasoning for Multi-Turn VLM Agents</b>
</p>

<p align="center" style="font-size: 16px;">
  Kangrui Wang*, Pingyue Zhang*, Zihan Wang*, Yaning Gao*, Linjie Li*, Qineng Wang, Hanyang Chen, Chi Wan, Yiping Lu, Zhengyuan Yang, Lijuan Wang, Ranjay Krishna, Jiajun Wu, Li Fei-Fei, Yejin Choi, Manling Li
</p>
<p align="center" style="font-size: 12px;"><i>(* equal contribution)</i></p>

<p align="center">
  <a href="https://arxiv.org/abs/YOUR_ARXIV_ID_HERE"><img src="https://img.shields.io/badge/📜_Paper-B31B1B?style=for-the-badge&logo=arXiv&logoColor=white" alt="Paper"></a>
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

<!--
<table>
  <tr>
    <td align="center"><b>FrozenLake</b><br>
      <img src="https://github.com/user-attachments/assets/6d72800a-9b4d-45ec-b528-ac81efb93966" width="150"/><br>
      <img src="https://github.com/user-attachments/assets/6f283f99-fa15-4e26-9f99-6649a7d72374" width="150"/>
    </td>
    <td align="center"><b>Navigation</b><br>
      <img src="https://github.com/user-attachments/assets/b364e6c9-4c2c-46d0-afca-ee42f271c59c" width="150"/><br>
      <img src="https://github.com/user-attachments/assets/65662eb0-9440-4555-9436-8b9272791ac4" width="150"/>
    </td>
    <td align="center"><b>Sokoban</b><br>
      <img src="https://github.com/user-attachments/assets/145352b5-3a9e-4248-bb94-d3fa46e6c493" width="150"/><br>
      <img src="https://github.com/user-attachments/assets/676de052-37d6-4c99-a7eb-200a58d11ed4" width="150"/>
    </td>
    <td align="center"><b>ManiSkill</b><br>
      <img src="https://github.com/user-attachments/assets/c597f17d-5c62-4319-bdaa-b7fa8e4564e1" width="150"/><br>
      <img src="https://github.com/user-attachments/assets/f61ea55c-ea79-4ead-9345-45be06d24e81" width="150"/>
    </td>
    <td align="center"><b>SVG</b><br>
      <img src="https://github.com/user-attachments/assets/8646da5f-69be-4283-a078-969f9b8f3f3b" width="150"/><br>
      <img src="https://github.com/user-attachments/assets/691b896a-ce30-4acc-ac49-af2d89452bdd" width="150"/>
    </td>
  </tr>
</table>
-->

This repository contains the official implementation of our paper, **"Reinforcing Visual State Reasoning for Multi-Turn VLM Agents"**.

We introduce **VAGEN**, a multi-turn reinforcement learning framework designed specifically for training vision-language model (VLM) agents. Built upon this framework, we propose **Visual Reasoning RL**, a novel reinforcement learning approach that significantly improves the multi-turn performance of VLMs by explicitly supervising their visual state reasoning process.

<!--
![bi-level-gae](https://github.com/user-attachments/assets/fbf0ec24-6bb4-40ce-b545-818d83d04e05)
-->
![image](https://github.com/user-attachments/assets/834b32fa-9bfc-4e0f-a148-99cd6fc3141e)


## News

**[2025/05]** We are excited to release our paper, **"Reinforcing Visual State Reasoning for Multi-Turn VLM Agents"**, introducing the **Visual Reasoning RL** method!

**[2025/04]** We've introduced a new modular design for environments and services in VAGEN:
- Enhanced environment framework for easier creation of custom environments
- New service architecture for efficient distributed training
- Check out our new guides:
  - [Creating Environments](./docs/envs/create-env.md): New environment protocal.
  - [Creating Services](./docs/envs/create-service.md): We now support hosting environments in a separate process

**[2025/03]** We release VAGEN, a multi-turn reinforcement learning framework for training VLM Agents!

<!--
## Why Visual Reasoning RL?
Standard RL methods applied to VLMs struggle with multi-turn agentic tasks due to:
1. **Visual State Ambiguity**: VLMs lack mechanisms to explicitly interpret and track evolving visual environments.
2. **Precision Bottlenecks**: Existing representations fall short in tasks requiring fine-grained spatial or temporal understanding.

Our approach, **Visual Reasoning RL**, addresses these challenges through:
1. **Visual State Reasoning Prompts**: Injects structured prompts like grounding (current state description) and world modeling (future state prediction) to scaffold the model’s internal reasoning.
2. **Reinforcement Learning with Reasoning Rewards**: Reinforces visual understanding with:
   - **Turn-level reasoning rewards** for supervising accuracy.
   - **Bi-Level GAE** for fine-grained credit assignment at both turn and token levels.
-->

## Why Visual Reasoning RL?
Standard RL methods applied to VLMs struggle with multi-turn agentic tasks due to:
1. **Visual State Ambiguity**: VLMs lack mechanisms to explicitly interpret and track evolving visual environments.
2. **Precision Bottlenecks**: Existing representations fall short in tasks requiring fine-grained spatial or temporal understanding.

Our approach, **Visual Reasoning RL**, addresses these challenges through:

### Boost #1: Visual Reasoning
1. **Reasoning Prompts**: Injects structured prompts like grounding (current state description) and world modeling (future state prediction) to scaffold the model’s internal reasoning.
2. **Reasoning Rewards**: We use LLM-as-Judge to reward the agent when its predicted or observed visual state matches the ground truth.

### Boost #2: Turn-level
1. **Turn-level reasoning rewards** for supervising accuracy.
2. **Bi-Level GAE** for fine-grained credit assignment at both turn and token levels.

<table>
  <tr>
    <td align="center">
      <img src="https://github.com/user-attachments/assets/406cdf7f-79c0-4732-8df1-009c893f3840" width="600"/><br/>
      <sub>Boost #1: Visual Reasoning</sub>
    </td>
    <td align="center">
      <img src="https://github.com/user-attachments/assets/fbf0ec24-6bb4-40ce-b545-818d83d04e05" width="600"/><br/>
      <sub>Boost #2: Turn-level</sub>
    </td>
  </tr>
</table>

## Key Innovations of VAGEN

Two key innovations are introduced in VAGEN to support methods like Visual Reasoning RL:

1. **Selective Token Masking** - Focuses optimization on action-critical tokens through:
   - Loss masking (`M^loss`): Identifies tokens to update during policy optimization
   - Advantage masking (`M^adv`): Determines tokens to include in advantage calculations

2. **Cross-turn Credit Assignment** - Enables more effective credit attribution through:
   - Bi-level advantage estimation with separate discount factors for cross-turn (`γ_turn`) and within-turn (`γ_token`) calculations
   - Turn-level rewards applied at each interaction boundary

## Why VAGEN Works Better for VLM Agents

Traditional RL frameworks for LLM agents treat all tokens in a trajectory equally. This approach is suboptimal for VLM agents due to:

- **Distribution Shift**: Most VLMs aren't pretrained to generate image tokens
- **State Redundancy**: Visual tasks contain excessive low-level information in long-context inputs

VAGEN addresses these challenges by focusing optimization on the most critical decision-making tokens and creating a more nuanced reward structure across interaction turns.

## The VAGEN Workflow

We present the workflow of **VAGEN** in the image below. The `rollout.py` module facilitates interactions between `ray_trainer.py` and various environments. Our framework operates with two forms of “language”: token sequences (used by the model) and structured information from the environments. `rollout.py` serves as a translator, parsing structured environment data into tokens for the model and converting model outputs back into structured actions or observations. It also records data of each step to form the entire trajectory.
<!--
![framework](https://github.com/user-attachments/assets/183cea78-2345-4b5e-82c5-a0679c5f112a)
-->
<div align="center">
  <img src="https://github.com/user-attachments/assets/4cd3752c-e1ad-4cfd-8928-9ef9b0180e5d" width="800"/>
</div>


## Installation

```bash
# Create a new conda environment
conda create -n vagen python=3.10 -y
conda activate vagen

# verl
git clone https://github.com/JamesKrW/verl.git
cd verl
pip install -e .
cd ../

# vagen
git clone https://github.com/RAGEN-AI/VAGEN.git
cd VAGEN
bash scripts/install.sh
# This script installs dependencies for Frozenlake and Sokoban, for other environments, please refer to vagen/env/README.md
```


## Usage
```
# Login to wandb
wandb login

# You can run different environments and algorithms:
bash scripts/examples/masked_grpo/frozenlake/run_tmux.sh
bash scripts/examples/masked_turn_ppo/frozenlake/run_tmux.sh
bash scripts/examples/vagen_base/sokoban/run_tmux.sh

# Use Visual Reasoning Reward
# Setup OPENAI_API_KEY in the Environment
bash scripts/examples/vagen_full/sokoban/run_tmux.sh
```
## How to Add New Environment and Services

See our [Creating Environments](./docs/envs/create-env.md) guide. You may also want to check our [Creating Service](./docs/envs/create-service.md) for scaling your environments.

## How to Add New Model

1. Refer to [VERL](https://verl.readthedocs.io/en/latest/index.html) for adding new MLLM.
2. Refer to [QwenVLRolloutManager](vagen/rollout/qwen_rollout/rollout_manager.py) to understand how rollout works. In most cases, you can use QwenVLRolloutManager directly with only minor modifications to the model's special tokens

## Experimental Results
We benchmark closed- and open-sourced models on five environments. Reasoning on visual states, including both grounding and world modeling, can improve the performance. 
<!--
<img width="1093" alt="image" src="https://github.com/user-attachments/assets/162820e8-a4f3-49b7-b8f8-c7963a5ac6f1" />
-->
<img width="1253" alt="image" src="https://github.com/user-attachments/assets/201d633b-910d-4384-88c9-e1dd0acaa88c" />




Incorporating **Visual Reasoning RL** leads to improved performance.
<!--
<img width="1319" alt="image" src="https://github.com/user-attachments/assets/cba16487-c24b-4b25-9ecf-a668d4cd8ac6" />
-->
- VAGEN-Base uses the Grounding-WorldModeling reasoning strategy along with format and task-specific rewards.
- VAGEN-Full builds on this and incorporates Visual Reasoning RL
<img width="1253" alt="image" src="https://github.com/user-attachments/assets/066adeb0-ef7f-449b-8e01-21fb643eee2b" />


## Environments

**Note:** VAGEN currently supports several environments: sokoban, frozenlake, svg, navigation, and primitive skill. 
<img width="1084" alt="image" src="https://github.com/user-attachments/assets/f59f9a65-b93a-44b7-81c1-89df0da91b2e" />


## Cases

### Preview (click to show full cases)

<!--
<img width="923" alt="image" src="https://github.com/user-attachments/assets/dd412cc2-836b-4d23-81e0-a29d4eaf22b2" />
<img width="923" alt="image" src="https://github.com/user-attachments/assets/d3e07add-5233-46d7-b955-23111ac0c0d7" />
[![preview](https://github.com/user-attachments/assets/d3e07add-5233-46d7-b955-23111ac0c0d7)](https://github.com/user-attachments/assets/dd412cc2-836b-4d23-81e0-a29d4eaf22b2)
[![preview](https://github.com/user-attachments/assets/d3e07add-5233-46d7-b955-23111ac0c0d7)](public/cases_full.png)
-->

<a href="https://raw.githubusercontent.com/RAGEN-AI/VAGEN/refs/heads/main/public/cases_full.png" target="_blank">
  <img src="https://github.com/user-attachments/assets/d3e07add-5233-46d7-b955-23111ac0c0d7">
</a>

## Acknowledgement
We thank [RAGEN](https://github.com/RAGEN-AI/RAGEN) for its innovative exploration in multi-turn reinforcement learning for LLM agents. We thank [verl](https://github.com/volcengine/verl) for its RL framework. We thank [EasyR1](https://github.com/hiyouga/EasyR1) for adding initial support for VLMs to verl.

## References
[RAGEN](https://github.com/RAGEN-AI/RAGEN): Training Agents by Reinforcing Reasoning

[verl](https://www.notion.so/VAGEN-Training-VLM-Agents-with-Multi-Turn-Reinforcement-Learning-1bfde13afb6e80b792f6d80c7c2fcad0?pvs=21): Volcano Engine Reinforcement Learning for LLM

[ArCHer](https://arxiv.org/abs/2402.19446v1): Hierarchical Multi-Turn RL Agent Training Framework

[Search-R1](https://github.com/PeterGriffinJin/Search-R1): Train your LLMs to reason and call a search engine with reinforcement learning

[Agent-R1](https://github.com/0russwest0/Agent-R1): Training Powerful LLM Agents with End-to-End Reinforcement Learning

[OpenManus-RL](https://github.com/OpenManus/OpenManus-RL): A live stream development of RL tunning for LLM agents


## Citation

If you find our framework and paper useful, we appreciate it if you could cite our work:

```bibtex
@misc{wang2025vagen,
  title={Reinforcing Visual State Reasoning for Multi-Turn VLM Agents},
  author={Kangrui Wang* and Pingyue Zhang* and Zihan Wang* and Yaning Gao* and Linjie Li* and Qineng Wang and Hanyang Chen and Chi Wan and Yiping Lu and Zhengyuan Yang and Lijuan Wang and Ranjay Krishna and Jiajun Wu and Li Fei-Fei and Yejin Choi and Manling Li},
  year={2025},
  url={https://github.com/RAGEN-AI/VAGEN}
}
```
