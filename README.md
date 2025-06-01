<h1 align="center">Reinforcing Visual State Reasoning for Multi-Turn VLM Agents</h1>
<!-- <p align="center" style="font-size: 18px;">
  Reinforcing Visual State Reasoning for Multi-Turn VLM Agents<br>
</p> -->
<p align="center">
  <a href="https://vagen.readthedocs.io/en/latest"><img src="https://img.shields.io/badge/📚_Documentation-4285F4?style=for-the-badge&logoColor=white" alt="Documentation"></a>
  <a href="https://mll-lab.notion.site/vagen"><img src="https://img.shields.io/badge/📝_Blog-FF5722?style=for-the-badge&logoColor=white" alt="Blog"></a>
  <a href="https://api.wandb.ai/links/ragen-V/nlb40e7l"><img src="https://img.shields.io/badge/📊_Experiment_Log-FB8C00?style=for-the-badge&logoColor=white" alt="Experiment Log"></a>
</p>



<!--
VAGEN is a multi-turn reinforcement learning framework designed specifically for training VLM Agents. VAGEN leverages the TRICO algorithm to efficiently train VLMs for visual agentic tasks.
-->
We introduce Visual Reasoning RL, a reinforcement learning approach that improves multi-turn performance of vision-language models (VLMs) by explicitly supervising visual state reasoning.
We propose **VAGEN**, a scalable training framework that enables this method across diverse visual environments

<!--
![vagen_new](https://github.com/user-attachments/assets/83c84052-89ba-4a77-9c13-85d882f52a3b)
-->
<img width="1835" alt="image" src="https://github.com/user-attachments/assets/e5b70eeb-21de-4808-90c0-9ee7d990acd1" />


<!--
## Key Innovations

VAGEN introduces the **Turn-aware Reason-Interaction Chain Optimization (TRICO)** algorithm which extends the traditional RICO approach with two key innovations:

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
-->

## Key Innovations
Standard RL methods applied to VLMs struggle with multi-turn agentic tasks due to:
1. **Visual State Ambiguity**: VLMs lack mechanisms to explicitly interpret and track evolving visual environments
2. **Precision Bottlenecks**: Existing representations fall short in tasks requiring fine-grained spatial or temporal understanding

Our framework addresses through:
1. **Visual State Reasoning Prompts** – Injects structured prompts like grounding (current state description) and world modeling (future state prediction) to scaffold the model’s internal reasoning
2. **Visual Reasoning RL** – Reinforces visual understanding with:
   - **Turn-level reasoning rewards** for supervising accuracy
   - **Bi-Level GAE** for fine-grained credit assignment at both turn and token levels


## News

**[2025/04]** We've introduced a new modular design for environments and services in VAGEN:
- Enhanced environment framework for easier creation of custom environments
- New service architecture for efficient distributed training
- Check out our new guides:
  - [Creating Environments](./docs/envs/create-env.md): New environment protocal.
  - [Creating Services](./docs/envs/create-service.md): We now support hosting environments in a separate process

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
# This script installs dependencies for Frozenlake and Sokoban, for other environments, please refer to vagen/env/README.md and uncomment the registration in vagen/env/__init__.py
```


## Examples

**Note:** VAGEN currently supports several environments: sokoban, frozenlake, svg, navigation, and primitive skill. 
<img width="1084" alt="image" src="https://github.com/user-attachments/assets/f59f9a65-b93a-44b7-81c1-89df0da91b2e" />

For simplifying installation and execution, we have **commented out** all environments except sokoban and frozenlake. If you wish to run other environments, please **uncomment** the corresponding sections in `scripts/install.sh` and `vagen/env/__init__.py`.
```
# Login to wandb
wandb login

# Then, you can run different environments and algorithms:

#Run a server process in a saperate tmux session if you want to train in env-as-service mode
python vagen/server/server.py

# Frozen Lake Environment
bash scripts/examples/frozen_lake_aico/run.sh         # AICO without service
bash scripts/examples/frozen_lake_trico/run.sh        # TRICO without service
bash scripts/examples/frozen_lake_aico_service/run.sh # AICO with service

# SVG Generation
bash scripts/examples/svg_aico/run.sh                 # AICO without service
bash scripts/examples/svg_trico/run.sh                # TRICO without service
```
## How to Add New Environment and Services

See our [Creating Environments](./docs/envs/create-env.md) guide. You may also want to check our [Creating Service](./docs/envs/create-service.md) for scaling your environments.

## How to Add New Model

1. Refer to [VERL](https://verl.readthedocs.io/en/latest/index.html) for adding new MLLM.
2. Refer to [QwenVLRolloutManager](vagen/rollout/qwen_rollout/rollout_manager.py) to understand how rollout works. In most cases, you can use QwenVLRolloutManager directly with only minor modifications to the model's special tokens

<!--
## Experimental Results
> To reproduce our experiment, please refer to document: [Reproduce Experiments](docs/reproduce-exp.md)


Our experiments on visual Sokoban using a Qwen-VL 3B model show:
- TRICO significantly outperforms RICO in visual agentic tasks
- Both selective token masking and cross-turn credit assignment contribute to performance gains
- AICO (Action-centric Interaction Chain Optimization), which uses only selective token masking, outperforms TRICO on simple tasks
- TRICO demonstrates superior exploration capabilities on more complex problems

<img width="800" alt="image" src="./public/1.png" />

<img width="800" alt="image" src="./public/2.png" />

<img width="800" alt="image" src="./public/3.png" />
-->
## Experimental Results
We benchmark closed- and open-sourced models on five environments. Reasoning on visual states, including both grounding and world modeling, can significantly improve
the RL performance. 
<img width="1093" alt="image" src="https://github.com/user-attachments/assets/162820e8-a4f3-49b7-b8f8-c7963a5ac6f1" />

Incorporating Visual Reasoning RL leads to improved performance
<img width="1319" alt="image" src="https://github.com/user-attachments/assets/cba16487-c24b-4b25-9ecf-a668d4cd8ac6" />



<!--
## Cases
We present several cases selected from validation steps during training models with AICO and TRICO, as shown below. You can view all the cases in our [Experiment Log](https://api.wandb.ai/links/ragen-V/nlb40e7l).

### Cases from AICO Training
<img width="1107" alt="image (4)" src="https://github.com/user-attachments/assets/995ec921-faf8-4832-a4c0-1c2ce559a55c" />

![image (5)](https://github.com/user-attachments/assets/78bbc376-7e61-4a24-9911-eb28416eed37)

### Cases from TRICO Training
![image (6)](https://github.com/user-attachments/assets/60b251a2-e395-4079-a9aa-ceb4455b0a7a)

![image (7)](https://github.com/user-attachments/assets/ddea7352-0a14-45a5-94a9-655a07c9fe3e)
-->
## Cases
<img width="1261" alt="image" src="https://github.com/user-attachments/assets/59fbce0c-4932-4f82-ab87-d407d84ebad4" />


# Project Roadmap
- 🗓️ Mar 25, 2025: We release VAGEN, a multi-turn reinforcement learning framework for training VLM Agents!
- [ ] Merge to RAGEN for better package mangement
- [ ] Expand evaluation framework to more diverse visual environments
- [ ] Scaling to larger models and applying TRICO to text-only tasks


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

If you find our repo useful, we appreciate it if you could cite our work at:

```bibtex
@misc{VAGEN,
  title={Reinforcing Visual State Reasoning for Multi-Turn VLM Agents},
  author={Kangrui Wang* and Pingyue Zhang* and Zihan Wang* and Yaning Gao* and Linjie Li* and Qineng Wang and Hanyang Chen and Chi Wan and Yiping Lu and Zhengyuan Yang and Lijuan Wang and Ranjay Krishna and Jiajun Wu and Li Fei-Fei and Yejin Choi and Manling Li},
  url={https://github.com/RAGEN-AI/VAGEN},
  year={2025},
}
```
