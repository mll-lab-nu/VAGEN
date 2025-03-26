<h1 align="center">VAGEN: Training VLM Agents with Multi-Turn Reinforcement Learning</h1>
<!-- <p align="center" style="font-size: 18px;">
  <strong>VAGEN</strong>: Multi-turn Reinforcement Learning for Visual Reasoning Agents<br>
</p> -->
<p align="center">
  <a href="#"><img src="https://img.shields.io/badge/📚_Documentation-4285F4?style=for-the-badge&logoColor=white" alt="Documentation"></a>
  <a href="https://mll-lab.notion.site/vagen"><img src="https://img.shields.io/badge/📝_Blog-FF5722?style=for-the-badge&logoColor=white" alt="Blog"></a>
  <a href="https://api.wandb.ai/links/ragen-V/nlb40e7l"><img src="https://img.shields.io/badge/📊_Experiment_Log-FB8C00?style=for-the-badge&logoColor=white" alt="Experiment Log"></a>
</p>




VAGEN is a multi-turn reinforcement learning framework designed specifically for training VLM Agents. VAGEN leverages the TRICO algorithm to efficiently train VLMs for visual agentic tasks.

![vagen_new](https://github.com/user-attachments/assets/83c84052-89ba-4a77-9c13-85d882f52a3b)



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

## Experimental Results
Our experiments on visual Sokoban using a Qwen-VL 3B model show:
- TRICO significantly outperforms RICO in visual agentic tasks
- Both selective token masking and cross-turn credit assignment contribute to performance gains
- AICO (Action-centric Interaction Chain Optimization), which uses only selective token masking, outperforms TRICO on simple tasks
- TRICO demonstrates superior exploration capabilities on more complex problems

<img width="800" alt="image" src="./public/1.png" />

<img width="800" alt="image" src="./public/2.png" />

<img width="800" alt="image" src="./public/3.png" />
  

## Comparison of Algorithms

| **Feature** | **PPO** | **RICO** | **TRICO (Ours)** |
| --- | --- | --- | --- |
| **Sequence Structure** | Single response | Multiple turn interaction | Multiple turn interaction |
| **LM output** | No special structure | `<think>...</think><ans>...</ans>` | `<think>...</think><ans>...</ans><eoa>` |
| **Discounting** | Single discount rate | Single discount rate | Bi-level discounting |
| **Optimization** | All tokens equally | All tokens equally | Selective token optimization |

## Training Configuration

We used the following settings in our experiments:

- **Model**: Qwen 2.5 VL-instruction 3B
- **Environment**: Visual Sokoban (puzzle-solving task)
- **Rewards**: Box on target (+1.0), All boxes placed (+10.0), Format correct (+0.5), Step penalty (-0.1)
- **Hyperparameters**: `γ_turn`=0.95, `γ_token`=1.0, KL penalty=0.001, Actor LR=1e-6, Critic LR=1e-5


----

## Installation

```bash
# Create a new conda environment
conda create -n vagen python=3.10 -y

# verl
git clone git@github.com:JamesKrW/verl.git
cd verl
git checkout release
pip install -e .
cd ../

# vagen
git clone git@github.com:RAGEN-AI/vagen.git
git checkout release
cd vagen
bash scripts/install.sh
```

## Running Experiments

```bash
# To reproduce our reults, you can run
bash vagen/vagen/examples/release_experiments/gae.sh # rico-gae
bash vagen/vagen/examples/release_experiments/grpo_mask_loss.sh # rico-grpo + loss mask
bash vagen/vagen/examples/release_experiments/grpo.sh # rico-grpo
bash vagen/vagen/examples/release_experiments/mask_gae_mask_loss_bi_level.sh # trico - turn reward
bash vagen/vagen/examples/release_experiments/mask_gae_mask_loss_turnwise_gae.sh # trico - turn reward - bi-level gae + turn-level gae
bash vagen/vagen/examples/release_experiments/mask_gae_mask_loss_turnwise_reward_bi_level.sh # trico
bash vagen/vagen/examples/release_experiments/mask_gae_mask_loss.sh # aico
bash vagen/vagen/examples/release_experiments/mask_gae.sh # aico - loss mask
bash vagen/vagen/examples/release_experiments/mask_loss.sh # aico - gae mask
```
Each run takes ~4 hours to reach 150 steps on 4 H100s. You can decrease testing frequency to speed up training. Training might be unstable due to loss spikes; we recommend restoring from the latest checkpoint when encountering such cases. We will resolve this issue in future work (see roadmap).

## Algorithm Settings

| Setting           | GRPO | GAE | Bi-Level GAE | Turn-Wise GAE | Masked-GAE |
|-------------------|------|-----|--------------|---------------|------------|
| with_loss_mask    | ✓    | ✓   | ✓            | ✓             | ✓          |
| multi-turn-reward | ✗    | ✓   | ✓            | ✓             | ✓          |
| with_gae_mask     | ✗    | ✗   | ✓            | ✓             | ✓          |

### Algorithm Options

- **GRPO**: Whether to use GRPO
  - `algorithm.adv_estimator=grpo`
- **GAE**: Whether to use GAE
  - `algorithm.adv_estimator=gae`
- **Bi-Level-GAE**: Whether to use multi-turn GAE (first estimates turn-level advantage, then estimates advantage in each turn)
  - `algorithm.adv_estimator=bi_level_gae`
- **Turn-Wise-GAE**: Whether to use turn-aware GAE (each turn will have only one same advantage estimation)
  - `algorithm.adv_estimator=turn_wise_gae`
- **Masked-GAE**: Whether to use masked GAE (skips observation tokens from environment when estimating advantages)
  - `algorithm.adv_estimator=masked_gae`

### Configuration Settings

- **multi-turn-reward**: Whether to use multi-turn reward (gives step reward for last token of each turn, instead of summing all rewards for last token of whole trajectory)
  - `rollout_manager.use_multi_turn_reward=True`
- **with_loss_mask**: Whether to use loss mask to only calculate the loss of tokens output by the models
  - `rollout_manager.use_loss_mask=True`
- **with_loss_mask**: Whether to use gae mask to only calculate the gae of tokens output by the models
  - `rollout_manager.use_gae_mask=True`

 ## Cases
  We present several cases selected from validation steps during training models with AICO and TRICO, as shown below. You can view all the cases in our [Experiment Log](https://api.wandb.ai/links/ragen-V/nlb40e7l).

### Cases from AICO Training
<img width="1107" alt="image (4)" src="https://github.com/user-attachments/assets/995ec921-faf8-4832-a4c0-1c2ce559a55c" />

![image (5)](https://github.com/user-attachments/assets/78bbc376-7e61-4a24-9911-eb28416eed37)

### Cases from TRICO Training
![image (6)](https://github.com/user-attachments/assets/60b251a2-e395-4079-a9aa-ceb4455b0a7a)

![image (7)](https://github.com/user-attachments/assets/ddea7352-0a14-45a5-94a9-655a07c9fe3e)


# Project Roadmap
- 🗓️ Mar 25, 2025: We release VAGEN, a multi-turn reinforcement learning framework for training VLM Agents!
- [ ] Merge to RAGEN for better package mangement
- [ ] Address training stability issues
  - [ ] Lora Support
  - [ ] Implement improved early stopping and gradient clipping strategies
  - [ ] Reward normalization (refer to RAGEN)
- [ ] Expand evaluation framework to more diverse visual environments
- [ ] Scaling to larger models and applying TRICO to text-only tasks


## Acknowledgement
We thank [RAGEN](https://github.com/RAGEN-AI/RAGEN) for its innovative exploration in multi-turn reinforcement learning for LLM agents. We thank [verl](https://github.com/volcengine/verl) for its RL framework. We thank [EasyR1](https://github.com/hiyouga/EasyR1) for adding initial support for VLMs to verl.

## References
[RAGEN](https://github.com/RAGEN-AI/RAGEN): Training Agents by Reinforcing Reasoning

[verl](https://www.notion.so/VAGEN-Training-VLM-Agents-with-Multi-Turn-Reinforcement-Learning-1bfde13afb6e80b792f6d80c7c2fcad0?pvs=21): Volcano Engine Reinforcement Learning for LLM

[Search-R1](https://github.com/PeterGriffinJin/Search-R1): Train your LLMs to reason and call a search engine with reinforcement learning

[Agent-R1](https://github.com/0russwest0/Agent-R1): Training Powerful LLM Agents with End-to-End Reinforcement Learning

[OpenManus-RL](https://github.com/OpenManus/OpenManus-RL): A live stream development of RL tunning for LLM agents

## Citation

If you find our repo useful, we appreciate it if you could cite our work at:

```bibtex
@misc{VAGEN,
  title={VAGEN: Training VLM Agents with Multi-Turn Reinforcement Learning},
  author={Kangrui Wang* and Pingyue Zhang* and Zihan Wang* and Qineng Wang* and Linjie Li* and Zhengyuan Yang and Chi Wan and Yiping Lu and Manling Li},
  year={2025},
}
```
