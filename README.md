<h1 align="center">VAGEN: Training VLM Agents with Multi-Turn Reinforcement Learning</h1>
<!-- <p align="center" style="font-size: 18px;">
  <strong>VAGEN</strong>: Multi-turn Reinforcement Learning for Visual Reasoning Agents<br>
</p> -->
<p align="center">
  <a href="#"><img src="https://img.shields.io/badge/📚_Documentation-4285F4?style=for-the-badge&logoColor=white" alt="Documentation"></a>
  <a href="#"><img src="https://img.shields.io/badge/📝_Blog-FF5722?style=for-the-badge&logoColor=white" alt="Blog"></a>
  <a href="#"><img src="https://img.shields.io/badge/🔍_Post-34A853?style=for-the-badge&logoColor=white" alt="Post"></a>
</p>




VAGEN is a multi-turn reinforcement learning framework designed specifically for training Vision-Language Model (VLM) Agents. Building upon existing approaches for LLM agents like RAGEN, Search-R1, and Agent-R1, VAGEN introduces enhancements that better handle the unique challenges of visual agents.

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

## Comparison of Algorithms

| **Feature** | **PPO** | **RICO** | **TRICO (Ours)** |
| --- | --- | --- | --- |
| **Sequence Structure** | Single response | Multiple turn interaction | Multiple turn interaction |
| **LM output** | No special structure | `<think>...</think><ans>...</ans>` | `<think>...</think><ans>...</ans><eoa>` |
| **Discounting** | Single discount rate | Single discount rate | Bi-level discounting |
| **Optimization** | All tokens equally | All tokens equally | Selective token optimization |

## Getting Started

```bash
# Clone the repository
git clone https://github.com/RAGEN-AI/VAGEN.git
cd VAGEN

# Install dependencies
pip install -e .

# Run an example experiment
python vagen/examples/...
```

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

# Clone VERL repository
git clone git@github.com:JamesKrW/verl.git
cd verl
pip install -e .
cd ../

# Clone VAGEN repository
git clone git@github.com:RAGEN-AI/vagen.git
cd vagen
bash scripts/install.sh
```

## Running Experiments

```bash
# To reproduce our reults, you can run
bash vagen/vagen/examples/release_experiments/gae.sh
bash vagen/vagen/examples/release_experiments/grpo_mask_loss.sh
bash vagen/vagen/examples/release_experiments/grpo.sh
bash vagen/vagen/examples/release_experiments/mask_gae_mask_loss_bi_level.sh
bash vagen/vagen/examples/release_experiments/mask_gae_mask_loss_turnwise_gae.sh
bash vagen/vagen/examples/release_experiments/mask_gae_mask_loss_turnwise_reward_bi_level.sh
bash vagen/vagen/examples/release_experiments/mask_gae_mask_loss.sh
bash vagen/vagen/examples/release_experiments/mask_gae.sh
bash vagen/vagen/examples/release_experiments/mask_loss.sh
```

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


 ## Limitations and Future Work

- Training can be unstable, often requiring early stopping
- We aim to expand evaluation to more diverse visual environments
- Future plans include scaling to larger models and applying TRICO to text-only tasks

## Citation

If you find this work useful, please cite our paper:

```
@misc{...
  title={VAGEN: Training VLM Agents with Multi-Turn Reinforcement Learning},
  author={...},
  year={2025},
  ...
}
```
