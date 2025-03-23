<h1 align="center">VAGEN</h1>
<p align="center" style="font-size: 18px;">
  <strong>VAGEN</strong>: Multi-turn Reinforcement Learning for Visual Reasoning Agents<br>
</p>
<p align="center">
  <a href="#"><img src="https://img.shields.io/badge/📚_Documentation-4285F4?style=for-the-badge&logoColor=white" alt="Documentation"></a>
  <a href="#"><img src="https://img.shields.io/badge/📝_Blog-FF5722?style=for-the-badge&logoColor=white" alt="Blog"></a>
  <a href="#"><img src="https://img.shields.io/badge/📄_Paper-EA4335?style=for-the-badge&logoColor=white" alt="Paper"></a>
  <a href="#"><img src="https://img.shields.io/badge/🔍_Post-34A853?style=for-the-badge&logoColor=white" alt="Post"></a>
</p>

## Overview

VAGEN is a multi-turn Reinforcement Learning (RL) framework designed to optimize Vision-Language Models (VLMs) for visual reasoning tasks. The framework introduces the **Turn-aware Reason-Interaction Chain Optimization (TRICO)** algorithm, which enhances VLM performance on complex visual reasoning problems.

### Key Innovations

- **Selective Training**: Uses masks to control which tokens contribute to the learning process
- **Selective Reward Assignment**: Assigns turn-wise or trajectory rewards to specific tokens
- **Selective Advantage Calculation**: Implements different advantage calculations for within-turn vs. cross-turn dependencies

## TRICO Overview

VAGEN introduces TRICO, a specialized algorithm for visual agents that extends previous approaches with several key improvements:

<p align="center"><img src="./public/overview_1.pdf" width="800px" alt="TRICO Overview -1 " /></p>
<p align="center"><img src="./public/overview_2.pdf" width="800px" alt="TRICO Overview -2 " /></p>

Building on recent advances in LLM reinforcement learning, TRICO explores several key innovations for visual agents:

* **Selective Training**: Uses masks to control which tokens contribute to the learning process and advantage estimation process
   * **LM**: Loss Mask, state tokens (image tokens and text description tokens) are masked during the critic and actor update
   * **GM**: GAE Mask, state tokens are masked during generalized advantage estimation (GAE)
* **Selective Reward Assignment**: Assign turn-wise reward or trajectory reward to tokens
   * **Turn-wise reward**: Reward is assigned to each turn's <eos> token
   * **Trajectory reward**: Reward is summed up and assigned to the last token of the trajectory
* **Selective Advantage Calculation**: Different advantage calculation for within-turn vs. cross-turn dependencies
   * **Bi-level GAE**: We use two discounting factors:
      * $γ_{turn}$: Estimates advantage for each turn through cross-turn temporal difference learning
      * $γ_{token}$: Estimates advantage for each token through within-turn temporal difference learning
   * **Turn-Wise GAE**: We use $γ_{turn}$ to compute a single advantage through cross-turn temporal difference learning and assign the same advantage for each token in the turn


### Algorithm Features

| **Feature** | **PPO-LLM (Standard)** | **RICO** | **TRICO (Ours)** |
| --- | --- | --- | --- |
| **Sequence Structure** | Single response (y) to prompt (x) | Multiple turn interaction (y₀, x₁, y₁, ...) | Multiple turn interaction with masking |
| **Reasoning Representation** | No special structure | `<think>...</think><ans>...</ans>` | `<think>...</think><ans>...</ans>` |
| **Discounting** | Single discount rate γ | Single discount rate γ | Bi-level: γ_turn and γ_token |
| **Optimization** | All tokens equally | All tokens equally | Selective token optimization |
| **Reward Assignment** | Trajectory | Trajectory | Trajectory / Turn-wise |

## Performance

We evaluated VAGEN on the visual puzzle-solving Sokoban task, demonstrating significant improvements over previous approaches.

<p align="center">
    <img src="./public/performance.png" width="1000px" alt="Performance Results" />
</p>

### Experiment Settings

- **Task:** Sokoban with visual input
- **Reward Engineering:**
  - Box on target: +1.0
  - All boxes placed: +10.0
  - Format correct: +0.5
- **Evaluation Metrics:** Score + Success Rate

## Example Trajectories (LM + GM + Turn-wise Reward + Bi-Level GAE)

The visualizations below show how the agent reasons through sequential steps to solve Sokoban puzzles, cherry picked from validation steps:

<p align="center">
    <img src="./public/example_1.png" width="1000px" alt="Example 1" />
</p>

<p align="center">
    <img src="./public/example_2.png" width="1000px" alt="Example 2" />
</p>

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
# Run experiments with different settings
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

## Conclusion

VAGEN, powered by the TRICO algorithm, is well-suited for visual agent learning because:

1. **Visual agents require multi-turn reasoning**: The agent observes, thinks, acts, and receives updated observations
2. **Not all tokens are equally important**: In visual reasoning, the action tokens are more critical than observation tokens for VLMs to learn
3. **Bi-level time dependencies**: Bi-Level discount factors help model both immediate and long-term consequences

## Future Work

1. Expanding to more environments, real-world applications like GUI agents and embodied agents
2. Scaling to larger models
