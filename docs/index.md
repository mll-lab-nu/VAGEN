# Welcome to VAGEN Documentation!

VAGEN is a multi-turn reinforcement learning framework designed for training Visual Language Model (VLM) agents efficiently.

## Quick Navigation

- [Run Experiment](run-exp.md)
- [Create your Own Environment](create-env.md)
- [Configuration](config.md)

Use the links above to explore the core functionalities of the project.

## Algorithm Settings

VAGEN supports several advantage estimation algorithms, each with different properties for training VLM agents. The table below summarizes which features are enabled by default with each algorithm:

| Setting           | GRPO | GAE | Bi-Level GAE | Turn-Wise GAE | Masked-GAE |
|-------------------|------|-----|--------------|---------------|------------|
| with_loss_mask    | ✓    | ✓   | ✓            | ✓             | ✓          |
| multi-turn-reward | ✗    | ✓   | ✓            | ✓             | ✓          |
| with_gae_mask     | ✗    | ✗   | ✓            | ✓             | ✓          |

### Algorithm Configurations
#### RICO (Traditional RL)
```
algorithm.adv_estimator=grpo
rollout_manager.use_loss_mask=True
rollout_manager.use_gae_mask=False
rollout_manager.use_multi_turn_reward=False
```
#### AICO (Action-centric Optimization)
```
algorithm.adv_estimator=masked_gae
rollout_manager.use_loss_mask=True
rollout_manager.use_gae_mask=True
rollout_manager.use_multi_turn_reward=False
```
#### TRICO (Turn-aware Optimization)
```
algorithm.adv_estimator=bi_level_gae
algorithm.high_level_gamma=0.95
rollout_manager.use_loss_mask=True
rollout_manager.use_gae_mask=True
rollout_manager.use_multi_turn_reward=True
```
