# Algorithm Configurations
VAGEN supports several advantage estimation algorithms, each with different properties for training VLM agents. 

## Algorithm Quick Reference
#### RICO (Traditional RL)
```python
algorithm.adv_estimator=grpo
rollout_manager.use_loss_mask=True
rollout_manager.use_gae_mask=False
rollout_manager.use_multi_turn_reward=False
```
#### AICO (Action-centric Optimization)
```python
algorithm.adv_estimator=masked_gae
rollout_manager.use_loss_mask=True
rollout_manager.use_gae_mask=True
rollout_manager.use_multi_turn_reward=False
```
#### TRICO (Turn-aware Optimization)
```python
algorithm.adv_estimator=bi_level_gae
algorithm.high_level_gamma=0.95
rollout_manager.use_loss_mask=True
rollout_manager.use_gae_mask=True
rollout_manager.use_multi_turn_reward=True
```


## Algorithm Settings
The table below summarizes which features are enabled by default with each algorithm:

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

### Algorithm Configuration Settings

- **multi-turn-reward**: Whether to use multi-turn reward (gives step reward for last token of each turn, instead of summing all rewards for last token of whole trajectory)
  - `rollout_manager.use_multi_turn_reward=True`
- **with_loss_mask**: Whether to use loss mask to only calculate the loss of tokens output by the models
  - `rollout_manager.use_loss_mask=True`
- **with_loss_mask**: Whether to use gae mask to only calculate the gae of tokens output by the models
  - `rollout_manager.use_gae_mask=True`