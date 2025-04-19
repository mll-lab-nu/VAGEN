## Reproduce Experiments
To reproduce our reults, please go to release branch of verl and v25.3.25 of vagen

```bash
cd ../verl
git checkout release
cd ../VAGEN
git checkout v25.3.25

wandb login # login into wandb

# Then, you can run
bash vagen/examples/release_experiments/gae.sh # rico-gae
bash vagen/examples/release_experiments/grpo_mask_loss.sh # rico-grpo + loss mask
bash vagen/examples/release_experiments/grpo.sh # rico-grpo
bash vagen/examples/release_experiments/mask_gae_mask_loss_bi_level.sh # trico - turn reward
bash vagen/examples/release_experiments/mask_gae_mask_loss_turnwise_gae.sh # trico - turn reward - bi-level gae
bash vagen/examples/release_experiments/mask_gae_mask_loss_turnwise_reward_bi_level.sh # trico
bash vagen/examples/release_experiments/mask_gae_mask_loss.sh # aico
bash vagen/examples/release_experiments/mask_gae.sh # aico - loss mask
bash vagen/examples/release_experiments/mask_loss.sh # aico - gae mask
```
Each run takes ~4 hours to reach 150 steps on 4 H100s. You can decrease testing frequency to speed up training. Training might be unstable due to loss spikes; we recommend restoring from the latest checkpoint when encountering such cases. We will resolve this issue in future work (see roadmap).

### Training Configuration

We used the following settings in our experiments:

- **Model**: Qwen 2.5 VL-instruction 3B
- **Environment**: Visual Sokoban (puzzle-solving task)
- **Rewards**: Box on target (+1.0), All boxes placed (+10.0), Format correct (+0.5), Step penalty (-0.1)
- **Hyperparameters**: `γ_turn`=0.95, `γ_token`=1.0, KL penalty=0.001, Actor LR=1e-6, Critic LR=1e-5


Make sure your are in the right branch (v25.3.25) and use same setting before reproduce the experiment. The result should be similar as follow:


<img width="800" alt="image" src="../public/1.png" />

<img width="800" alt="image" src="../public/2.png" />

<img width="800" alt="image" src="../public/3.png" />
