# My Project with VERL Submodule

This repository includes [VERL (Volcengine Reinforcement Learning)](https://github.com/volcengine/verl) as a submodule.

## Setup

To clone this repository with its submodules:

```bash
git clone --recursive [your-repo-url]
```

Or if you've already cloned it:

```bash
git submodule update --init --recursive
```


## TODO
1. Remove <box_start> and <box_end> in the rollout
2. Loss mask
3. Implement the validation in ray_trainer
4. Transfer the old ray_trainer file from RAGEN to VAGEN
    - Implement the metric for wandb
5. Add environment specific metrics for wandb logging

## NOTE
Does not support use_dynamic_bsz for now