## Setup

```
conda create -n vagen python=3.10 -y
git clone git@github.com:JamesKrW/verl.git
cd verl
pip install -e .
cd ../
git clone git@github.com:JamesKrW/vagen.git
cd vagen
bash scripts/install.sh
```

## Run
```
bash vagen/examples/sokoban/debug_qwen0_5_1_gpu_grpo.sh
bash vagen/examples/sokoban/debug_qwen0_5_4_gpu_ppo.sh

# Verified on 1 and 4 A100 GPUs
```

## TODO
1. Implement real grpo: rollout.n>1
2. Make PPO runnable (modify ppo scripts and code)
3. Add more metrics and image visualization

