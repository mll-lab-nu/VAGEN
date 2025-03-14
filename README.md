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
bash vagen/examples/sokoban/debug_qwen2_5_vl_4gpu_grpo.sh

# Verified on 1 and 4 A100 GPUs
```
## Current Status
1. sokoban-text is runnbale: both single A100 and 4 A100s, grpo (fake), performance is bad
2. sokoban-vision is testing in 4 A100s, minor bugs to fix


## TODO
1. DEBUG: 之前看validation好像有连续2个llm response中间没有user的情况，不知到现在的code fix 没有
1. Make sure loss mask is working correctly
2. Implement real grpo: rollout.n>1
3. Make PPO runnable (modify ppo scripts and code)
4. Add more metrics and image visualization

