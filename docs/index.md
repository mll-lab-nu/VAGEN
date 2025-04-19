# Welcome to VAGEN Documentation!

## Introduction
VAGEN is a multi-turn reinforcement learning framework designed for training Visual Language Model (VLM) agents efficiently.

## Document Structure

### Quick Strat
- [Installation and Run Experiment](run-exp.md): Get VAGEN up and running

### Configurations
- [General Configuration](configs/general-config.md): Understanding VAGEN's configuration system
- [Algorithm Configuration](configs/algo-config.md): Configure different algorithms

### Environments
- [Create your Own Environment](envs/create-env.md): Build custom environments
- [Create your Own Service](envs/create-service.md): Scale your training infrastructure

### Experiments
- [Reproduce Experiments](reproduce-exp.md): Reproduce our experiments


#### Comparison of Algorithms

| **Feature** | **PPO** | **RICO** | **TRICO (Ours)** |
| --- | --- | --- | --- |
| **Sequence Structure** | Single response | Multiple turn interaction | Multiple turn interaction |
| **LM output** | No special structure | `<think>...</think><ans>...</ans>` | `<think>...</think><ans>...</ans><eoa>` |
| **Discounting** | Single discount rate | Single discount rate | Bi-level discounting |
| **Optimization** | All tokens equally | All tokens equally | Selective token optimization |


## Citation

If you find VAGEN useful, we appreciate it if you could cite our work at:

```bibtex
@misc{VAGEN,
  title={VAGEN: Training VLM Agents with Multi-Turn Reinforcement Learning},
  author={Kangrui Wang* and Pingyue Zhang* and Zihan Wang* and Qineng Wang* and Linjie Li* and Zhengyuan Yang and Chi Wan and Yiping Lu and Manling Li},
  year={2025},
}
```

## License
Licensed under the MIT License. 