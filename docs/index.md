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


#### Comparison of Algorithms

| **Feature** | **PPO & GRPO** | **VAGEN-Base** | **VAGEN-Full** |
| --- | --- | --- | --- |
| **Sequence Structure** | Single response | Multiple turn interaction | Multiple turn interaction |
| **LM output** | No special structure | `<think>...</think><ans>...</ans>` | `<think>...</think><ans>...</ans><eoa>` |
| **Discounting** | Single discount rate | Single discount rate | Bi-level discounting |
| **Optimization** | All tokens equally | All tokens equally | Selective token optimization |


## Citation
If you find VAGEN useful, we appreciate it if you could cite our work at:

```bibtex
@misc{wang2025vagen,
  title={Reinforcing Visual State Reasoning for Multi-Turn VLM Agents},
  author={Kangrui Wang* and Pingyue Zhang* and Zihan Wang* and Yaning Gao* and Linjie Li* and Qineng Wang and Hanyang Chen and Chi Wan and Yiping Lu and Zhengyuan Yang and Lijuan Wang and Ranjay Krishna and Jiajun Wu and Li Fei-Fei and Yejin Choi and Manling Li},
  year={2025},
  url={https://arxiv.org/abs/2510.16907}
}
```

## License
Licensed under the MIT License. 