#!/bin/bash
#SBATCH --partition=gpuA100x4
#SBATCH --account=bgig-delta-gpu
#SBATCH --gpus=4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=128G
#SBATCH --time=48:00:00
#SBATCH --job-name=sokoban_grpo
#SBATCH --output=sokoban_%j.log

conda activate vagen && bash examples/sokoban/train_grpo_qwen25vl3b_test.sh