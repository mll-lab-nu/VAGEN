#!/bin/bash
#SBATCH --job-name=vagen_ppo_eb_alfred
#SBATCH --partition=gpuA100x4
#SBATCH --account=bgig-delta
#SBATCH --nodes=1
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=64
#SBATCH --mem=200G
#SBATCH --time=48:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

# ---------------------------------------------------------------
# Before submitting, start the env server on your local machine:
#   python -m vagen.envs.eb_alfred.serve --port 8000
#
# Then create a reverse SSH tunnel from your local machine to the
# Delta login node so the compute node can reach your env server:
#   ssh -R 8000:localhost:8000 jma6@dt-login01.delta.ncsa.illinois.edu
#
# Keep that tunnel open for the duration of the job.
# ---------------------------------------------------------------

set -x

# Load modules
source /sw/rh9.4/python/miniforge3/etc/profile.d/conda.sh
module load miniforge3-python
conda activate /scratch/bgig/jma6/envs/vagen

# Forward the login node's port 8000 to this compute node's localhost:8000
# so the training script can connect to the env server via localhost.
ssh -f -N -L 8000:localhost:8000 dt-login01.delta.ncsa.illinois.edu
echo "SSH tunnel established: localhost:8000 -> dt-login01:8000 -> your local env server"

# Wait a moment for tunnel to be ready
sleep 3

cd /u/jma6/workspace/VAGEN
bash examples/train/eb_alfred/train_ppo_no_concat_qwen25vl3b.sh
