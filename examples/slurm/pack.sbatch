#!/bin/bash
# Run several 4-GPU training runs inside one allocation, split by CUDA_VISIBLE_DEVICES.
#
# Two reasons to pack rather than submit one job each:
#   * a QOS usually caps the number of jobs per user well below the number of GPUs, so
#     packing is what lets a sweep use the GPU quota it actually has;
#   * runs that share a node share the model's page cache, which matters when every one of
#     them reads the same weights off a network filesystem.
#
#   RUNS='name1|env|est|lam|loss|lam_low|harness|extra   name2|...' \
#     sbatch --account=$SLURM_ACCOUNT --qos=$GPU_QOS --nodes=1 \
#            --gpus-per-node=$GPU_TYPE:8   # 8 = the whole node; two 4-GPU runs --cpus-per-task=96 --mem=800G \
#            --time=48:00:00 --requeue examples/slurm/pack.sbatch
#
# One pipe-separated record per run, whitespace between records. Empty fields are allowed
# ('name|sokoban|token_level_gae|1.0||||'). GPUS_PER_RUN defaults to 4.
set -uo pipefail
: "${CODE_ROOT:?}" "${DATA_ROOT:?}" "${RUNS:?set RUNS to the pipe-separated run list}"
G=${GPUS_PER_RUN:-4}
# Not $BASH_SOURCE: Slurm copies the batch script into /var/spool/slurmd/job<id>/ before
# running it, so the script's own path says nothing about where the repo is.
TRAIN=$CODE_ROOT/VAGEN/examples/slurm/train.sbatch
[ -f "$TRAIN" ] || { echo "no train.sbatch at $TRAIN -- is CODE_ROOT right?"; exit 1; }

i=0
for rec in $RUNS; do
  IFS='|' read -r name env est lam loss low harness extra <<< "$rec"
  first=$((i * G)); last=$((first + G - 1))
  dev=$(seq -s, "$first" "$last")
  echo "[pack] $name -> GPUs $dev"
  CUDA_VISIBLE_DEVICES=$dev SLURM_GPUS_ON_NODE=$G \
  EXP_NAME=$name ENV_NAME=$env ADV_EST=$est LAM=$lam \
  LOSS_MODE=$loss LAM_LOW=$low HARNESS=${harness:-concat} EXTRA_ARGS=$extra \
  STATE_REWARD=off \
    bash "$TRAIN" &
  i=$((i + 1))
  # Stagger: several Ray clusters coming up at once contend for ports and shared memory,
  # and the first one to start warms the weights in page cache for the rest.
  sleep 120
done
wait
