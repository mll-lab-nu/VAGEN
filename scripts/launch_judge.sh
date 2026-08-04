#!/bin/bash
# A small instruct model that turns the agent's descriptions into structured items.
#
# It is a parser, not part of training, so it gets one GPU and a small share of it --
# the rollout and the critic need the rest. Reachable at the base_url the trainer's
# state_reward config points at.
#
# Instruct-only on purpose: a thinking model would spend its budget reasoning about a
# format conversion, on the critical path of every turn of every rollout.
set -x
MODEL=${MODEL:-Qwen/Qwen3-4B-Instruct-2507}
PORT=${PORT:-8123}
GPU=${GPU:-0}

CUDA_VISIBLE_DEVICES=$GPU $HOME/miniconda3/envs/verl/bin/python -m sglang.launch_server \
  --model-path $MODEL \
  --port $PORT \
  --mem-fraction-static 0.25 \
  --max-running-requests 64 \
  --attention-backend flashinfer \
  --log-level warning
