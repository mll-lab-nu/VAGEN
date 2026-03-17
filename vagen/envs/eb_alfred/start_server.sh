#!/bin/bash
# Start the EB-ALFRED server.
# GPUs and Xorg servers are auto-detected and started by serve.py.
# Override with --devices='[0,1]' if needed.
python -m vagen.envs.eb_alfred.serve \
    --port 8000 \
    --capacity 90 \
    --startup_concurrency 6
