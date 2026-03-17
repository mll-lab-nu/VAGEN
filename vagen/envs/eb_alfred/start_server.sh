#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

cp "$SCRIPT_DIR/xorg0.conf" /tmp/xorg0.conf
cp "$SCRIPT_DIR/xorg1.conf" /tmp/xorg1.conf

Xorg -noreset +extension GLX -config /tmp/xorg0.conf :0 &
Xorg -noreset +extension GLX -config /tmp/xorg1.conf :1 &

sleep 2

python -m vagen.envs.eb_alfred.serve \
    --port 8000 \
    --capacity 90 \
    --startup-concurrency 6 \
    --x-displays 0,1
