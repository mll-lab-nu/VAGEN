#!/bin/bash
set -e

echo "Starting installation process..."

# echo "Initializing git submodules..."
# git submodule update --init

# echo "Installing verl package..."
# cd verl
# pip install -e .
# cd ../

echo "Installing vagen dependencies..."
pip install 'qwen-vl-utils'
pip install 'mathruler'
pip install 'gym'
pip install 'gym-sokoban'
pip install 'matplotlib'
pip install 'gymnasium'

echo "Installing flash-attn with no build isolation..."
pip install flash-attn --no-build-isolation

echo "Installing vagen package..."
pip install -e .

echo "Installation complete!"