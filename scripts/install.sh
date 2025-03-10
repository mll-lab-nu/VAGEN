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
pip install 'transformers>=4.49.0'
pip install 'vllm>=0.7.3'
pip install 'qwen-vl-utils'
pip install 'mathruler'
pip install 'gym'
pip install 'gym-sokoban'
pip install 'matplotlib'

echo "Installing flash-attn with no build isolation..."
pip install flash-attn --no-build-isolation

echo "Installing vagen package..."
pip install -e .

echo "Installation complete!"