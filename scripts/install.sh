#!/bin/bash
set -e

echo "Starting installation process..."

# Initialize and update git submodules
echo "Initializing git submodules..."
git submodule update --init

# First install verl subdirectory as editable package
echo "Installing verl package..."
cd verl
pip install -e .
cd ../

# Now install the main requirements
echo "Installing vagen dependencies..."
pip install 'transformers>=4.49.0'
pip install 'vllm>=0.7.3'
pip install 'qwen-vl-utils'
pip install 'mathruler'

# Install flash-attn with special flags
echo "Installing flash-attn with no build isolation..."
pip install flash-attn --no-build-isolation

# Install the main package
echo "Installing vagen package..."
pip install -e .

echo "Installation complete!"