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
pip install 'matplotlib'
pip install 'flask'


echo "Installing flash-attn with no build isolation..."
pip install flash-attn --no-build-isolation

echo "Installing vagen package..."
pip install -e .

echo "Installing Sokoban dependencies"
pip install 'gym'
pip install 'gym-sokoban'

echo "Installing Frozenlake dependencies"
pip install 'gymnasium'
pip install "gymnasium[toy-text]"

echo "Installing Navigation dependencies"
pip install ai2thor==5.0.0
pip install numpy==1.25.1

echo "Installing SVG dependencies"
pip install "bs4"
pip install "svgpathtools"
pip install "cairosvg"

echo "Installation complete!"