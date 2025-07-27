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
pip install flash-attn==2.7.4.post1

pip install timm # for internvl

echo "Installing vagen package..."
pip install -e .

echo "Installing Sokoban dependencies"
pip install 'gym'
pip install 'gym-sokoban'

echo "Installing Frozenlake dependencies"
pip install 'gymnasium'
pip install "gymnasium[toy-text]"

pip install together # together ai api for process reward
echo "Installation complete, to install dependencies for other environments, refer to env/readme"
