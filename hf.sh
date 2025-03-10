#!/bin/bash


HF_BASE_DIR=""


mkdir -p "$HF_BASE_DIR/models"
mkdir -p "$HF_BASE_DIR/datasets"
mkdir -p "$HF_BASE_DIR/hub"

export HF_HOME="$HF_BASE_DIR"
export TRANSFORMERS_CACHE="$HF_BASE_DIR/models"
export HF_DATASETS_CACHE="$HF_BASE_DIR/datasets"
export HF_HUB_CACHE="$HF_BASE_DIR/hub"


echo "HF_HOME: $HF_HOME"
echo "TRANSFORMERS_CACHE: $TRANSFORMERS_CACHE"
echo "HF_DATASETS_CACHE: $HF_DATASETS_CACHE"
echo "HF_HUB_CACHE: $HF_HUB_CACHE"