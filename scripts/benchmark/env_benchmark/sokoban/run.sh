#!/bin/bash
# Setup script for environment benchmark

# Create required directories
mkdir -p env_benchmark_results
mkdir -p data

echo "Generating datasets for benchmark..."

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Generate dataset
python -m vagen.env.create_dataset \
    --yaml_path "$SCRIPT_DIR/dataset_config.yaml" \
    --train_path data/sokoban-vision-benchmark/train.parquet \
    --test_path data/sokoban-vision-benchmark/test.parquet \
    --force_gen

# Set environment variables
export PYTHONPATH=$(pwd):$PYTHONPATH

# Run benchmark
echo "Running environment benchmark"
python -m vagen.utils_benchmark.env_benchmark --config "$SCRIPT_DIR/benchmark_config.yaml"

echo "Benchmark complete. Results saved to env_benchmark_results directory."