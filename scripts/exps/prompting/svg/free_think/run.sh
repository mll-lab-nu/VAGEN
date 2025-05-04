export VLLM_ATTENTION_BACKEND=XFORMERS
export PYTHONHASHSEED=0

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Extract experiment name from the path
# This will take the last 3 parts of the path: format/sokoban/free_think
EXPERIMENT_NAME=$(echo $SCRIPT_DIR | rev | cut -d'/' -f1-2 | rev | tr '/' '-')

echo "Experiment name: $EXPERIMENT_NAME"
# run python -m vagen.server.server in a tmux session first
python -m vagen.env.create_dataset \
    --yaml_path "$SCRIPT_DIR/env_config.yaml" \
    --train_path "data/$EXPERIMENT_NAME/train.parquet" \
    --test_path "data/$EXPERIMENT_NAME/test.parquet" \
    --force_gen

python -m vagen.inference.run_inference \
    --inference_config_path="$(dirname $SCRIPT_DIR)/inference_config.yaml" \
    --model_config_path="$(dirname $SCRIPT_DIR)/model_config.yaml" \
    --val_files_path="data/$EXPERIMENT_NAME/test.parquet" \
    --wandb_path_name="$EXPERIMENT_NAME"