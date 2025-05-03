"""
Utility functions for config loading and management.
"""

import yaml
import json
import os
from typing import Dict, Any

def load_config(config_path: str) -> Dict:
    """
    Load configuration from YAML file.

    Args:
        config_path: Path to configuration file

    Returns:
        Configuration dictionary
    """
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config

def save_configs(output_dir: str, model_config: Dict, inference_config: Dict, args: Any) -> None:
    """
    Save configurations to output directory.

    Args:
        output_dir: Output directory
        model_config: Model configuration
        inference_config: Inference configuration
        args: Command line arguments
    """
    # Save model config
    with open(os.path.join(output_dir, "model_config.yaml"), "w") as f:
        yaml.dump(model_config, f, default_flow_style=False)

    # Save inference config
    with open(os.path.join(output_dir, "inference_config.yaml"), "w") as f:
        yaml.dump(inference_config, f, default_flow_style=False)

    # Save command line args
    with open(os.path.join(output_dir, "args.json"), "w") as f:
        json.dump(vars(args), f, indent=2)