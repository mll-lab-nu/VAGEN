# vagen/inference/utils/environment.py

import logging
import pandas as pd
from typing import List, Dict
from pathlib import Path

logger = logging.getLogger(__name__)

def load_environment_configs_from_parquet(val_files_path: str) -> List[Dict]:
    """
    Loads environment configurations from a parquet file.
    Extracts env_name, env_config, and seed from extra_info field.
    
    Args:
        val_files_path: Path to the parquet file containing validation data
        
    Returns:
        List of environment configurations
    """
    env_configs = []
    
    try:
        # Load the parquet file
        df = pd.read_parquet(val_files_path)
        
        for i in range(len(df)):
            row = df.iloc[i].to_dict()
            if 'extra_info' in row and isinstance(row['extra_info'], dict):
                extra_info = row['extra_info']
                config = {
                    "env_name": extra_info.get("env_name"),
                    "env_config": extra_info.get("env_config", {}),
                    "seed": extra_info.get("seed", 42)
                }
                env_configs.append(config)
        
        logger.info(f"Loaded {len(env_configs)} environment configs from {val_files_path}")
        return env_configs
        
    except Exception as e:
        logger.error(f"Failed to load environment configs from {val_files_path}: {e}")
        raise