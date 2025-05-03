import yaml
import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)

def load_yaml_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from a YAML file.
    
    Args:
        config_path: Path to YAML configuration file
        
    Returns:
        Dictionary containing the configuration
    """
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        logger.info(f"Successfully loaded config from {config_path}")
        return config
        
    except Exception as e:
        logger.error(f"Failed to load config from {config_path}: {e}")
        raise