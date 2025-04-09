import re
import json
import logging
import os
from pathlib import Path
from typing import Dict, Tuple, List, Optional, Any

def extract_svg_code(text: str) -> str:
    """Extract SVG code from the response text
    
    Args:
        text: Raw text that may contain SVG code
        
    Returns:
        Extracted SVG code or empty string if none found
    """
    svg_match = re.search(r'<svg.*?</svg>', text, re.DOTALL)
    if svg_match:
        return svg_match.group(0)

    if '<svg' in text and '</svg>' in text:
        start_idx = text.find('<svg')
        end_idx = text.rfind('</svg>') + 6  # 6 is the length of '</svg>'
        if start_idx < end_idx:
            return text[start_idx:end_idx]

    return ""

def parse_llm_response(text: str, special_token_list: List[str], action_sep: str = ",") -> Dict:
    """Parse the raw text response from LLM to extract thinking and answer.
    
    Args:
        text: Raw text from LLM
        special_token_list: List of special tokens to look for
        action_sep: Separator for actions in the answer

    Returns:
        dict with keys: llm_raw_response, think, answer_list, answer
    """
    # Extract content from <think> tags
    think_match = re.search(r'<think>(.*?)</think>', text, re.DOTALL)
    
    # Extract content from <answer> tags
    answer_match = re.search(r'<answer>(.*?)</answer>', text, re.DOTALL)

    answer_list, thinking, answer_content = [], "", ""
    
    if think_match:
        thinking = think_match.group(1).strip()
    
    if answer_match:
        # Get the answer content and split by comma if needed
        answer_content = answer_match.group(1).strip()
        # Split by comma and strip whitespace from each item if separator is comma
        if action_sep == ",":
            answer_list = [item.strip() for item in answer_content.split(',') if item.strip()]
        else:
            answer_list = [answer_content]
    
    return {
        'llm_raw_response': text,
        'answer_list': answer_list,
        'think': thinking,
        'answer': answer_content,
    }

def setup_analysis_logging(env_id: int, data_dir: str) -> Tuple[logging.Logger, logging.Logger]:
    """Set up logging for analysis mode
    
    Args:
        env_id: Unique ID for the environment instance
        data_dir: Directory to store logs
        
    Returns:
        Tuple of (failure_logger, success_logger)
    """
    log_dir = Path(data_dir) / 'analysis_logs'
    os.makedirs(log_dir, exist_ok=True)
    
    # Failure logger
    failure_logger = logging.getLogger(f'svg_failure_{env_id}')
    failure_logger.setLevel(logging.INFO)
    
    if not failure_logger.handlers:
        failure_handler = logging.FileHandler(log_dir / 'failure_cases.log')
        failure_handler.setFormatter(logging.Formatter('%(message)s'))
        failure_logger.addHandler(failure_handler)
    
    # Success logger
    success_logger = logging.getLogger(f'svg_success_{env_id}')
    success_logger.setLevel(logging.INFO)
    
    if not success_logger.handlers:
        success_handler = logging.FileHandler(log_dir / 'success_cases.log')
        success_handler.setFormatter(logging.Formatter('%(message)s'))
        success_logger.addHandler(success_handler)
        
    return failure_logger, success_logger

def close_loggers(loggers: List[logging.Logger]) -> None:
    """Close all loggers and remove handlers
    
    Args:
        loggers: List of logger instances to close
    """
    for logger in loggers:
        if logger:
            for handler in logger.handlers:
                handler.close()
                logger.removeHandler(handler)

def log_failure(logger: Optional[logging.Logger], failure_info: Dict[str, Any]) -> None:
    """Log a failure case
    
    Args:
        logger: Logger instance or None
        failure_info: Information about the failure
    """
    if logger:
        logger.info(json.dumps(failure_info))

def log_success(logger: Optional[logging.Logger], success_info: Dict[str, Any]) -> None:
    """Log a success case
    
    Args:
        logger: Logger instance or None
        success_info: Information about the success
    """
    if logger:
        logger.info(json.dumps(success_info))