import numpy as np
import cv2
import os
from vagen.env.svg.dino import DINOScoreCalculator


def calculate_structural_accuracy(gt_im, gen_im):
    "range from 0 - 1"
    gt_gray = np.array(gt_im.convert('L'))
    gen_gray = np.array(gen_im.convert('L'))
    
    gt_edges = cv2.Canny(gt_gray, 100, 200)
    gen_edges = cv2.Canny(gen_gray, 100, 200)
    
    intersection = np.logical_and(gt_edges, gen_edges).sum()
    union = np.logical_or(gt_edges, gen_edges).sum()
    
    return intersection / union if union > 0 else 0


def calculate_total_score(gt_im, gen_im, gt_code, gen_code, score_config, dino_model=None):
    """
    Calculate all metrics and return a comprehensive score
    
    Args:
        gt_im: Ground truth image
        gen_im: Generated image
        gt_code: Ground truth SVG code
        gen_code: Generated SVG code
        score_config: Dictionary containing scoring parameters
            - model_size: small, base, large
            - dino_only: Whether to use only DINO for scoring
            - dino_weight: Weight for DINO score
            - structural_weight: Weight for structural score
        dino_model: Pre-loaded DINO model (optional)
        
    Returns:
        dict: Dictionary of all scores including the total weighted score
    """
    # Get configuration parameters with defaults
    model_size = score_config.get("model_size", "small")
    dino_only = score_config.get("dino_only", False)
    
    # Define default weights based on model size
    default_weights = {
        "small": {"dino": 3.0, "structural": 7.0},
        "base": {"dino": 5.0, "structural": 5.0},
        "large": {"dino": 6.0, "structural": 4.0}
    }
    
    # Get weights with defaults
    weights = {
        "dino": score_config.get("dino_weight", default_weights[model_size]["dino"]),
        "structural": score_config.get("structural_weight", default_weights[model_size]["structural"])
    }
    
    # Initialize scores
    scores = {
        "dino_score": 0.0,
        "structural_score": 0.0,
        "total_score": 0.0
    }
    
    # Calculate DINO score if needed
    if weights["dino"] > 0:
        if dino_model is None:
            from vagen.env.svg.dino import get_dino_model
            dino_model = get_dino_model(model_size)
        scores["dino_score"] = float(dino_model.calculate_DINOv2_similarity_score(gt_im=gt_im, gen_im=gen_im))
    
    # If DINO only mode, return only DINO score
    if dino_only:
        scores["total_score"] = scores["dino_score"]
        return scores
    
    # Calculate structural score if needed
    if weights["structural"] > 0:
        scores["structural_score"] = max(0.0, float(calculate_structural_accuracy(gt_im, gen_im)))
    
    # Calculate weighted total score
    weighted_sum = (
        scores["dino_score"] * weights["dino"] +
        scores["structural_score"] * weights["structural"]
    )
    scores['total_score'] = max(0.0, weighted_sum)
    
    return scores