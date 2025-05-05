import numpy as np
import cv2
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


def calculate_total_score(gt_im, gen_im, gt_code, gen_code, score_config, dino_model=None, dreamsim_model=None):
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
            - dreamsim_weight: Weight for DreamSim score
            - device: Dictionary with keys "dino" and "dreamsim" specifying device
        dino_model: Pre-loaded DINO model (optional)
        dreamsim_model: Pre-loaded DreamSim model (optional)
        
    Returns:
        dict: Dictionary of all scores including the total weighted score
    """
    # Get configuration parameters with defaults
    model_size = score_config.get("model_size", "small")
    dino_only = score_config.get("dino_only", False)
    
    # Get device configuration with defaults
    devices = score_config.get("device", {"dino": "cuda:0", "dreamsim": "cuda:0"})
    dino_device = devices.get("dino", "cuda:0")
    dreamsim_device = devices.get("dreamsim", "cuda:0")
    
    # Define default weights based on model size
    default_weights = {
        "small": {"dino": 3.0, "structural": 7.0, "dreamsim": 5.0},
        "base": {"dino": 5.0, "structural": 5.0, "dreamsim": 5.0},
        "large": {"dino": 6.0, "structural": 4.0, "dreamsim": 5.0}
    }
    
    # Get weights with defaults
    weights = {
        "dino": score_config.get("dino_weight", default_weights[model_size]["dino"]),
        "structural": score_config.get("structural_weight", default_weights[model_size]["structural"]),
        "dreamsim": score_config.get("dreamsim_weight", default_weights[model_size]["dreamsim"])
    }
    
    # Initialize scores
    scores = {
        "dino_score": 0.0,
        "structural_score": 0.0,
        "dreamsim_score": 0.0,
        "total_score": 0.0
    }
    
    # Calculate DINO score if needed
    if weights["dino"] > 0:
        if dino_model is None:
            from vagen.env.svg.dino import get_dino_model
            dino_model = get_dino_model(model_size, device=dino_device)
        scores["dino_score"] = float(dino_model.calculate_DINOv2_similarity_score(gt_im=gt_im, gen_im=gen_im))
    
    # Calculate DreamSim score if needed
    if weights["dreamsim"] > 0:
        if dreamsim_model is None:
            from vagen.env.svg.dreamsim import get_dreamsim_model
            dreamsim_model = get_dreamsim_model(device=dreamsim_device)
        scores["dreamsim_score"] = float(dreamsim_model.calculate_similarity_score(gt_im=gt_im, gen_im=gen_im))
    
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
        scores["structural_score"] * weights["structural"] +
        scores["dreamsim_score"] * weights["dreamsim"]
    )
    scores['total_score'] = max(0.0, weighted_sum)
    
    return scores


def calculate_total_score_batch(gt_images, gen_images, gt_codes, gen_codes, score_configs, dino_model=None,
                                dreamsim_model=None):
    """
    Calculate scores for multiple image pairs in batch mode

    Args:
        gt_images: List of ground truth images
        gen_images: List of generated images
        gt_codes: List of ground truth SVG codes
        gen_codes: List of generated SVG codes
        score_configs: List of scoring parameters dictionaries
        dino_model: Pre-loaded DINO model (optional)
        dreamsim_model: Pre-loaded DreamSim model (optional)

    Returns:
        List of dictionaries containing all scores
    """
    batch_size = len(gt_images)
    if batch_size == 0:
        return []

    # Verify all inputs have same batch size
    if not (len(gen_images) == len(gt_codes) == len(gen_codes) == len(score_configs) == batch_size):
        raise ValueError("All input lists must have the same length")

    # Initialize results
    batch_results = [{
        "dino_score": 0.0,
        "structural_score": 0.0,
        "dreamsim_score": 0.0,
        "total_score": 0.0
    } for _ in range(batch_size)]

    # Check if we need to calculate DINO scores and get device
    need_dino = False
    dino_device = "cuda:0"
    for score_config in score_configs:
        if score_config.get("dino_weight", 0.0) > 0:
            need_dino = True
            devices = score_config.get("device", {"dino": "cuda:0", "dreamsim": "cuda:0"})
            dino_device = devices.get("dino", "cuda:0")
            break

    # Check if we need to calculate DreamSim scores and get device
    need_dreamsim = False
    dreamsim_device = "cuda:0"
    for score_config in score_configs:
        if score_config.get("dreamsim_weight", 0.0) > 0:
            need_dreamsim = True
            devices = score_config.get("device", {"dino": "cuda:0", "dreamsim": "cuda:0"})
            dreamsim_device = devices.get("dreamsim", "cuda:0")
            break

    # Calculate DINO scores in batch if needed
    if need_dino:
        if dino_model is None:
            from vagen.env.svg.dino import get_dino_model
            # Default to small model size if not specified
            model_size = score_configs[0].get("model_size", "small") if score_configs else "small"
            dino_model = get_dino_model(model_size, device=dino_device)

        # Calculate all DINO scores at once using batch processing
        dino_scores = dino_model.calculate_batch_scores(gt_images, gen_images)

        # Assign scores to results
        for i, score in enumerate(dino_scores):
            batch_results[i]["dino_score"] = float(score)

    # Calculate DreamSim scores in batch if needed
    if need_dreamsim:
        if dreamsim_model is None:
            from vagen.env.svg.dreamsim import get_dreamsim_model
            dreamsim_model = get_dreamsim_model(device=dreamsim_device)

        # Calculate all DreamSim scores at once using batch processing
        dreamsim_scores = dreamsim_model.calculate_batch_scores(gt_images, gen_images)

        # Assign scores to results
        for i, score in enumerate(dreamsim_scores):
            batch_results[i]["dreamsim_score"] = float(score)

    # Calculate structural scores and total scores
    for i in range(batch_size):
        score_config = score_configs[i]
        result = batch_results[i]

        # Check if DINO-only mode
        dino_only = score_config.get("dino_only", False)
        if dino_only:
            result["total_score"] = result["dino_score"]
            continue

        # Get model size for default weights
        model_size = score_config.get("model_size", "small")

        # Define default weights based on model size
        default_weights = {
            "small": {"dino": 3.0, "structural": 7.0, "dreamsim": 5.0},
            "base": {"dino": 5.0, "structural": 5.0, "dreamsim": 5.0},
            "large": {"dino": 6.0, "structural": 4.0, "dreamsim": 5.0}
        }

        # Get weights with defaults
        weights = {
            "dino": score_config.get("dino_weight", default_weights[model_size]["dino"]),
            "structural": score_config.get("structural_weight", default_weights[model_size]["structural"]),
            "dreamsim": score_config.get("dreamsim_weight", default_weights[model_size]["dreamsim"])
        }

        # Calculate structural score if needed
        if weights["structural"] > 0:
            from vagen.env.svg.score import calculate_structural_accuracy
            result["structural_score"] = max(0.0, float(calculate_structural_accuracy(gt_images[i], gen_images[i])))

        # Calculate weighted total score
        weighted_sum = (
                result["dino_score"] * weights["dino"] +
                result["structural_score"] * weights["structural"] +
                result["dreamsim_score"] * weights["dreamsim"]
        )
        result["total_score"] = max(0.0, weighted_sum)

    return batch_results