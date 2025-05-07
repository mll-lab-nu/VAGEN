import numpy as np
import cv2

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
    """
    # Get configuration parameters with defaults
    model_size = score_config.get("model_size", "small")
    
    # Get device configuration with defaults
    devices = score_config.get("device", {"dino": "cuda:0", "dreamsim": "cuda:0"})
    dino_device = devices.get("dino", "cuda:0")
    dreamsim_device = devices.get("dreamsim", "cuda:0")
    
    # Get weights with defaults
    weights = {
        "dino": score_config.get("dino_weight", 0.0),
        "structural": score_config.get("structural_weight", 0.0),
        "dreamsim": score_config.get("dreamsim_weight", 0.0)
    }
    
    # Initialize scores
    scores = {
        "dino_score": 0.0,
        "structural_score": 0.0,
        "dreamsim_score": 0.0,
        "total_score": 0.0
    }
    
    scores["dino_score"] = float(dino_model.calculate_DINOv2_similarity_score(gt_im=gt_im, gen_im=gen_im))
    scores["dreamsim_score"] = float(dreamsim_model.calculate_similarity_score(gt_im=gt_im, gen_im=gen_im))
    
    if weights["structural"] > 0:
        scores["structural_score"] = max(0.0, float(calculate_structural_accuracy(gt_im, gen_im)))
    
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
    Batch score calculation that leverages model batch processing.
    Always calculates all scores for metrics, regardless of weights.
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
    
    if dino_model is None:
        raise ValueError("DINO model must be provided by the service")
    if dreamsim_model is None:
        raise ValueError("DreamSim model must be provided by the service")
    
    dino_scores = dino_model.calculate_batch_scores(gt_images, gen_images)
    dreamsim_scores = dreamsim_model.calculate_batch_scores(gt_images, gen_images)
    
    structural_scores = [calculate_structural_accuracy(gt_images[i], gen_images[i]) 
                          for i in range(batch_size)]

    # Assign scores and calculate total scores
    for i in range(batch_size):
        batch_results[i]["dino_score"] = float(dino_scores[i])
        batch_results[i]["dreamsim_score"] = float(dreamsim_scores[i])
        batch_results[i]["structural_score"] = max(0.0, float(structural_scores[i]))
        
        weights = {
            "dino": score_configs[i].get("dino_weight", 0.0),
            "structural": score_configs[i].get("structural_weight", 0.0),
            "dreamsim": score_configs[i].get("dreamsim_weight", 0.0)
        }

        weighted_sum = (
            batch_results[i]["dino_score"] * weights["dino"] +
            batch_results[i]["structural_score"] * weights["structural"] +
            batch_results[i]["dreamsim_score"] * weights["dreamsim"]
        )
        batch_results[i]["total_score"] = max(0.0, weighted_sum)

    return batch_results