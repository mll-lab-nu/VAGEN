import numpy as np
import cv2
import os
from vagen.env.svgdino.dino import DINOScoreCalculator


def calculate_structural_accuracy(gt_im, gen_im):
    "range from 0 - 1"
    gt_gray = np.array(gt_im.convert('L'))
    gen_gray = np.array(gen_im.convert('L'))
    
    gt_edges = cv2.Canny(gt_gray, 100, 200)
    gen_edges = cv2.Canny(gen_gray, 100, 200)
    
    intersection = np.logical_and(gt_edges, gen_edges).sum()
    union = np.logical_or(gt_edges, gen_edges).sum()
    
    return intersection / union if union > 0 else 0


def calculate_color_fidelity(gt_im, gen_im):
    "range from 0 - 1"
    gt_lab = cv2.cvtColor(np.array(gt_im), cv2.COLOR_RGB2LAB)
    gen_lab = cv2.cvtColor(np.array(gen_im), cv2.COLOR_RGB2LAB)
    
    mse = np.mean((gt_lab - gen_lab) ** 2)
    sim = np.exp(-mse / 1000)
    return sim


def calculate_code_efficiency(gt_code, gen_code):
    if not gen_code:
      return 0
    gt_len = len(gt_code)
    gen_len = len(gen_code)

    if gen_len == gt_len:
        return 0.8

    if gen_len < gt_len:
        ratio = gen_len / gt_len 
        score = 0.8 + 0.2 * (1 - ratio)
        return min(score, 1.0)

    ratio = gt_len / gen_len 
    score = 0.8 * ratio
    return max(score, 0.0)

_model_cache = {}

def get_model(model_size, device="cuda"):
    cache_key = f"{model_size}_{device}"
    if cache_key not in _model_cache:
        _model_cache[cache_key] = DINOScoreCalculator(model_size=model_size, device=device)
    return _model_cache[cache_key]

#@TODO make it into class?
def calculate_total_score(gt_im, gen_im, gt_code, gen_code, score_config):
    """
    calculate all metrics on average
    
    Args:
        gt_im: gt image
        gen_im: generated image
        gt_code: gt code
        gen_svg: generated code
        score_config:
          - model_size: small, base, large
          - dino_only: whether only use dino as score
          - dino_weight
          - structural_weight
          - color_weight
          - code_weight
        
    Returns:
        dict: include all scores
    """
    model_size = score_config.get("model_size", "large")
    dino_only = score_config.get("dino_only", False)

    reward_model = get_model(model_size)
    dino_score = reward_model.calculate_DINOv2_similarity_score(gt_im=gt_im, gen_im=gen_im)
    
    if dino_only:
        return {
            'dino_score': dino_score,
            'total_score': dino_score
        }
    
    structural_score = calculate_structural_accuracy(gt_im, gen_im)
    color_score = calculate_color_fidelity(gt_im, gen_im)
    code_score = calculate_code_efficiency(gt_code, gen_code)
    
    default_weights = {
        "small":{"dino": 3.0, "structural": 7.0, "color": 0.0, "code": 0.0},
        "base":{"dino": 5.0, "structural": 5.0, "color": 0.0, "code": 0.0},
        "large":{"dino": 6.0, "structural": 4.0, "color": 0.0, "code": 0.0}
    }

    weights = {
        "dino": score_config.get("dino_weight", default_weights[model_size]["dino"]),
        "structural": score_config.get("structural_weight", default_weights[model_size]["structural"]),
        "color": score_config.get("color_weight", default_weights[model_size]["color"]),
        "code": score_config.get("code_weight", default_weights[model_size]["code"]),
    }

    
    scores = {
        'dino_score': dino_score,
        'structural_score': structural_score,
        'color_score': color_score,
        'code_score': code_score
    }
    
    weighted_sum = (
        dino_score * weights["dino"] +
        structural_score * weights["structural"] +
        color_score * weights["color"] +
        code_score * weights["code"]
    )
        
    scores['total_score'] = max(0.0, weighted_sum)
    
    return scores