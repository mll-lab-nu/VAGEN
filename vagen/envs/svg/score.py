import numpy as np
import cv2


def calculate_structural_accuracy(gt_im, gen_im):
    """Edge-based IoU. Range [0, 1]."""
    gt_gray = np.array(gt_im.convert('L'))
    gen_gray = np.array(gen_im.convert('L'))

    gt_edges = cv2.Canny(gt_gray, 100, 200)
    gen_edges = cv2.Canny(gen_gray, 100, 200)

    intersection = np.logical_and(gt_edges, gen_edges).sum()
    union = np.logical_or(gt_edges, gen_edges).sum()

    return intersection / union if union > 0 else 0


def calculate_total_score(gt_im, gen_im, gt_code, gen_code, score_config,
                          dino_model=None, dreamsim_model=None):
    """Calculate weighted similarity score using DINO + DreamSim + structural."""
    devices = score_config.get("device", {"dino": "cuda:0", "dreamsim": "cuda:0"})

    weights = {
        "dino": score_config.get("dino_weight", 0.0),
        "structural": score_config.get("structural_weight", 0.0),
        "dreamsim": score_config.get("dreamsim_weight", 0.0),
    }

    scores = {
        "dino_score": 0.0,
        "structural_score": 0.0,
        "dreamsim_score": 0.0,
        "total_score": 0.0,
    }

    if dino_model is not None:
        scores["dino_score"] = float(
            dino_model.calculate_DINOv2_similarity_score(gt_im=gt_im, gen_im=gen_im)
        )
    if dreamsim_model is not None:
        scores["dreamsim_score"] = float(
            dreamsim_model.calculate_similarity_score(gt_im=gt_im, gen_im=gen_im)
        )
    if weights["structural"] > 0:
        scores["structural_score"] = max(0.0, float(calculate_structural_accuracy(gt_im, gen_im)))

    weighted_sum = (
        scores["dino_score"] * weights["dino"]
        + scores["structural_score"] * weights["structural"]
        + scores["dreamsim_score"] * weights["dreamsim"]
    )
    scores["total_score"] = max(0.0, weighted_sum)
    return scores
