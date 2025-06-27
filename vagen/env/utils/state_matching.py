import json
from collections import Counter
import numpy as np
from scipy.optimize import linear_sum_assignment

def calculate_f1_score(total_match_score: float, total_predicted_items: int, total_groundtruth_items: int) -> float:
    """
    Calculates an F1-like score given a total match score (can be fractional),
    total predicted items, and total ground truth items.

    Args:
        total_match_score: The sum of similarity scores from the optimal matching.
                           Can be fractional.
        total_predicted_items: Total number of items (instances) in the prediction.
        total_groundtruth_items: Total number of items (instances) in the ground truth.

    Returns:
        The F1-like score (between 0.0 and 1.0).
    """
    # Handle edge cases where item counts are zero
    total_predicted_items = max(0, total_predicted_items)
    total_groundtruth_items = max(0, total_groundtruth_items)
    total_match_score = max(0.0, total_match_score) # Score cannot be negative

    # If both lists are empty, it's a perfect match.
    if total_predicted_items == 0 and total_groundtruth_items == 0:
        return 1.0

    # If one list is empty but the other is not, F1 is 0.
    # This is covered by the precision/recall calculation where total_match_score will be 0.
    # But explicitly checking can clarify intent.
    # if total_predicted_items == 0 or total_groundtruth_items == 0:
    #     return 0.0 # total_match_score will be 0 anyway, leading to P=0 or R=0

    # Calculate Precision and Recall based on the total match score
    # Precision: How much of the predicted "stuff" was correct?
    precision = total_match_score / total_predicted_items if total_predicted_items > 0 else 0.0
    # Recall: How much of the ground truth "stuff" was captured?
    recall = total_match_score / total_groundtruth_items if total_groundtruth_items > 0 else 0.0

    # Calculate F1 score
    if precision + recall == 0:
        return 0.0

    f1 = 2 * (precision * recall) / (precision + recall)
    return f1

def calculate_item_similarity(pred_item: dict, gt_item: dict) -> float:
    """
    Calculates a similarity score between a predicted item and a ground truth item.
    Score is (vertical_match + horizontal_match) / 2.0

    Args:
        pred_item: Predicted item dict {'vertical_relation': v, 'horizontal_relation': h}
        gt_item: Ground truth item dict {'vertical_relation': v, 'horizontal_relation': h}

    Returns:
        A score between 0.0 (no match) and 1.0 (full match).
    """
    pv = pred_item.get("vertical_relation")
    ph = pred_item.get("horizontal_relation")
    gv = gt_item.get("vertical_relation")
    gh = gt_item.get("horizontal_relation")

    v_match = 1 if pv is not None and gv is not None and pv == gv else 0
    h_match = 1 if ph is not None and gh is not None and ph == gh else 0

    # Note: If a relation is None in one but not the other, they don't match for that relation.
    # If both are None, they also don't contribute to the match score for that relation.
    # This naturally results from the `is not None and ... and ==` check.

    return (v_match + h_match) / 2.0


def calculate_visual_reasoning_reward_bipartite(
    predicted_list: list[dict],
    groundtruth_list: list[dict],
    object_weights: dict[str, float] # e.g., {"target": 0.6, "box": 0.4} or {"target": 0.5, "hole": 0.5}
) -> float:
    """
    Calculates a weighted F1-like score comparing predicted and groundtruth
    lists of relative object positions. Handles multiple objects of the same ID
    by using maximum weight bipartite matching based on paired relation similarity.

    Args:
        predicted_list: A list of dictionaries from the LLM parser output.
                        [{'object_id': 'target', 'vertical_relation': 'above', 'horizontal_relation': 'left'}, ...]
        groundtruth_list: A list of dictionaries representing the ground truth.
                        [{'object_id': 'target', 'vertical_relation': 'above', 'horizontal_relation': 'left'}, ...]
        object_weights: A dictionary mapping object_id (e.g., "target", "box", "hole")
                        to its weight in the final score calculation. Weights should sum to 1.0
                        for the objects relevant to the environment.

    Returns:
        A weighted reward score between 0.0 and 1.0 (assuming weights sum to 1.0).
        Returns 0.0 if no relevant objects (based on weights) are found in either list.
    """
    if not isinstance(predicted_list, list) or not isinstance(groundtruth_list, list):
        print("Warning: Inputs must be lists.")
        return 0.0

    # Group predicted and groundtruth items by object_id
    # Only include items where object_id is in object_weights and has required keys
    predicted_by_id = {}
    for item in predicted_list:
        obj_id = item.get("object_id")
        v_rel = item.get("vertical_relation")
        h_rel = item.get("horizontal_relation")
        # Only include if obj_id is relevant AND at least one relation key exists (even if value is None)
        if obj_id and obj_id in object_weights and ("vertical_relation" in item or "horizontal_relation" in item):
             if obj_id not in predicted_by_id:
                 predicted_by_id[obj_id] = []
             # Store the item (could include None values)
             predicted_by_id[obj_id].append({"vertical_relation": v_rel, "horizontal_relation": h_rel})


    groundtruth_by_id = {}
    for item in groundtruth_list:
        obj_id = item.get("object_id")
        v_rel = item.get("vertical_relation")
        h_rel = item.get("horizontal_relation")
        # Only include if obj_id is relevant AND at least one relation key exists (even if value is None)
        if obj_id and obj_id in object_weights and ("vertical_relation" in item or "horizontal_relation" in item):
             if obj_id not in groundtruth_by_id:
                 groundtruth_by_id[obj_id] = []
             # Store the item
             groundtruth_by_id[obj_id].append({"vertical_relation": v_rel, "horizontal_relation": h_rel})

    weighted_f1_sum = 0.0
    total_relevant_weight = 0.0 # Sum of weights for objects actually considered

    # Get all unique object IDs that are relevant (have weights) and are in either list
    relevant_object_ids = set(predicted_by_id.keys()).union(set(groundtruth_by_id.keys()))

    # Calculate total weight for normalization based *only* on objects present in *either* list
    total_weight_for_normalization = sum(object_weights.get(obj_id, 0.0) for obj_id in relevant_object_ids)

    if total_weight_for_normalization == 0.0:
         # This occurs if no relevant objects (from object_weights) were found in either list
         # If *both* input lists were [] but object_weights has keys, this will return 0.
         # If *both* input lists were [], and object_weights is {}, this also returns 0.
         # If object_weights has keys, and both lists are empty, arguably reward should be 1.0?
         # Let's adjust: if both lists are empty and object_weights is not empty, reward is 1.0.
         if not predicted_list and not groundtruth_list and object_weights:
             # This assumes an empty list means perfect match for the lack of predicted objects.
             # Could be debated, but let's go with 1.0 for now.
             return 1.0
         return 0.0 # Otherwise, no relevant objects means no score contribution

    for obj_id in relevant_object_ids:
        pred_items = predicted_by_id.get(obj_id, [])
        gt_items = groundtruth_by_id.get(obj_id, [])
        weight = object_weights.get(obj_id, 0.0) # Get weight, default to 0 if not in weights

        # If weight is 0, it won't contribute to weighted_f1_sum or total_weight_for_normalization, skip.
        if weight == 0.0:
             continue

        total_predicted_items = len(pred_items)
        total_groundtruth_items = len(gt_items)

        # Calculate total match score using bipartite matching
        total_match_score_id = 0.0

        if total_predicted_items > 0 and total_groundtruth_items > 0:
            # Create a weight matrix (M x N) where M = predicted, N = groundtruth
            # weight_matrix[i][j] is the similarity between predicted item i and gt item j
            weight_matrix = np.zeros((total_predicted_items, total_groundtruth_items))
            for i in range(total_predicted_items):
                for j in range(total_groundtruth_items):
                    weight_matrix[i, j] = calculate_item_similarity(pred_items[i], gt_items[j])

            # Use linear_sum_assignment to find the maximum weight matching
            # It minimizes cost, so use the negative of the weight matrix
            row_indices, col_indices = linear_sum_assignment(-weight_matrix)

            # The total weight of the optimal matching is the sum of weights at the assigned indices
            total_match_score_id = weight_matrix[row_indices, col_indices].sum()

        elif total_predicted_items == 0 and total_groundtruth_items == 0:
             # Both lists are empty for this object type, perfect match for this object type
             f1_id = 1.0
             weighted_f1_sum += weight * f1_id
             continue # Move to next object_id, F1 already calculated as 1.0

        # Calculate F1 for this object ID based on matched items
        # Note: total_match_score_id can be fractional
        f1_id = calculate_f1_score(total_match_score_id, total_predicted_items, total_groundtruth_items)

        weighted_f1_sum += weight * f1_id

    # Normalize by the total weight of all relevant objects that were processed
    # This is sum of weights for objects in relevant_object_ids (which includes objects with count 0 in both lists)
    # if total_weight_for_normalization == 0.0: # Already handled at the beginning
    #      return 0.0

    return weighted_f1_sum / total_weight_for_normalization # This gives a score between 0 and 1


def calculate_f1_with_max_matching(p_list, gt_list, match_func):
    """
    Calculate F1 score between predicted list and ground truth list using maximum matching algorithm
    
    Args:
        p_list: predicted list
        gt_list: ground truth list  
        match_func: matching function that takes two parameters (item1, item2) and returns boolean
        
    Returns:
        dict: dictionary containing precision, recall, f1, matches, total_predicted, total_ground_truth
    """
    
    # Build bipartite graph matching matrix
    # Use greedy algorithm to find maximum matching
    used_gt_indices = set()
    used_p_indices = set()
    matches = 0
    
    # Find best match for each predicted item
    for p_idx, p_item in enumerate(p_list):
        best_match_idx = None
        
        # Find first unused matching item
        for gt_idx, gt_item in enumerate(gt_list):
            if gt_idx not in used_gt_indices and match_func(p_item, gt_item):
                best_match_idx = gt_idx
                break
        
        # If match found, mark as used
        if best_match_idx is not None:
            used_gt_indices.add(best_match_idx)
            used_p_indices.add(p_idx)
            matches += 1
    
    # Calculate precision, recall, f1
    total_predicted = len(p_list)
    total_ground_truth = len(gt_list)
    
    precision = matches / total_predicted if total_predicted > 0 else 0.0
    recall = matches / total_ground_truth if total_ground_truth > 0 else 0.0
    
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'matches': matches,
        'total_predicted': total_predicted,
        'total_ground_truth': total_ground_truth
    }