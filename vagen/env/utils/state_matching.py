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


# --- Example Usage ---
if __name__ == "__main__":
    # Example Predicted and Groundtruth lists (FrozenLake Format)
    predicted_fl_1 = [
    {
        "object_id": "target",
        "vertical_relation": "below",
        "horizontal_relation": "right"
    },
    {
        "object_id": "hole",
        "vertical_relation": "below",
        "horizontal_relation": "right"
    },
    ]

    groundtruth_fl_1 = [
    {
        "object_id": "target",
        "vertical_relation": "below",
        "horizontal_relation": "left" # Different horizontal
    },
    {
        "object_id": "hole",
        "vertical_relation": "below",
        "horizontal_relation": "right" # Exact match
    },
    ]

    # Example with multiple objects of the same ID (Sokoban Format)
    predicted_sokoban_2 = [
        {"object_id": "box", "vertical_relation": "above", "horizontal_relation": "left"},       # Box A
        {"object_id": "box", "vertical_relation": "same", "horizontal_relation": "right"},      # Box B
        {"object_id": "target", "vertical_relation": "above", "horizontal_relation": "same"},    # Target A (Matches GT target)
        {"object_id": "target", "vertical_relation": "below", "horizontal_relation": "left"},     # Target B (Extra)
    ]

    groundtruth_sokoban_2 = [
        {"object_id": "box", "vertical_relation": "above", "horizontal_relation": "left"},       # Box X (Matches Pred Box A, score 1.0)
        {"object_id": "box", "vertical_relation": "above", "horizontal_relation": "right"},      # Box Y (Matches Pred Box A vertically (0.5), matches Pred Box B horizontally (0.5))
        {"object_id": "target", "vertical_relation": "above", "horizontal_relation": "same"},    # Target X (Matches Pred Target A, score 1.0)
    ]
    # Optimal box matching: (Pred Box A <-> GT Box X) score 1.0. Pred Box B has no other GT box to match perfectly.
    # Pred Box B (same, right) vs GT Box Y (above, right). V match=0, H match=1. Score = 0.5.
    # Max matching: (Pred Box A <-> GT Box X) score 1.0 + (Pred Box B <-> GT Box Y) score 0.5. Total Match Score for Box = 1.5.
    # Total Pred Boxes = 2, Total GT Boxes = 2. F1_box = calculate_f1_score(1.5, 2, 2) = 2 * (1.5/2 * 1.5/2) / (1.5/2 + 1.5/2) = 2 * (0.75*0.75) / 1.5 = 2 * 0.5625 / 1.5 = 1.125 / 1.5 = 0.75.

    # Target matching: (Pred Target A <-> GT Target X) score 1.0. Pred Target B has no other GT target.
    # Max matching: (Pred Target A <-> GT Target X) score 1.0. Total Match Score for Target = 1.0.
    # Total Pred Targets = 2, Total GT Targets = 1. F1_target = calculate_f1_score(1.0, 2, 1) = 2 * (1.0/2 * 1.0/1) / (1.0/2 + 1.0/1) = 2 * (0.5 * 1.0) / (0.5 + 1.0) = 2 * 0.5 / 1.5 = 1 / 1.5 = 0.6667.
    # Weighted F1 = 0.4 * 0.6667 + 0.6 * 0.75 = 0.2667 + 0.45 = 0.7167. Reward = 0.7167.


    # Example with missing relations
    predicted_sokoban_3 = [
        {"object_id": "box", "vertical_relation": "above", "horizontal_relation": None},       # Partial prediction
        {"object_id": "target", "vertical_relation": "same", "horizontal_relation": "same"},   # Perfect match
    ]
    groundtruth_sokoban_3 = [
        {"object_id": "box", "vertical_relation": "above", "horizontal_relation": "left"},       # GT is complete
        {"object_id": "target", "vertical_relation": "same", "horizontal_relation": "same"},   # GT is complete
    ]
    # Box: Pred: [('above', None)]. GT: [('above', 'left')]. Similarity = (1 + 0)/2 = 0.5. Total Match Score = 0.5.
    # Total Pred Boxes = 1, Total GT Boxes = 1. F1_box = calculate_f1_score(0.5, 1, 1) = 2 * (0.5/1 * 0.5/1) / (0.5/1 + 0.5/1) = 2 * (0.5*0.5) / 1.0 = 0.5.
    # Target: Pred: [('same', 'same')]. GT: [('same', 'same')]. Similarity = (1+1)/2 = 1.0. Total Match Score = 1.0.
    # Total Pred Targets = 1, Total GT Targets = 1. F1_target = calculate_f1_score(1.0, 1, 1) = 1.0.
    # Weighted F1 = 0.4 * 1.0 + 0.6 * 0.5 = 0.4 + 0.3 = 0.7. Reward = 0.7.

    # Define weights for FrozenLake
    frozenlake_weights = {"target": 0.7, "hole": 0.3}

    # Define weights for Sokoban
    sokoban_weights = {"target": 0.5, "box": 0.5} # Example weights


    print("--- FrozenLake Example 1 (Bipartite Matching) ---")
    reward_fl_1_bipartite = calculate_visual_reasoning_reward_bipartite(predicted_fl_1, groundtruth_fl_1, frozenlake_weights)
    print(f"Reward: {reward_fl_1_bipartite:.4f}") # Expected: 0.5

    print("\n--- Sokoban Example 2 (Bipartite Matching) ---")
    reward_sokoban_2_bipartite = calculate_visual_reasoning_reward_bipartite(predicted_sokoban_2, groundtruth_sokoban_2, sokoban_weights)
    print(f"Reward: {reward_sokoban_2_bipartite:.4f}") # Expected: ~0.7167

    print("\n--- Sokoban Example 3 (Bipartite Matching & Partial) ---")
    reward_sokoban_3_bipartite = calculate_visual_reasoning_reward_bipartite(predicted_sokoban_3, groundtruth_sokoban_3, sokoban_weights)
    print(f"Reward: {reward_sokoban_3_bipartite:.4f}") # Expected: 0.7


    print("\n--- Edge Case: Both lists empty, weights exist ---")
    reward_empty_both = calculate_visual_reasoning_reward_bipartite([], [], frozenlake_weights)
    print(f"Reward: {reward_empty_both:.4f}") # Expected: 1.0

    print("\n--- Edge Case: Pred empty, GT has relevant objects ---")
    reward_pred_empty = calculate_visual_reasoning_reward_bipartite([], [{"object_id":"target", "vertical_relation":"above","horizontal_relation":"left"}], frozenlake_weights)
    print(f"Reward: {reward_pred_empty:.4f}") # Expected: 0.0

    print("\n--- Edge Case: GT empty, Pred has relevant objects ---")
    reward_gt_empty = calculate_visual_reasoning_reward_bipartite([{"object_id":"target", "vertical_relation":"above","horizontal_relation":"left"}], [], frozenlake_weights)
    print(f"Reward: {reward_gt_empty:.4f}") # Expected: 0.0

    print("\n--- Edge Case: Lists have objects, but not in weights ---")
    reward_irrelevant_objects = calculate_visual_reasoning_reward_bipartite([{"object_id":"irrelevant", "vertical_relation":"above","horizontal_relation":"left"}], [{"object_id":"another", "vertical_relation":"above","horizontal_relation":"left"}], frozenlake_weights)
    print(f"Reward: {reward_irrelevant_objects:.4f}") # Expected: 0.0

    # Example: multiple objects, some with None
    predicted_sokoban_4 = [
        {"object_id": "box", "vertical_relation": "above", "horizontal_relation": "left"},       # Box A (Match GT Box X)
        {"object_id": "box", "vertical_relation": "above", "horizontal_relation": None},       # Box B (Partial match GT Box Y vert)
        {"object_id": "target", "vertical_relation": "above", "horizontal_relation": "same"},    # Target A (Match GT Target X)
        {"object_id": "box", "vertical_relation": "below", "horizontal_relation": "right"},      # Box C (No GT match)
    ]

    groundtruth_sokoban_4 = [
        {"object_id": "box", "vertical_relation": "above", "horizontal_relation": "left"},       # Box X
        {"object_id": "box", "vertical_relation": "above", "horizontal_relation": "right"},      # Box Y
        {"object_id": "target", "vertical_relation": "above", "horizontal_relation": "same"},    # Target X
    ]

    print("\n--- Sokoban Example 4 (Bipartite Matching & Partial & Extra) ---")
    reward_sokoban_4_bipartite = calculate_visual_reasoning_reward_bipartite(predicted_sokoban_4, groundtruth_sokoban_4, sokoban_weights)
    print(f"Reward: {reward_sokoban_4_bipartite:.4f}")
    # Box: Pred: [('above', 'left'), ('above', None), ('below', 'right')]. Count = 3.
    # Box: GT: [('above', 'left'), ('above', 'right')]. Count = 2.
    # Weights:
    # ('above', 'left') vs ('above', 'left') -> (1+1)/2 = 1.0
    # ('above', 'left') vs ('above', 'right') -> (1+0)/2 = 0.5
    # ('above', None) vs ('above', 'left') -> (1+0)/2 = 0.5
    # ('above', None) vs ('above', 'right') -> (1+0)/2 = 0.5
    # ('below', 'right') vs ('above', 'left') -> (0+0)/2 = 0.0
    # ('below', 'right') vs ('above', 'right') -> (0+1)/2 = 0.5
    # Weight matrix (Pred rows, GT cols):
    # [[1.0, 0.5],
    #  [0.5, 0.5],
    #  [0.0, 0.5]]
    # Assignment on negative cost matrix:
    # (0,0) [cost -1.0] + (1,1) [cost -0.5] -> total cost -1.5 -> total weight 1.5
    # (0,1) [cost -0.5] + (1,0) [cost -0.5] -> total cost -1.0 -> total weight 1.0
    # (0,0) [cost -1.0] + (2,1) [cost -0.5] -> total cost -1.5 -> total weight 1.5  <-- Optimal (also others might tie)
    # e.g., match (Pred Box A <-> GT Box X) score 1.0, (Pred Box B <-> GT Box Y) score 0.5. Total match score = 1.5.
    # TP_box = 1.5. Total Pred Box = 3, Total GT Box = 2.
    # F1_box = calculate_f1_score(1.5, 3, 2) = 2 * (1.5/3 * 1.5/2) / (1.5/3 + 1.5/2) = 2 * (0.5 * 0.75) / (0.5 + 0.75) = 2 * 0.375 / 1.25 = 0.75 / 1.25 = 0.6.
    # Target: Pred: [('above', 'same')]. Count = 1.
    # Target: GT: [('above', 'same')]. Count = 1.
    # Match: ('above', 'same') vs ('above', 'same'). Similarity = 1.0. Total Match Score = 1.0.
    # TP_target = 1.0. Total Pred Target = 1, Total GT Target = 1. F1_target = calculate_f1_score(1.0, 1, 1) = 1.0.
    # Weighted F1 = 0.4 * 1.0 + 0.6 * 0.6 = 0.4 + 0.36 = 0.76. Reward = 0.76.