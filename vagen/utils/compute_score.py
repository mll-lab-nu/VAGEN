"""
The script implement the score function used to compute reward
"""

def compute_score(
        data_source: str,
        solution_str: str,
        ground_truth: dict,
        extra_info: dict
):
    """
    Currently implement directly using given reward
    """
    return ground_truth['reward']