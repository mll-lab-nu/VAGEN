"""
Cognitive Map Utility Functions

This module provides utility functions for evaluating cognitive maps using turn logs
and LLM interfaces.
"""

from typing import List, Dict, Any
from ..managers.cognitive_map_manager import CognitiveMapManager
from .. import Room, Agent

def _evaluate_cogmaps(
    cognitive_map_manager: CognitiveMapManager,
    responses_by_type: Dict[str, str],
    turn_log: Dict[str, Any],
):
    """Evaluate cognitive maps including unexplored areas.
    
    Args:
        cognitive_map_manager: CognitiveMapManager instance
        responses_by_type: Dict mapping map_type to LLM response text
        turn_log: Turn log dict containing room_state, agent_state, exploration_log etc.
    """
    room_state = Room.from_dict(turn_log['room_state'])
    agent_state = Agent.from_dict(turn_log['agent_state'])

    # Determine observed items and (unexplored) candidate/correct coords
    all_correct_coords = None
    all_candidate_coords = None
    if turn_log.get('is_exploration_phase'):
        exploration_log = turn_log.get('exploration_log', {})
        observed_items = exploration_log.get('observed_items', [obj.name for obj in room_state.all_objects])
        # Extract correct coordinates directly
        all_correct_coords_raw = exploration_log.get('all_correct_coords', [])
        if all_correct_coords_raw:
            all_correct_coords = [(int(pt[0]), int(pt[1])) for pt in all_correct_coords_raw]
        all_candidate_coords_raw = exploration_log.get('all_candidate_coords', [])
        if all_candidate_coords_raw:
            all_candidate_coords = [(int(pt[0]), int(pt[1])) for pt in all_candidate_coords_raw]
    elif (not turn_log.get('is_exploration_phase', True)) and 'falsebelief' in turn_log.get('evaluation_log', {}).get('task_type', '').lower():
        observed_items = [turn_log.get('evaluation_log', {}).get('evaluation_data', {}).get('kwargs', {}).get('rotated_object')]
    else:
        raise ValueError("turn_log must be either exploration phase or false belief evaluation")

    # Evaluate cognitive maps using selected responses
    return cognitive_map_manager.evaluate_cogmaps(
        responses_by_type,
        room_state,
        agent_state,
        observed_items,
        all_correct_coords=all_correct_coords,
        all_candidate_coords=all_candidate_coords,
    )

__all__ = [
    "_evaluate_cogmaps",
]