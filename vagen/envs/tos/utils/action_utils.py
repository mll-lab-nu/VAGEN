"""
Utility functions for converting action results to text observations.
"""
from typing import List
from ..actions import ActionResult


def action_results_to_text(
    action_results: List[ActionResult],
    placeholder: str = None,
    include_action_sequence: bool = False,
) -> str:
    """Convert list of ActionResults to text observation.
    
    Args:
        action_results: List of ActionResult objects from action execution
    
    Returns:
        Text observation string
    """
    assert action_results, "action_results is empty"
    if include_action_sequence:
        cmds = [r.action_command for r in action_results if getattr(r, "action_command", "")]
        seq = f"Actions: [{', '.join(cmds)}]."
        messages = []
        for result in action_results:
            if placeholder and 'observe' in result.action_type:
                messages.append(f"You observe: {placeholder}.")
            else:
                messages.append(result.message)
        return f"{seq} {' '.join(messages)}"

    messages = []
    for result in action_results:
        if placeholder and 'observe' in result.action_type:
            messages.append(f"You observe: {placeholder}.")
        else:
            messages.append(result.message)
    return " ".join(messages)