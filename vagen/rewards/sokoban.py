"""Sokoban's side of the state reward: what the agent is asked to describe, and truth.

Positions are reported relative to the player, because that is what the agent can see
and reason about without a coordinate frame.
"""

from __future__ import annotations

import numpy as np

from vagen.rewards.state_reward import StateRewardSpec

VERTICAL = {-1: "above", 0: "same", 1: "below"}
HORIZONTAL = {-1: "left", 0: "same", 1: "right"}


def relations(env) -> list[dict]:
    """Where the boxes and targets are, relative to the player, right now."""
    room = env.env.room_state
    fixed = env.env.room_fixed

    players = np.argwhere((room == 5) | (room == 6))
    if not len(players):
        return []
    row, col = players[0]

    items = []
    for object_id, positions in (
        ("box", np.argwhere((room == 3) | (room == 4))),
        ("target", np.argwhere(fixed == 2)),
    ):
        for r, c in positions:
            items.append(
                {
                    "object_id": object_id,
                    "vertical_relation": VERTICAL[int(np.sign(r - row))],
                    "horizontal_relation": HORIZONTAL[int(np.sign(c - col))],
                }
            )
    return items


JUDGE_PROMPT = """Extract the spatial relations described below into JSON.

Output a JSON array. One object per thing mentioned, with exactly these keys:
  "object_id"           "box" or "target". Any box -- "box0", "a box", "the box" -- is "box".
  "vertical_relation"   "above", "below", "same", or null if not stated.
  "horizontal_relation" "left", "right", "same", or null if not stated.

Relations are relative to the player. Describe only what the text states; do not infer.
If nothing mappable is described, output [].

Text:
{content}

JSON:"""


INSTRUCTIONS = """Before acting, state what you see; after choosing, state what will follow.

<observation>[{"object_id":"box","vertical_relation":"below","horizontal_relation":"same"},\
{"object_id":"target","vertical_relation":"below","horizontal_relation":"same"}]</observation>
<think>The box is directly below me and the target below it, so pushing down twice solves it.</think>
<prediction>[{"object_id":"box","vertical_relation":"below","horizontal_relation":"same"},\
{"object_id":"target","vertical_relation":"same","horizontal_relation":"same"}]</prediction>
<answer>Down</answer>

Relations are relative to you: vertical is above/below/same, horizontal is left/right/same.
Both descriptions are scored against the real layout, so guessing costs more than it gains."""


SPEC = StateRewardSpec(
    relations=relations,
    judge_prompt=JUDGE_PROMPT,
    # A target missed matters more than one box among several: the target is what the
    # task is about, and boxes are numerous enough to score well by accident.
    object_weights={"target": 0.5, "box": 0.5},
    instructions=INSTRUCTIONS,
)
