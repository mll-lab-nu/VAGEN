"""Sokoban's side of the state reward: what the agent is asked to describe, and truth.

Positions are reported relative to the player, because that is what the agent can see
and reason about without a coordinate frame.
"""

from __future__ import annotations

import numpy as np

from vagen.envs._common.rewards import StateRewardSpec

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


# Plain sentences, deliberately. The judge exists to turn a description into structured
# relations; showing the agent the structure instead makes it emit JSON, the judge a
# re-parser of its own output format, and the score a measure of format compliance
# rather than of whether the agent can see where things are. It also flatters the F1,
# because none of the ambiguity the judge is there to absorb ever arises.
EXAMPLES = {
    "state_estimation": (
        "Before acting, say in plain words what you see:\n"
        "<observation>A box is directly below me in my column, and the target is also "
        "below me, in that same column.</observation>"
    ),
    "transition_prediction": (
        "After choosing, say in plain words what will follow:\n"
        "<prediction>The box will still be below me in my column, and I will have moved "
        "onto the target's row, so the target will be level with me.</prediction>"
    ),
}

AXES = """Relations are relative to you: vertical is above/below/same, horizontal is left/right/same.
Each description is scored against the real layout, so inventing objects costs more than
omitting them, and describing something that is not there scores nothing for it."""


SPEC = StateRewardSpec(
    relations=relations,
    judge_prompt=JUDGE_PROMPT,
    # A missed target matters as much as the boxes together: the target is what the task
    # is about, while boxes are numerous enough to score well by accident.
    object_weights={"target": 0.5, "box": 0.5},
    examples=EXAMPLES,
    axes=AXES,
)
