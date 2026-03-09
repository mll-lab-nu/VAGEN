"""
Cognitive map prompt — global cognitive map only.
"""

GLOBAL_COGMAP_PROMPT = """\
## Global Cognitive Map (JSON)

Represent the scene as a JSON map.

### Schema
- position: [x, y] integers
- facing: direction the object's front face points — "north|south|east|west"

### Rules
- Include only observed objects.
- MUST include facing key for objects that have a facing direction.
- Grid: concise global map on an N×M grid.
- Frame: origin [0,0] is your initial position; your initial facing direction is north.
- Positions: derive each object's [x, y] from its observed grid coordinates during exploration.
- Content: include all observed objects and gates; do NOT include the agent.
- Facing: use "north|south|east|west" (cardinal direction only). Project diagonal headings to the nearest cardinal.

### Example
```json
{
    "chair": {"position": [2, 4], "facing": "north"},
    "sofa": {"position": [5, 1], "facing": "west"}
}
```
"""
