SHARED_INTRO_VISION = (
    "You are a spatial reasoner in a 3D simulated environment. "
    "The world is rendered in 3D but abstracted into a discrete 2D grid of size N×M. "
    "Every entity, including yourself, is represented by integer coordinates (x, y) on this grid."
)

SHARED_MULTIROOM_RULES = """\
Multi-room rules (may exist multiple rooms):
- Your vision is confined to your current room.
- Doors block vision between rooms.
- Exception: When located in a doorway, door is open and invisible, you can see into both connected rooms.
- Rooms connect via doors on vertical (front/back) or horizontal (left/right) walls.
"""

SHARED_RULES_COMMON = """\
- FOV is 90°. Track your position and facing after every Rotate() or JumpTo().
- Agent facing uses 8 headings: north/northeast/east/southeast/south/southwest/west/northwest.
- World axes are unchanged: +y=north, +x=east.
- Object facing (for non-agent objects) is still only one of north/east/south/west.
"""

ACTIVE_RULES_EXTRA = """\
- Explore: jump to doors early (doorway sees both rooms), then cover each room systematically.
- Coordinates: start=(0,0) north; +y=north, +x=east.
"""

VISION_OBSERVATION_INSTRUCTIONS = (
    "Use the rendered image as the primary observation signal. "
    "Do not assume hidden objects; only use what has been observed."
)

VISION_EXAMPLE = """\
Here is an example of your observation: blue cylinder 1 m straight ahead; red cylinder 2 m straight ahead; yellow cylinder 2 m at 45° to your front-left; green cylinder 3 m at 22.5° to your front-slight-right:
{image_placeholder}

The image shows all objects in the room. Each tile is numbered (1-N) in the top-left, matching the object order in the room layout.
For items with a facing direction, two copies are shown side-by-side: the left copy has its front facing the camera; the right copy has its front facing left.
Items without a meaningful facing direction are shown once.
{image_placeholder}
"""
