from dataclasses import dataclass
import numpy as np
from .object import Object


@dataclass
class ObjectInfo:
    name: str
    has_orientation: bool

AGENT_NAME = 'you'

# Objects with orientation (have a clear front/back, directional)
OBJECT_NAMES_WITH_ORIENTATION = [
    'chair',
    'sofa',
    'bookshelf',
    'cabinet',
    'television',
    'refrigerator',
    'microwave',
    'oven',
    'computer',
    'printer',
    'scanner',
    'monitor',
    'projector',
    'whiteboard',
    'air conditioner',
]
OBJECTS_WITH_ORIENTATION = [ObjectInfo(name, True) for name in OBJECT_NAMES_WITH_ORIENTATION]


# Objects without orientation (omnidirectional, no clear front/back)
OBJECT_NAMES_WITHOUT_ORIENTATION = [
    'table',
    'lamp',
    'plant',
    'vase',
    'trash can',
]
OBJECTS_WITHOUT_ORIENTATION = [ObjectInfo(name, False) for name in OBJECT_NAMES_WITHOUT_ORIENTATION]

# Combined lists for backward compatibility
CANDIDATE_OBJECTS = OBJECTS_WITH_ORIENTATION + OBJECTS_WITHOUT_ORIENTATION

_ADDITIONAL_OBJECT_NAMES = [
    'curtains',
    'mirror',
    'rug',
    'pillow',
    'blanket',
    'dresser',
    'nightstand',
    'fan',
    'air conditioner',
    'heater',
    'trash can',
    'picture frame',
    'wall art',
    'throw pillow',
    'coffee table',
    'ottoman',
    'ceiling light',
    'phone charger',
    'coaster',
    'decorative bowl'
]

ADDITIONAL_CANDIDATE_OBJECTS = [
    ObjectInfo(name, has_orientation=name in OBJECT_NAMES_WITH_ORIENTATION) 
    for name in sorted(list(set(_ADDITIONAL_OBJECT_NAMES)))
]



# predefined room
easy_room_config = {
    "name": "easy_room",
    "objects": [
        Object(
            name="table",
            pos=np.array([3, 0]),
            ori=np.array([1, 0])),
        Object(
            name="chair",
            pos=np.array([3, 3]),
            ori=np.array([-1, 0])),
    ],
    "Agent": Object(
        name="agent",
        pos=np.array([0, 0]),
        ori=np.array([0, 1])
    ),
}

easy_room_config_2 = {
    "name": "easy_room_2",
    "objects": [
        Object(
            name="table", 
            pos=np.array([4, -3]),
            ori=np.array([1, 0])),
        Object(
            name="chair", 
            pos=np.array([3, 3]),
            ori=np.array([0, -1])),
        Object(
            name="sofa", 
            pos=np.array([2, 0]),
            ori=np.array([-1, 0])),
    ],
    "Agent": Object(
        name="agent", 
        pos=np.array([0, 0]), 
        ori=np.array([0, 1])),
}


# easy_room_config_3 = {
#     "name": "easy_room_3",
#     "objects": [
#         Object(
#             name="table", 
#             pos=np.array([4, -3]),
#             ori=np.array([1, 0])),
#         Object(
#             name="chair", 
#             pos=np.array([3, 3]),
#             ori=np.array([0, -1])),
#         Object(
#             name="sofa", 
#             pos=np.array([2, 0]),
#             ori=np.array([-1, 0])),
#     ],
# }

easy_room_config_3 = {
    "name": "easy_room_3",
    "objects": [
        Object(
            name="A", 
            pos=np.array([0, 0]),
            ori=np.array([1, 0])),
        Object(
            name="B", 
            pos=np.array([1, 0]),
            ori=np.array([0, -1])),
        Object(
            name="C", 
            pos=np.array([0, -1]),
            ori=np.array([-1, 0])),
        Object(
            name="D", 
            pos=np.array([-1, 0]),
            ori=np.array([0, 1])),
        Object(
            name="E", 
            pos=np.array([0, 1]),
            ori=np.array([0, 1])),
    ],
}