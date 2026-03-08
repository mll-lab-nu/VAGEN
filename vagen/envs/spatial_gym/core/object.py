from typing import List, Dict, Union
import numpy as np
from typing import Any
from dataclasses import dataclass, field


@dataclass
class Object:
    """
    Represents an object or agent in a 2D environment with position and orientation.

    Attributes:
        name (str): The identifier for the object
        pos (np.ndarray): A 2D coordinate representing position
        ori (np.ndarray): A 2D unit vector representing orientation
            - (1, 0)  → 0 degrees
            - (0, 1)  → 90 degrees
            - (-1, 0) → 180 degrees
            - (0, -1) → 270 degrees
        has_orientation (bool): Whether this object has meaningful orientation

    Raises:
        ValueError: If the orientation vector is not one of the valid orientations
                   (only for objects with has_orientation=True)
    """

    name: str
    pos: np.ndarray = field(default_factory=lambda: np.zeros(2))
    ori: np.ndarray = field(default_factory=lambda: np.array([0, 1]))
    has_orientation: bool = True
    label: str = None
    # Room membership: int for single room, List[int] for multi-room objects (e.g., gates)
    room_id: Union[int, List[int], None] = None

    def __post_init__(self):
        assert len(self.pos) == 2, "Position must be a 2D vector"
        assert len(self.ori) == 2, "Orientation must be a 2D vector"
        if self.has_orientation:
            self._validate()


    def _validate(self) -> None:
        VALID_ORIENTATIONS = [
            np.array([1, 0]),
            np.array([0, 1]),
            np.array([-1, 0]),
            np.array([0, -1])
        ]
        if not any(np.array_equal(self.ori, valid_ori) for valid_ori in VALID_ORIENTATIONS):
            raise ValueError(
                f"Orientation must be one of {[o.tolist() for o in VALID_ORIENTATIONS]}, "
                f"got {self.ori.tolist()}"
            )

    def to_dict(self) -> Dict[str, Union[str, List[float], bool, int, List[int], None]]:
        return {
            'name': self.name,
            'pos': self.pos.tolist(),
            'ori': self.ori.tolist(),
            'label': self.label,
            'has_orientation': self.has_orientation,
            'room_id': self.room_id
        }

    @classmethod
    def from_dict(cls, obj_dict: Dict[str, Union[str, List[float], bool, int, List[int], None]]) -> 'Object':
        return cls(
            name=obj_dict['name'],
            pos=np.array(obj_dict['pos']),
            ori=np.array(obj_dict['ori']),
            has_orientation=obj_dict.get('has_orientation', True),
            room_id=obj_dict.get('room_id'),
            label=obj_dict.get('label', None)
        )

    def __repr__(self) -> str:
        return (
            f"\nObject(\n"
            f"    name={self.name},\n"
            f"    pos={self.pos.tolist()},\n"
            f"    ori={self.ori.tolist()},\n"
            f"    has_orientation={self.has_orientation}\n"
            f")"
        )
    
    def copy(self):
        return Object(
            name=self.name,
            pos=self.pos.copy(),
            ori=self.ori.copy(),
            has_orientation=self.has_orientation,
            room_id=(list(self.room_id) if isinstance(self.room_id, list) else self.room_id)
        )

@dataclass
class Agent(Object):
    name: str = 'agent'
    pos: np.ndarray = field(default_factory=lambda: np.array([0, 0]))
    ori: np.ndarray = field(default_factory=lambda: np.array([0, 1]))
    has_orientation: bool = field(default=True)
    init_pos: np.ndarray = None
    init_ori: np.ndarray = None
    # track current and initial room ids
    room_id: Union[int, List[int], None] = None
    init_room_id: Union[int, List[int], None] = None

    def __post_init__(self):
        super().__post_init__()
        # default initial pose to current if not explicitly set
        if self.init_pos is None:
            self.init_pos = self.pos.copy()
        if self.init_ori is None:
            self.init_ori = self.ori.copy()

    def _validate(self) -> None:
        """Agent supports 8 headings (45-degree increments)."""
        valid = []
        for k in range(8):
            theta = np.deg2rad(45.0 * k)
            valid.append(np.array([np.sin(theta), np.cos(theta)]))
        if not any(np.allclose(self.ori, v, atol=1e-6) for v in valid):
            raise ValueError(
                "Agent orientation must be one of 8 headings "
                "(north, northeast, east, southeast, south, southwest, west, northwest), "
                f"got {self.ori.tolist()}"
            )

    def to_dict(self) -> Dict[str, Union[str, List[float], bool]]:
        base = super().to_dict()
        base.update({
            'init_pos': self.init_pos.tolist(),
            'init_ori': self.init_ori.tolist(),
            'init_room_id': self.init_room_id
        })
        return base

    @classmethod
    def from_dict(cls, obj_dict: Dict[str, Union[str, List[float], bool, int, List[int], None]]) -> 'Agent':
        return cls(
            name=obj_dict.get('name', 'agent'),
            pos=np.array(obj_dict.get('pos', [0, 0])),
            ori=np.array(obj_dict.get('ori', [0, 1])),
            has_orientation=obj_dict.get('has_orientation', True),
            init_pos=np.array(obj_dict.get('init_pos', obj_dict.get('pos', [0, 0]))),
            init_ori=np.array(obj_dict.get('init_ori', obj_dict.get('ori', [0, 1]))),
            room_id=obj_dict.get('room_id'),
            init_room_id=obj_dict.get('init_room_id', obj_dict.get('room_id'))
        )

    def copy(self) -> 'Agent':
        return Agent(
            name=self.name,
            pos=self.pos.copy(),
            ori=self.ori.copy(),
            has_orientation=self.has_orientation,
            init_pos=self.init_pos.copy(),
            init_ori=self.init_ori.copy(),
            room_id=(list(self.room_id) if isinstance(self.room_id, list) else self.room_id),
            init_room_id=(list(self.init_room_id) if isinstance(self.init_room_id, list) else self.init_room_id),
        )

@dataclass
class Gate(Object):
    name: str
    pos: np.ndarray = field(default_factory=lambda: np.array([0, 0]))
    ori: np.ndarray = field(default_factory=lambda: np.array([0, 1]))
    has_orientation: bool = field(default=True, init=False)
    # room-id -> orientation (vector points into that room)
    ori_by_room: Dict[int, np.ndarray] = field(default_factory=dict)

    def get_ori_for_room(self, room_id: int) -> np.ndarray:
        return self.ori_by_room.get(int(room_id), self.ori)

    def to_dict(self) -> Dict[str, Union[str, List[float], bool, int, List[int], None]]:
        base = super().to_dict()
        base.update({
            'ori_by_room': {str(k): v.tolist() for k, v in self.ori_by_room.items()},
        })
        return base

    @classmethod
    def from_dict(cls, obj_dict: Dict[str, Union[str, List[float], bool, int, List[int], None]]) -> 'Gate':
        return cls(
            name=obj_dict['name'],
            pos=np.array(obj_dict.get('pos', [0, 0])),
            ori=np.array(obj_dict.get('ori', [0, 1])),
            room_id=obj_dict.get('room_id'),
            ori_by_room={int(k): np.array(v) for k, v in obj_dict.get('ori_by_room', {}).items()},
        )
