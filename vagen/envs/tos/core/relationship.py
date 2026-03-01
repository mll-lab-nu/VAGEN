from enum import Enum, auto
from typing import Union, Optional, ClassVar, Tuple, Protocol
import numpy as np
from dataclasses import dataclass
from abc import ABC, abstractmethod


"""Relationship primitives for spatial reasoning."""


# ---- Bin System Protocol ----
class BinSystem(Protocol):
    """Angle binning system without perspective argument.

    Implementations should provide class attributes:
    - BINS: list of (lo_deg, hi_deg, label)
    - LABELS: list of labels (same order as BINS)
    """
    BINS: ClassVar[list]
    LABELS: ClassVar[list]

    @abstractmethod
    def bin(self, degree: float) -> Tuple[int, str]:
        """Return (bin_id, bin_label) for given degree."""
        pass
    
    @classmethod
    @abstractmethod
    def prompt(cls) -> str:
        """Return description of the bin system."""
        pass


class DistanceBinSystem(Protocol):
    """Distance binning system with unified attributes."""
    BINS: ClassVar[list]
    LABELS: ClassVar[list]

    @abstractmethod
    def bin(self, value: float) -> Tuple[int, str]:
        """Return (bin_id, bin_label) for given distance value."""
        pass
    
    @classmethod
    @abstractmethod
    def prompt(cls) -> str:
        """Return description of the distance bin system."""
        pass

class EgoFrontBins:
    """Ego-centric front-focused bins (front is (-1e-3, 1e-3); nodes at ±22.5°, ±45°)."""
    EPS: ClassVar[float] = 1e-3
    # open intervals; preserve ±1e-3 slack at the interior edges
    BINS = [
        (-180.0, -45.0 - 1e-3),
        (-45.0 - 1e-3, -22.5 - 1e-3),
        (-22.5 - 1e-3, -1e-3),
        (-1e-3, 1e-3),
        (1e-3, 22.5 + 1e-3),
        (22.5 + 1e-3, 45.0 + 1e-3),
        (45.0 + 1e-3, 180.0 + 1e-3),
    ]
    LABELS = ['beyond-fov', 'front-left', 'front-slight-left', 'front', 'front-slight-right', 'front-right', 'beyond-fov']

    def bin(self, degree: float):
        v = float(degree)
        for i, (lo, hi) in enumerate(self.BINS):
            if lo < v < hi:
                return i, self.LABELS[i]
        return len(self.LABELS) - 1, self.LABELS[-1]

    @classmethod
    def prompt(cls) -> str:
        lines = [
            "Egocentric angle bins (0° is front): ",
            "[-45°,-22.5°)→front-left, ",
            "[-22.5°,0°)→front-slight-left, ",
            "0°→front, ",
            "(0°,22.5°]→front-slight-right, ",
            "(22.5°,45°]→front-right, ",
            "otherwise→beyond-fov.",
        ]
        return "".join(lines)


class _CardinalBinsBase:
    """Cardinal direction bins (8 directions) without perspective args."""
    LABELS: ClassVar[list] = []
    # Boundaries: [-22.5, +22.5], [22.5, 67.5], ..., [292.5, 337.5]
    BINS: ClassVar[list] = [
        (-22.5, 22.5), (22.5, 67.5), (67.5, 112.5), (112.5, 157.5),
        (157.5, 202.5), (202.5, 247.5), (247.5, 292.5), (292.5, 337.5),
    ]

    def bin(self, degree: float) -> Tuple[int, str]:
        v = float(degree)
        w = (v + 360.0) % 360.0
        idx = int(((w + 22.5) // 45.0) % 8)
        return idx, self.LABELS[idx]

    @classmethod
    def prompt(cls) -> str:
        lines = ["Cardinal angle bins (45° each): "]
        for i, (lo, hi) in enumerate(cls.BINS):
            lines.append(f"({lo}°,{hi}°]→{cls.LABELS[i]}, ")
        return "".join(lines)


class CardinalBinsAllo(_CardinalBinsBase):
    LABELS = ['north', 'north east', 'east', 'south east', 'south', 'south west', 'west', 'north west']


class CardinalBinsEgo(_CardinalBinsBase):
    LABELS = [
        "around 12 o'clock",  # front
        'around 1:30',         # front-right
        "around 3 o'clock",   # right
        'around 4:30',         # back-right
        "around 6 o'clock",   # back
        'around 7:30',         # back-left
        "around 9 o'clock",   # left
        'around 10:30'         # front-left
    ]



class StandardDistanceBins:
    """Standard distance bins (open intervals with eps; includes same-distance bin)."""
    EPS: ClassVar[float] = 1e-6
    # open intervals; include same-distance bin (-EPS, +EPS)
    BINS = [
        (-1e-6, 1e-6),
        (1e-6, 2.0 + 1e-6),
        (2.0 + 1e-6, 4.0 + 1e-6),
        (4.0 + 1e-6, 8.0 + 1e-6),
        (8.0 + 1e-6, 16.0 + 1e-6),
        (16.0 + 1e-6, 32.0 + 1e-6),
        (32.0 + 1e-6, float('inf')),
    ]
    LABELS = ['same distance', 'near', 'mid distance', 'slightly far', 'far', 'very far', 'extremely far']
    
    def bin(self, value: float) -> Tuple[int, str]:
        d = float(value)
        for i, (lo, hi) in enumerate(self.BINS):
            if lo < d < hi:
                return i, self.LABELS[i]
        raise ValueError(f"Invalid distance: {d}")
    
    @classmethod
    def prompt(cls) -> str:
        lines = [
            "Distance bins: ",
            "=0→same distance, ",
            "(0,2]→near, ",
            "(2,4]→mid distance, ",
            "(4,8]→slightly far, ",
            "(8,16]→far, ",
            "(16,32]→very far, ",
            "32→extremely far.",
        ]
        return "".join(lines)


class Dir(Enum):
    """2D directions."""
    SAME = auto()
    FORWARD = auto()   # +y
    BACKWARD = auto()  # -y
    RIGHT = auto()     # +x
    LEFT = auto()      # -x
    UNKNOWN = auto()

    @classmethod
    def from_vertical_delta(cls, d: float) -> 'Dir':
        return cls.SAME if abs(d) < 1e-6 else (cls.FORWARD if d > 0 else cls.BACKWARD)

    @classmethod
    def from_horizontal_delta(cls, d: float) -> 'Dir':
        return cls.SAME if abs(d) < 1e-6 else (cls.RIGHT if d > 0 else cls.LEFT)


@dataclass(frozen=True)
class DirPair:
    """(horiz, vert) pair."""
    horiz: Dir
    vert: Dir

    def __post_init__(self):
        if self.horiz not in (Dir.SAME, Dir.RIGHT, Dir.LEFT, Dir.UNKNOWN):
            raise ValueError(f"Invalid horizontal direction: {self.horiz}")
        if self.vert not in (Dir.SAME, Dir.FORWARD, Dir.BACKWARD, Dir.UNKNOWN):
            raise ValueError(f"Invalid vertical direction: {self.vert}")

    def __getitem__(self, i) -> Dir:
        if i == 0: return self.horiz
        if i == 1: return self.vert
        raise IndexError(i)


@dataclass()
class DegreeRel:
    """Bearing degree only (no Dir/DirPair)."""
    degree: float = 0.0  # +: clockwise, -: counterclockwise, 0: forward/north

    FIELD_OF_VIEW: ClassVar[float] = 90.0

    def __eq__(self, other):
        if type(other) is not DegreeRel:
            return False
        return abs(self.degree - other.degree) < 1e-3

    def __hash__(self):
        return hash(round(self.degree, 6))

    @classmethod
    def prompt(cls) -> str:
        return "Bearing is a degree in [-180, 180]; 0° is front. +: clockwise, -: counterclockwise."

    def to_string(self, perspective: str = 'ego', kind: str = 'relation', gate_dir: 'DegreeRel' = None) -> str:
        return self._format_degree(self.degree)

    @staticmethod
    def _format_degree(v: float) -> str:
        if abs(v) < 1e-3: return "0°"
        sign = '+' if v > 0 else '-'
        return f"{sign}{abs(v):.0f}°"

    @classmethod
    def from_positions(cls, pos1: tuple, pos2: tuple, anchor_ori: tuple = (0, 1)) -> 'DegreeRel':
        p1, p2 = np.array(pos1, float), np.array(pos2, float)
        diff = p1 - p2
        ax, ay = float(anchor_ori[0]), float(anchor_ori[1])
        a_len = float(np.hypot(ax, ay)) or 1.0
        axn, ayn = ax / a_len, ay / a_len
        dx, dy = float(diff[0]), float(diff[1])
        if abs(dx) < 1e-6 and abs(dy) < 1e-6:
            deg = 0.0
        else:
            dot = axn * dx + ayn * dy
            cross = axn * dy - ayn * dx
            deg = -float(np.degrees(np.arctan2(cross, dot)))
        return cls(deg)


class OrientationRel:
    """Orientation only (Dir/DirPair based)."""


    # Transformations (used for orientation only)
    TRANSFORMS: ClassVar[dict[tuple[int, int], dict[Dir, Dir]]] = {
        (0, 1): {d: d for d in Dir},
        (1, 0): {Dir.FORWARD: Dir.LEFT, Dir.BACKWARD: Dir.RIGHT, Dir.RIGHT: Dir.FORWARD, Dir.LEFT: Dir.BACKWARD, Dir.SAME: Dir.SAME, Dir.UNKNOWN: Dir.UNKNOWN},
        (0, -1): {Dir.FORWARD: Dir.BACKWARD, Dir.BACKWARD: Dir.FORWARD, Dir.RIGHT: Dir.LEFT, Dir.LEFT: Dir.RIGHT, Dir.SAME: Dir.SAME, Dir.UNKNOWN: Dir.UNKNOWN},
        (-1, 0): {Dir.FORWARD: Dir.RIGHT, Dir.BACKWARD: Dir.LEFT, Dir.RIGHT: Dir.BACKWARD, Dir.LEFT: Dir.FORWARD, Dir.SAME: Dir.SAME, Dir.UNKNOWN: Dir.UNKNOWN},
    }

    @classmethod
    def prompt(cls) -> str:
        return (
            "Orientation:\n"
            "\t-forward/backward/right/left (ego) or north/east/south/west (allo).\n"
            "\t-When agent faces north: forward = north, right = east, etc.\n"
            "\t-Gate's orientation: report wall position (e.g., 'on left wall')."
        )

    @classmethod
    def transform(cls, pair: DirPair, orientation: tuple) -> DirPair:
        ori = tuple(map(int, orientation))
        if ori not in cls.TRANSFORMS: raise ValueError(f"Invalid orientation: {ori}")
        t = cls.TRANSFORMS[ori]
        h, v = t[pair.horiz], t[pair.vert]
        return DirPair(v, h) if (h in (Dir.FORWARD, Dir.BACKWARD)) or (v in (Dir.RIGHT, Dir.LEFT)) else DirPair(h, v)

    @classmethod
    def get_relative_orientation(cls, obj_ori: tuple, anchor_ori: tuple) -> DirPair:
        # Robust to 8-heading agent orientation: compare angles directly and
        # quantize to nearest egocentric cardinal (front/right/back/left).
        def _ang(v: tuple) -> float:
            x, y = float(v[0]), float(v[1])
            n = float(np.hypot(x, y)) or 1.0
            x, y = x / n, y / n
            return (float(np.degrees(np.arctan2(x, y))) + 360.0) % 360.0

        obj_a = _ang(obj_ori)
        anc_a = _ang(anchor_ori)
        diff = (obj_a - anc_a + 360.0) % 360.0
        nearest = int(round(diff / 90.0) * 90) % 360
        if nearest == 0:
            return DirPair(Dir.SAME, Dir.FORWARD)
        if nearest == 90:
            return DirPair(Dir.RIGHT, Dir.SAME)
        if nearest == 180:
            return DirPair(Dir.SAME, Dir.BACKWARD)
        return DirPair(Dir.LEFT, Dir.SAME)

    @classmethod
    def to_string(cls, o: DirPair, perspective: str = 'ego', kind: str = 'orientation', if_gate: bool = False) -> str:
        if if_gate:
            # Gate orientation relative to agent orientation → wall side
            # same → back wall; reverse → front wall; left → right wall; right → left wall
            # Determine primary axis of orientation 'o'
            if o.horiz != Dir.SAME and o.vert == Dir.SAME:
                primary = o.horiz
            elif o.vert != Dir.SAME and o.horiz == Dir.SAME:
                primary = o.vert
            else:
                raise ValueError(f"Invalid orientation: {o}")
            if primary == Dir.FORWARD:
                return "on back wall"
            if primary == Dir.BACKWARD:
                return "on front wall"
            if primary == Dir.LEFT:
                return "on right wall"
            if primary == Dir.RIGHT:
                return "on left wall"
            raise ValueError(f"Invalid orientation: {o}")
        # object facing
        if o.horiz != Dir.SAME and o.vert == Dir.SAME: primary = o.horiz
        elif o.vert != Dir.SAME and o.horiz == Dir.SAME: primary = o.vert
        else: raise ValueError(f"Invalid orientation: {o}")
        mapping = ( {Dir.FORWARD:'forward', Dir.BACKWARD:'backward', Dir.RIGHT:'right', Dir.LEFT:'left'}
                    if perspective=='ego' else
                    {Dir.FORWARD:'north', Dir.BACKWARD:'south', Dir.RIGHT:'east', Dir.LEFT:'west'} )
        return f"facing {mapping.get(primary, 'unknown')}"


@dataclass()
class DistanceRel:
    value: float
    
    def __eq__(self, other):
        if type(other) is not DistanceRel:
            return False
        return abs(self.value - other.value) < 1e-6
    
    def __hash__(self):
        # Round value to 6 decimal places for consistent hashing
        return hash(round(self.value, 6))

    def to_string(self, perspective: str = 'ego') -> str:
        d = float(self.value)
        return f"{d:.2f}"

    @classmethod
    def prompt(cls) -> str:
        return "Continuous distance: real-valued Euclidean length."

    @classmethod
    def get_distance(cls, pos1: tuple, pos2: tuple) -> 'DistanceRel':
        p1 = np.array(pos1, float)
        p2 = np.array(pos2, float)
        return cls(float(np.linalg.norm(p1 - p2)))


@dataclass()
class DegreeRelBinned(DegreeRel):
    """Degree with modular bin system."""
    bin_system: BinSystem = None
    bin_id: int = 0
    bin_label: str = ""

    def __post_init__(self):
        if self.bin_system is not None:
            self.bin_id, self.bin_label = self.bin_system.bin(self.degree)

    @classmethod
    def from_relation(cls, rel: DegreeRel, bin_system: BinSystem) -> 'DegreeRelBinned':
        bin_id, bin_label = bin_system.bin(rel.degree)
        return cls(degree=rel.degree, bin_system=bin_system, bin_id=bin_id, bin_label=bin_label)

    def to_string(self) -> str:
        return self.bin_label


@dataclass()
class DistanceRelBinned(DistanceRel):
    """Discrete bins for distance using modular bin system."""
    bin_system: DistanceBinSystem = None
    bin_id: int = 0
    bin_label: str = ""
    
    def __post_init__(self):
        if self.bin_system is not None:
            self.bin_id, self.bin_label = self.bin_system.bin(self.value)

    @classmethod
    def from_value(cls, value: float, bin_system: DistanceBinSystem = None) -> 'DistanceRelBinned':
        bin_system = bin_system or StandardDistanceBins()
        bin_id, bin_label = bin_system.bin(value)
        return cls(value=value, bin_system=bin_system, bin_id=bin_id, bin_label=bin_label)

    @classmethod
    def bin_distance(cls, value: float) -> Tuple[int, str]:
        # For backward compatibility
        bin_system = StandardDistanceBins()
        return bin_system.bin(value)

    def to_string(self) -> str:
        return self.bin_label


@dataclass(frozen=False)
class PairwiseRelationship:
    direction: Optional[DegreeRel] = None
    dist: Optional[DistanceRel] = None
    
    def __eq__(self, other):
        if type(other) is not PairwiseRelationship:
            return False
        return self.direction == other.direction and self.dist == other.dist
    
    def __hash__(self):
        return hash((self.direction, self.dist))

    @property
    def bearing(self) -> DegreeRel:
        return self.direction  # type: ignore[return-value]

    # --- external API helpers (hide DirectionRel/DistanceRel) ---
    @property
    def degree(self) -> float:
        return 0.0 if self.direction is None else float(self.direction.degree)

    @property
    def distance_value(self) -> float:
        return 0.0 if self.dist is None else float(self.dist.value)


    @staticmethod
    def format_degree(v: float) -> str:
        return DegreeRel._format_degree(v)

    @staticmethod
    def distance_to_string(value: float) -> str:
        return DistanceRel(float(value)).to_string()

    

    def to_string(self) -> str:
        if self.direction is None and self.dist is None:
            return ""
        if self.direction is None:
            return self.distance_to_string(self.distance_value)
        if self.dist is None:
            return self.direction.to_string('allo', 'relation')
        return f"{self.direction.to_string('allo', 'relation')}, {self.dist.to_string()}"

    @classmethod
    def relationship(cls, pos1: tuple, pos2: tuple, anchor_ori: Optional[tuple] = None, full: bool = True) -> 'PairwiseRelationship':
        d = DegreeRel.from_positions(pos1, pos2, (anchor_ori if anchor_ori is not None else (0, 1)))
        if not full:
            return cls(direction=d, dist=None)
        return cls(direction=d, dist=DistanceRel.get_distance(pos1, pos2))


    @classmethod
    def get_bearing_degree(cls, pos1: tuple, pos2: tuple, anchor_ori: tuple = (0, 1)) -> float:
        return DegreeRel.from_positions(pos1, pos2, anchor_ori).degree

    @classmethod
    def get_distance(cls, pos1: tuple, pos2: tuple) -> DistanceRel:
        return DistanceRel.get_distance(pos1, pos2)

    @classmethod
    def prompt(cls) -> str:
        return (
            "Relationship: bearing in degrees; distance is Euclidean. Use binned labels."
        )


@dataclass()
class PairwiseRelationshipDiscrete(PairwiseRelationship):
    direction: DegreeRelBinned  # type: ignore[assignment]
    dist: DistanceRelBinned  # type: ignore[assignment]
    
    def __eq__(self, other):
        if type(other) is not PairwiseRelationshipDiscrete:
            return False
        return (self.direction.bin_id == other.direction.bin_id and 
                self.dist.bin_id == other.dist.bin_id)
    
    def __hash__(self):
        return hash((self.direction.bin_id, self.dist.bin_id))

    @classmethod
    def prompt(cls, bin_system: BinSystem = None, distance_bin_system: DistanceBinSystem = None) -> str:
        return (
            "Binned relationship reporting:\n"
            f"EgoFront (egocentric, object-to-agent); Cardinal (object-to-object).\n"
            f"{EgoFrontBins.prompt()}\n"
            f"{CardinalBinsAllo.prompt()}\n"
            f"{StandardDistanceBins.prompt()}"
        )
    
    @classmethod
    def relationship(cls, pos1: tuple, pos2: tuple, anchor_ori: Optional[tuple] = None, 
                    bin_system: BinSystem = None, distance_bin_system: DistanceBinSystem = None) -> 'PairwiseRelationshipDiscrete':
        bin_system = bin_system or EgoFrontBins()
        distance_bin_system = distance_bin_system or StandardDistanceBins()
        rel = PairwiseRelationship.relationship(pos1, pos2, anchor_ori=anchor_ori, full=True)
        d = DegreeRelBinned.from_relation(rel.direction, bin_system)
        s = DistanceRelBinned.from_value(rel.dist.value if rel.dist else 0.0, distance_bin_system)
        return cls(direction=d, dist=s)

    def to_string(self) -> str:
        return f"{self.direction.to_string()}, {self.dist.to_string()}"
    
    


@dataclass()
class ProximityRelationship:
    """Object-to-object proximity relationship for nearby objects in agent's FOV."""
    pairwise_rel: PairwiseRelationshipDiscrete
    
    # Proximity threshold - objects must be within this distance to have proximity relationship
    PROXIMITY_THRESHOLD: ClassVar[float] = 2.0 # within near distance
    
    def __eq__(self, other):
        if type(other) is not ProximityRelationship:
            return False
        return self.pairwise_rel == other.pairwise_rel
    
    def __hash__(self):
        return hash(self.pairwise_rel)
    
    @classmethod
    def from_positions(cls, a_pos: tuple, b_pos: tuple, perspective_ori: tuple = (0, 1)) -> Optional['ProximityRelationship']:
        """Create proximity relationship between two close objects (both must be in agent's FOV)."""
        # Check if objects are close enough to each other
        a_to_b_dist = np.linalg.norm(np.array(a_pos) - np.array(b_pos))
        if not (a_to_b_dist <= cls.PROXIMITY_THRESHOLD + 1e-6):
            return None
            
        # Create pairwise relationship between the two objects using agent's perspective
        cardinal_bins = CardinalBinsAllo() # CardinalBinsEgo()
        # cardinal_bins = EgoFrontBins()
        distance_bins = StandardDistanceBins()
        pairwise_rel = PairwiseRelationshipDiscrete.relationship(a_pos, b_pos, perspective_ori, cardinal_bins, distance_bins)
        
        return cls(pairwise_rel=pairwise_rel)
    
    @classmethod
    def prompt(cls, bin_system: BinSystem = None, distance_bin_system: DistanceBinSystem = None) -> str:
        return f"Proximity: relations between close objects (≤{cls.PROXIMITY_THRESHOLD}m). Use cardinal directions and standard distance bins."
    
    def to_string(self, a_name: str, b_name: str) -> str:
        rel_str = self.pairwise_rel.to_string()
        return f"{a_name} is {rel_str} to {b_name}"



@dataclass()
class RelationTriple:
    """subject -> anchor with relation and perspective (anchor's orientation).
    Relationship is evaluated as subject relative to anchor from anchor's perspective.
    """
    subject: str
    anchor: str
    relation: Union[PairwiseRelationship, PairwiseRelationshipDiscrete, ProximityRelationship]
    orientation: Optional[tuple] = None  # anchor object's orientation










if __name__ == "__main__":
    pass