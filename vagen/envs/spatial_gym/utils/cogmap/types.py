from dataclasses import dataclass, field
from typing import Dict, List, Optional
import numpy as np


@dataclass
class BaseCogMetrics:
    overall: float = 0.0
    valid: bool = True

    def to_dict(self) -> Dict[str, float]:
        return {"overall": float(self.overall)}

    @classmethod
    def invalid(cls):
        return cls(overall=0.0, valid=False)

    @staticmethod
    def from_dict(d: Dict[str, float]) -> 'BaseCogMetrics':
        if not isinstance(d, dict) or not d:
            return BaseCogMetrics.invalid()
        return BaseCogMetrics(
            overall=float(d.get('overall', 0.0)),
            valid=True,
        )
    @staticmethod
    def average(items: List['BaseCogMetrics']) -> 'BaseCogMetrics':
        valid_items = [i for i in items if isinstance(i, BaseCogMetrics) and i.valid]
        if not valid_items:
            return BaseCogMetrics.invalid()
        return BaseCogMetrics(
            overall=float(np.mean([i.overall for i in valid_items])),
            valid=True,
        )
    
@dataclass
class MapCogMetrics(BaseCogMetrics):
    dir: float = 0.0
    facing: float = 0.0
    pos: float = 0.0

    def to_dict(self) -> Dict[str, float]:
        return {
            "dir": float(self.dir),
            "facing": float(self.facing),
            "pos": float(self.pos),
            "overall": float(self.overall),
        }

    @staticmethod
    def from_dict(d: Dict[str, float]) -> 'MapCogMetrics':
        if not isinstance(d, dict) or not d:
            return MapCogMetrics.invalid()
        return MapCogMetrics(
            dir=float(d.get('dir', 0.0)),
            facing=float(d.get('facing', 0.0)),
            pos=float(d.get('pos', 0.0)),
            overall=float(d.get('overall', 0.0)),
            valid=True,
        )

    def __add__(self, other: 'MapCogMetrics') -> 'MapCogMetrics':
        return MapCogMetrics(
            dir=self.dir + other.dir,
            facing=self.facing + other.facing,
            pos=self.pos + other.pos,
            overall=self.overall + other.overall,
            valid=self.valid and other.valid,
        )

    def __truediv__(self, scalar: float) -> 'MapCogMetrics':
        if scalar == 0:
            return MapCogMetrics.invalid()
        return MapCogMetrics(
            dir=self.dir / scalar,
            facing=self.facing / scalar,
            pos=self.pos / scalar,
            overall=self.overall / scalar,
            valid=self.valid,
        )

    @staticmethod
    def average(items: List['MapCogMetrics']) -> 'MapCogMetrics':
        valid_items = [i for i in items if isinstance(i, MapCogMetrics) and i.valid]
        if not valid_items:
            return MapCogMetrics.invalid()
        return MapCogMetrics(
            dir=float(np.mean([i.dir for i in valid_items])),
            facing=float(np.mean([i.facing for i in valid_items])),
            pos=float(np.mean([i.pos for i in valid_items])),
            overall=float(np.mean([i.overall for i in valid_items])),
            valid=True,
        )

    @classmethod
    def invalid(cls) -> 'MapCogMetrics':
        return cls(dir=0.0, facing=0.0, pos=0.0, overall=0.0, valid=False)


@dataclass
class AccuracyMetrics(BaseCogMetrics):
    """Binary/accuracy-style metric with validity flag.

    Uses 'overall' as the accuracy score to stay compatible with existing code.
    """

    def to_dict(self) -> Dict[str, float]:
        return {"overall": float(self.overall)}

    @classmethod
    def invalid(cls) -> 'AccuracyMetrics':
        return cls(overall=0.0, valid=False)

    @staticmethod
    def from_dict(d: Dict[str, float]) -> 'AccuracyMetrics':
        if not isinstance(d, dict) or not d:
            return AccuracyMetrics.invalid()
        # Accept either 'overall' or 'acc' keys
        val = d.get('overall', d.get('acc', 0.0))
        return AccuracyMetrics(overall=float(val), valid=True)

@dataclass
class ConsistencySummary:
    local_vs_global: Optional[MapCogMetrics] = None

    def to_dict(self) -> Dict:
        return {
            "local_vs_global": (self.local_vs_global.to_dict() if self.local_vs_global and self.local_vs_global.valid else {}),
        }


@dataclass
class UnexploredMetrics(BaseCogMetrics):
    """Metrics for evaluating unexplored area predictions.
    
    Attributes:
        precision: Fraction of predicted points that are in unexplored areas
        recall: Fraction of unexplored regions that are covered by predictions
        overall: Combined score (harmonic mean of precision and recall)
    """
    precision: float = 0.0
    recall: float = 0.0

    def to_dict(self) -> Dict[str, float]:
        return {
            "precision": float(self.precision),
            "recall": float(self.recall),
            "overall": float(self.overall),
        }

    @classmethod
    def invalid(cls) -> 'UnexploredMetrics':
        return cls(precision=0.0, recall=0.0, overall=0.0, valid=False)

    @staticmethod
    def from_dict(d: Dict[str, float]) -> 'UnexploredMetrics':
        if not isinstance(d, dict) or not d:
            return UnexploredMetrics.invalid()
        return UnexploredMetrics(
            precision=float(d.get('precision', 0.0)),
            recall=float(d.get('recall', 0.0)),
            overall=float(d.get('overall', 0.0)),
            valid=True,
        )

    @staticmethod
    def average(items: List['UnexploredMetrics']) -> 'UnexploredMetrics':
        valid_items = [i for i in items if isinstance(i, UnexploredMetrics) and i.valid]
        if not valid_items:
            return UnexploredMetrics.invalid()
        return UnexploredMetrics(
            precision=float(np.mean([i.precision for i in valid_items])),
            recall=float(np.mean([i.recall for i in valid_items])),
            overall=float(np.mean([i.overall for i in valid_items])),
            valid=True,
        )


__all__ = [
    "BaseCogMetrics",
    "MapCogMetrics",
    "AccuracyMetrics",
    "ConsistencySummary",
    "UnexploredMetrics",
]

