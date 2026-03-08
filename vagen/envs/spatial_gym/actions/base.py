from abc import ABC, abstractmethod
from typing import Optional, List, Tuple, Dict, Any
from dataclasses import dataclass, field
import re
import numpy as np

"""
Base action definitions and common functionality.
Contains the abstract base class and result types for all actions.
"""


@dataclass
class ActionResult:
    """Result of action execution"""
    success: bool
    message: str
    action_command: str = ""
    action_type: str = ""
    data: Dict[str, Any] = field(default_factory=dict)


class BaseAction(ABC):
    """Base class for all actions"""
    
    # Class attributes to be overridden by subclasses
    format_desc = ""
    description = ""
    example = ""
    format_pattern = ""
    cost = 0
    
    # Shared field of view for all actions
    _field_of_view: int = 90
    _DIRECTION_LABELS_8 = (
        "north",
        "northeast",
        "east",
        "southeast",
        "south",
        "southwest",
        "west",
        "northwest",
    )
    
    def __init__(self, parameters=None):
        self.parameters = parameters
    
    @classmethod
    def set_field_of_view(cls, field_of_view: int):
        """Set the field of view for all actions"""
        assert field_of_view in [90, 180], "Field of view must be 90 or 180 degrees"
        cls._field_of_view = field_of_view
    
    @classmethod
    def get_field_of_view(cls) -> int:
        """Get the current field of view"""
        return cls._field_of_view
    
    @abstractmethod
    def success_message(self, **kwargs) -> str:
        """Return success message for this action"""
        pass
    
    @abstractmethod
    def error_message(self, error_type: str) -> str:
        """Return error message for this action"""
        pass
    
    @abstractmethod
    def execute(self, room, agent, **kwargs) -> ActionResult:
        """Execute action on room state.
        
        Args:
            room: Room to execute action on
            agent: Agent executing the action
            **kwargs: Additional execution context (e.g., coordinate system info)
            
        Returns:
            ActionResult containing success status, message, and additional data
        """
        pass
    
    @staticmethod
    def _is_visible(from_obj, to_obj, field_of_view: int = None) -> bool:
        """
        Check if to_obj is visible from from_obj
        Args:
            from_obj: Object viewing from
            to_obj: Object being viewed
            field_of_view: Field of view in degrees (90 or 180). If None, uses class default.
            
        Returns:
            bool: True if to_obj is visible from from_obj's perspective
            
        Notes:
            - For 90-degree field of view: objects within 45° left and right of orientation
            - For 180-degree field of view: objects within 90° left and right of orientation
        """
        # same position means not visible
        if np.allclose(from_obj.pos, to_obj.pos):
            return False
        if field_of_view is None:
            field_of_view = BaseAction._field_of_view
        
        assert field_of_view in [90, 180], "Invalid field of view"
        # Require same room if room ids are present
        from_room = getattr(from_obj, 'room_id', None)
        to_room = getattr(to_obj, 'room_id', None)
        if from_room is not None and to_room is not None:
            def _as_set(v):
                return set(v) if isinstance(v, list) else ({v} if v is not None else set())
            if not _as_set(from_room).intersection(_as_set(to_room)):
                return False
        direction_vec = to_obj.pos - from_obj.pos
        if np.allclose(direction_vec, 0):
            return True
        direction_norm = direction_vec / np.linalg.norm(direction_vec)
        ori_norm = from_obj.ori / np.linalg.norm(from_obj.ori)
        return np.dot(direction_norm, ori_norm) >= (0.707 - 1e-3) if field_of_view == 90 else np.dot(direction_norm, ori_norm) >= (0.0 - 1e-3)
    
    @staticmethod
    def _get_rotation_matrix(degrees: int) -> np.ndarray:
        """Get rotation matrix. Positive is clockwise; negative is counterclockwise."""
        theta = np.deg2rad(float(degrees))
        c, s = np.cos(theta), np.sin(theta)
        return np.array([[c, -s], [s, c]], dtype=float)

    @staticmethod
    def _ori_to_angle(ori: np.ndarray) -> float:
        """Convert orientation vector to angle (north=0, clockwise positive)."""
        vec = np.asarray(ori, dtype=float)
        norm = float(np.linalg.norm(vec))
        if norm < 1e-9:
            return 0.0
        x, y = vec[0] / norm, vec[1] / norm
        # atan2(x, y): north->0, east->90, south->180, west->-90
        return float(np.rad2deg(np.arctan2(x, y)))

    @staticmethod
    def _angle_to_ori(angle_deg: float) -> np.ndarray:
        """Convert angle to nearest 45-degree orientation unit vector."""
        snapped = round(float(angle_deg) / 45.0) * 45.0
        theta = np.deg2rad(snapped)
        x = float(np.sin(theta))
        y = float(np.cos(theta))
        return np.array([x, y], dtype=float)

    @staticmethod
    def _ori_to_direction_label(ori: np.ndarray) -> str:
        """Map orientation vector to one of 8 direction labels."""
        angle = BaseAction._ori_to_angle(ori)
        idx = int(round(angle / 45.0)) % 8
        return BaseAction._DIRECTION_LABELS_8[idx]
    
    @staticmethod
    def get_anchor_name(room, agent) -> str:
        if np.allclose(agent.pos, agent.init_pos):
            return 'initial_pos'
        at_objs = [o for o in room.all_objects if np.allclose(o.pos, agent.pos)]
        assert len(at_objs) == 1, "Only one object can be at the same position as the agent"
        return at_objs[0].name
    
    @staticmethod
    def is_final() -> bool:
        """Check if this is a final action (ends the sequence)"""
        return False
    
    @staticmethod
    def is_term() -> bool:
        """Check if this is a termination action"""
        return False

    @staticmethod
    def requires_observed_items() -> bool:
        """Whether this action should be gated by previously observed items."""
        return False
    
    @classmethod
    def parse(cls, action_str: str):
        """Parse action string and return instance if matches"""
        if match := re.match(cls.format_pattern, action_str):
            return cls(*match.groups())
        return None

    def get_feedback(self, success: bool, error_type: str = None, **kwargs) -> str:
        """Generate feedback based on execution result"""
        return self.success_message(**kwargs) if success else self.error_message(error_type) 