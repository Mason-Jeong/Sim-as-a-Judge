# sim_judge/collision_checker.py
"""Per-frame collision detection during Isaac Sim replay."""

from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np


@dataclass
class CollisionResult:
    score: float = 1.0
    collision_count: int = 0
    collision_frames: list[int] = field(default_factory=list)
    max_contact_force: float = 0.0
    failure_reasons: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        raise NotImplementedError("Implementation not provided in public release.")


class CollisionChecker:
    """Accumulates collision events frame-by-frame.

    Supports two modes:
    - check_frame(): Uses PhysX ContactSensor data
    - check_frame_no_sensor(): Heuristic from joint saturation
    """

    def __init__(self, force_threshold: float = 1.0):
        raise NotImplementedError("Implementation not provided in public release.")

    def check_frame(self, frame_idx: int, contact_forces: np.ndarray) -> None:
        raise NotImplementedError("Implementation not provided in public release.")

    def check_frame_no_sensor(self, frame_idx: int, joint_positions: np.ndarray,
                               joint_limits_lower: np.ndarray, joint_limits_upper: np.ndarray) -> None:
        raise NotImplementedError("Implementation not provided in public release.")

    def result(self) -> CollisionResult:
        raise NotImplementedError("Implementation not provided in public release.")
