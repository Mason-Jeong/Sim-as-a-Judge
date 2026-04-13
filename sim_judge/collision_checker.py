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
        return {
            "score": self.score,
            "collision_count": self.collision_count,
            "collision_frame_count": len(self.collision_frames),
            "max_contact_force": self.max_contact_force,
            "failure_reasons": self.failure_reasons,
        }


class CollisionChecker:
    """Accumulates collision events frame-by-frame.

    Usage inside replay loop:
        checker = CollisionChecker(force_threshold=1.0)
        for t in range(num_frames):
            world.step()
            contacts = contact_sensor.get_current_frame()["net_force"]
            checker.check_frame(t, contacts)
        result = checker.result()

    If no ContactSensor is available, use check_frame_from_penetration()
    with overlap distances instead.
    """

    def __init__(self, force_threshold: float = 1.0):
        self.force_threshold = force_threshold
        self.total_frames = 0
        self._collision_frames = []
        self._max_force = 0.0

    def check_frame(self, frame_idx: int, contact_forces: np.ndarray) -> None:
        """Check a single frame using PhysX contact sensor data.

        Args:
            frame_idx: Current frame number
            contact_forces: Net contact forces array from ContactSensor
        """
        self.total_frames += 1
        if contact_forces is None or len(contact_forces) == 0:
            return

        force_mag = np.linalg.norm(contact_forces)
        if force_mag > self.force_threshold:
            self._collision_frames.append(frame_idx)
            self._max_force = max(self._max_force, force_mag)

    def check_frame_no_sensor(self, frame_idx: int, joint_positions: np.ndarray,
                               joint_limits_lower: np.ndarray, joint_limits_upper: np.ndarray) -> None:
        """Fallback: infer potential collisions from joint position saturation.

        If joints are pushed to their limits, the robot may be colliding.
        This is a heuristic when no ContactSensor is available.
        """
        self.total_frames += 1
        at_lower = np.abs(joint_positions - joint_limits_lower) < 0.01
        at_upper = np.abs(joint_positions - joint_limits_upper) < 0.01
        saturated_count = np.sum(at_lower | at_upper)
        if saturated_count > 3:  # multiple joints saturated = likely collision
            self._collision_frames.append(frame_idx)

    def result(self) -> CollisionResult:
        count = len(self._collision_frames)
        score = 1.0 - (count / max(self.total_frames, 1))

        failure_reasons = []
        if count > 0:
            failure_reasons.append(
                f"collision: {count} frames with contact force > {self.force_threshold}N "
                f"(max: {self._max_force:.1f}N)"
            )

        return CollisionResult(
            score=score,
            collision_count=count,
            collision_frames=self._collision_frames,
            max_contact_force=self._max_force,
            failure_reasons=failure_reasons,
        )
