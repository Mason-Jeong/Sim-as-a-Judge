# sim_judge/joint_limit_checker.py
"""Per-frame joint limit validation during Isaac Sim replay."""

from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np


@dataclass
class JointLimitResult:
    score: float = 1.0
    total_violations: int = 0
    violation_frames: list[int] = field(default_factory=list)
    per_joint_violations: dict[str, dict[str, int | float]] = field(default_factory=dict)
    failure_reasons: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        raise NotImplementedError("Implementation not provided in public release.")


class JointLimitChecker:
    """Accumulates joint limit violations frame-by-frame.

    Usage inside replay loop:
        checker = JointLimitChecker(joint_names, lower_limits, upper_limits)
        for t in range(num_frames):
            robot.set_joint_positions(...)
            world.step()
            joint_pos = robot.get_joint_positions()
            checker.check_frame(t, joint_pos)
        result = checker.result()
    """

    def __init__(
        self,
        joint_names: list[str],
        lower_limits: np.ndarray,
        upper_limits: np.ndarray,
        tolerance: float = 0.05,
    ):
        raise NotImplementedError("Implementation not provided in public release.")

    def check_frame(self, frame_idx: int, joint_positions: np.ndarray) -> None:
        raise NotImplementedError("Implementation not provided in public release.")

    def result(self) -> JointLimitResult:
        raise NotImplementedError("Implementation not provided in public release.")
