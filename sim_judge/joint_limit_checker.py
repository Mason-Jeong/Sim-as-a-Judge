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
        return {
            "score": self.score,
            "total_violations": self.total_violations,
            "violation_frame_count": len(self.violation_frames),
            "per_joint_violations": {
                k: {"count": v["count"], "max_violation": v["max_violation"]}
                for k, v in self.per_joint_violations.items()
            },
            "failure_reasons": self.failure_reasons,
        }


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
        self.joint_names = joint_names
        self.lower = np.array(lower_limits)
        self.upper = np.array(upper_limits)
        self.tolerance = tolerance
        self.n_joints = len(joint_names)
        self.total_frames = 0

        self._violations = []
        self._per_joint = {name: {"count": 0, "max_violation": 0.0} for name in joint_names}

    def check_frame(self, frame_idx: int, joint_positions: np.ndarray) -> None:
        """Check a single frame for joint limit violations."""
        self.total_frames += 1
        pos = np.array(joint_positions[:self.n_joints])

        under = pos < (self.lower - self.tolerance)
        over = pos > (self.upper + self.tolerance)
        violated = under | over

        if violated.any():
            self._violations.append(frame_idx)
            for j in np.where(violated)[0]:
                name = self.joint_names[j]
                violation_mag = max(
                    abs(pos[j] - self.lower[j]) if under[j] else 0,
                    abs(pos[j] - self.upper[j]) if over[j] else 0,
                )
                self._per_joint[name]["count"] += 1
                self._per_joint[name]["max_violation"] = max(
                    self._per_joint[name]["max_violation"], violation_mag
                )

    def result(self) -> JointLimitResult:
        """Aggregate all frames into final result."""
        total_violations = sum(v["count"] for v in self._per_joint.values())
        total_checks = self.total_frames * self.n_joints
        score = 1.0 - (total_violations / max(total_checks, 1))

        failure_reasons = []
        for name, v in self._per_joint.items():
            if v["max_violation"] > self.tolerance * 2:
                failure_reasons.append(
                    f"joint_limit: {name} exceeded by {v['max_violation']:.3f} rad"
                )

        violated_joints = {k: v for k, v in self._per_joint.items() if v["count"] > 0}

        return JointLimitResult(
            score=score,
            total_violations=total_violations,
            violation_frames=self._violations,
            per_joint_violations=violated_joints,
            failure_reasons=failure_reasons,
        )
