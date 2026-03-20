# sim_judge/gravity_checker.py
"""Per-frame gravity consistency validation during Isaac Sim replay."""

from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np


@dataclass
class GravityResult:
    score: float = 1.0
    violation_frames: list[int] = field(default_factory=list)
    max_upward_accel: float = 0.0
    failure_reasons: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "score": self.score,
            "violation_frame_count": len(self.violation_frames),
            "max_upward_accel": self.max_upward_accel,
            "failure_reasons": self.failure_reasons,
        }


class GravityChecker:
    """Detects gravity violations by tracking object vertical motion.

    Usage inside replay loop:
        checker = GravityChecker(dt=1/30)
        for t in range(num_frames):
            world.step()
            obj_pos = get_prim_translate(object_prim)  # [x, y, z]
            checker.check_frame(t, obj_pos)
        result = checker.result()
    """

    def __init__(self, dt: float = 1 / 30, upward_accel_threshold: float = 4.9):
        self.dt = dt
        self.threshold = upward_accel_threshold  # 0.5g default
        self.total_frames = 0

        self._positions = []
        self._violation_frames = []
        self._max_upward = 0.0

    def check_frame(self, frame_idx: int, object_position: np.ndarray):
        """Record object position. Violations computed after 3+ frames."""
        self.total_frames += 1
        pos = np.array(object_position, dtype=np.float64)
        self._positions.append((frame_idx, pos))

        if len(self._positions) >= 3:
            z0 = self._positions[-3][1][2]
            z1 = self._positions[-2][1][2]
            z2 = self._positions[-1][1][2]

            vel1 = (z1 - z0) / self.dt
            vel2 = (z2 - z1) / self.dt
            accel = (vel2 - vel1) / self.dt

            # Positive accel = upward. Flag if object accelerates up without support
            if accel > self.threshold:
                self._violation_frames.append(frame_idx)
                self._max_upward = max(self._max_upward, accel)

    def result(self) -> GravityResult:
        count = len(self._violation_frames)
        score = 1.0 - (count / max(self.total_frames - 2, 1))

        failure_reasons = []
        if count > 0:
            failure_reasons.append(
                f"gravity_violation: {count} frames with upward accel > {self.threshold:.1f} m/s² "
                f"(max: {self._max_upward:.1f} m/s²)"
            )

        return GravityResult(
            score=score,
            violation_frames=self._violation_frames,
            max_upward_accel=self._max_upward,
            failure_reasons=failure_reasons,
        )
