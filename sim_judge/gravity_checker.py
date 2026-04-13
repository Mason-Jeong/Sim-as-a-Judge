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
        raise NotImplementedError("Implementation not provided in public release.")


class GravityChecker:
    """Detects gravity violations by tracking object vertical motion.

    Usage inside replay loop:
        checker = GravityChecker(dt=1/30)
        for t in range(num_frames):
            world.step()
            obj_pos = get_prim_translate(object_prim)
            checker.check_frame(t, obj_pos)
        result = checker.result()
    """

    def __init__(self, dt: float = 1 / 30, upward_accel_threshold: float = 4.9):
        raise NotImplementedError("Implementation not provided in public release.")

    def check_frame(self, frame_idx: int, object_position: np.ndarray) -> None:
        raise NotImplementedError("Implementation not provided in public release.")

    def result(self) -> GravityResult:
        raise NotImplementedError("Implementation not provided in public release.")
