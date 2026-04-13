# sim_judge/task_fidelity_checker.py
"""Per-frame task fidelity scoring during Isaac Sim replay."""

from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np


@dataclass
class TaskFidelityResult:
    score: float = 1.0
    subtask_scores: dict[str, float] = field(default_factory=dict)
    failure_reasons: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        raise NotImplementedError("Implementation not provided in public release.")


class TaskFidelityChecker:
    """Scores pick-and-place subtask completion frame-by-frame.

    Evaluates: reach, grasp, lift, place — each scored 0 to 1.

    Usage inside replay loop:
        checker = TaskFidelityChecker(target_pos=[0.6, -0.3, 0.7])
        for t in range(num_frames):
            world.step()
            checker.check_frame(t, ee_pos, obj_pos, hand_closed)
        result = checker.result()
    """

    def __init__(
        self,
        target_position: list[float] | None = None,
        reach_threshold: float = 0.10,
        grasp_threshold: float = 0.08,
        lift_min_delta: float = 0.05,
        place_threshold: float = 0.10,
        subtask_weights: dict[str, float] | None = None,
    ):
        raise NotImplementedError("Implementation not provided in public release.")

    def check_frame(
        self,
        frame_idx: int,
        ee_position: np.ndarray,
        object_position: np.ndarray,
        hand_closed: bool = False,
    ) -> None:
        raise NotImplementedError("Implementation not provided in public release.")

    def result(self) -> TaskFidelityResult:
        raise NotImplementedError("Implementation not provided in public release.")
