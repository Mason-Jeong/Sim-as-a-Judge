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
        return {
            "score": self.score,
            "subtask_scores": self.subtask_scores,
            "failure_reasons": self.failure_reasons,
        }


class TaskFidelityChecker:
    """Scores pick-and-place subtask completion frame-by-frame.

    Usage inside replay loop:
        checker = TaskFidelityChecker(target_pos=[0.6, -0.3, 0.7])
        for t in range(num_frames):
            world.step()
            ee_pos = get_prim_translate(ee_prim)
            obj_pos = get_prim_translate(object_prim)
            hand_closed = np.mean(hand_joint_pos) > 0.5
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
        self.target_pos = np.array(target_position) if target_position is not None else None
        self.reach_thresh = reach_threshold
        self.grasp_thresh = grasp_threshold
        self.lift_delta = lift_min_delta
        self.place_thresh = place_threshold
        self._weights = subtask_weights or {
            "reach": 0.2, "grasp": 0.3, "lift": 0.3, "place": 0.2,
        }

        self._min_ee_obj_dist = float("inf")
        self._grasp_detected = False
        self._initial_obj_z = None
        self._max_obj_z = -float("inf")
        self._final_obj_pos = None
        self._total_frames = 0

    def check_frame(
        self,
        frame_idx: int,
        ee_position: np.ndarray,
        object_position: np.ndarray,
        hand_closed: bool = False,
    ) -> None:
        """Score one frame."""
        self._total_frames += 1
        ee = np.array(ee_position)
        obj = np.array(object_position)

        # Reach
        dist = np.linalg.norm(ee[:3] - obj[:3])
        self._min_ee_obj_dist = min(self._min_ee_obj_dist, dist)

        # Grasp
        if dist < self.grasp_thresh and hand_closed:
            self._grasp_detected = True

        # Lift
        if self._initial_obj_z is None:
            self._initial_obj_z = obj[2]
        self._max_obj_z = max(self._max_obj_z, obj[2])

        # Place (track final position)
        self._final_obj_pos = obj[:3]

    def result(self) -> TaskFidelityResult:
        subtasks = {}

        # Reach: did EE get close?
        subtasks["reach"] = 1.0 if self._min_ee_obj_dist < self.reach_thresh else \
            max(0, 1.0 - self._min_ee_obj_dist / 0.5)

        # Grasp: was object grasped?
        subtasks["grasp"] = 1.0 if self._grasp_detected else 0.0

        # Lift: did object height increase?
        lift_delta = self._max_obj_z - (self._initial_obj_z or 0)
        subtasks["lift"] = 1.0 if lift_delta > self.lift_delta else \
            max(0, lift_delta / self.lift_delta)

        # Place: did object reach target zone?
        if self.target_pos is not None and self._final_obj_pos is not None:
            place_dist = np.linalg.norm(self._final_obj_pos - self.target_pos)
            subtasks["place"] = 1.0 if place_dist < self.place_thresh else \
                max(0, 1.0 - place_dist / 0.5)
        else:
            subtasks["place"] = 0.5  # cannot determine

        # Weighted average
        score = sum(self._weights[k] * subtasks[k] for k in self._weights)

        failure_reasons = []
        failed = [k for k, v in subtasks.items() if v < 0.5]
        if failed:
            failure_reasons.append(f"task_fidelity: failed subtasks: {', '.join(failed)}")

        return TaskFidelityResult(
            score=score,
            subtask_scores=subtasks,
            failure_reasons=failure_reasons,
        )
