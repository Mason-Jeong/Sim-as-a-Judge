# sim_judge/aggregate.py
"""Combine checker results into a single SimJudgeVerdict."""

from __future__ import annotations
from dataclasses import dataclass, field

from .joint_limit_checker import JointLimitResult
from .collision_checker import CollisionResult
from .gravity_checker import GravityResult
from .task_fidelity_checker import TaskFidelityResult


@dataclass
class SimJudgeVerdict:
    passed: bool
    overall_score: float
    joint_limit: JointLimitResult
    collision: CollisionResult
    gravity: GravityResult
    task_fidelity: TaskFidelityResult
    failure_reasons: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "passed": self.passed,
            "overall_score": round(self.overall_score, 4),
            "joint_limit": self.joint_limit.to_dict(),
            "collision": self.collision.to_dict(),
            "gravity": self.gravity.to_dict(),
            "task_fidelity": self.task_fidelity.to_dict(),
            "failure_reasons": self.failure_reasons,
        }

    def summary(self) -> str:
        status = "PASS ✓" if self.passed else "FAIL ✗"
        lines = [
            f"{'Metric':<18} {'Score':>6}  Details",
            "-" * 55,
            f"{'Joint Limits':<18} {self.joint_limit.score:>5.2f}  {len(self.joint_limit.violation_frames)} violation frames",
            f"{'Collision':<18} {self.collision.score:>5.2f}  {self.collision.collision_count} collision frames",
            f"{'Gravity':<18} {self.gravity.score:>5.2f}  {len(self.gravity.violation_frames)} violation frames",
            f"{'Task Fidelity':<18} {self.task_fidelity.score:>5.2f}  {self.task_fidelity.subtask_scores}",
            "-" * 55,
            f"{'Overall':<18} {self.overall_score:>5.2f}  {status}",
        ]
        if self.failure_reasons:
            lines.append(f"\nFailures:")
            for r in self.failure_reasons:
                lines.append(f"  - {r}")
        return "\n".join(lines)


def aggregate(
    joint_limit: JointLimitResult,
    collision: CollisionResult,
    gravity: GravityResult,
    task_fidelity: TaskFidelityResult,
    weights: dict[str, float] | None = None,
    pass_threshold: float = 0.8,
) -> SimJudgeVerdict:
    """Combine all checker results into a final verdict."""
    w = weights or {
        "joint_limit": 0.3,
        "collision": 0.3,
        "gravity": 0.2,
        "task_fidelity": 0.2,
    }
    overall = (
        w["joint_limit"] * joint_limit.score
        + w["collision"] * collision.score
        + w["gravity"] * gravity.score
        + w["task_fidelity"] * task_fidelity.score
    )

    failure_reasons = (
        joint_limit.failure_reasons
        + collision.failure_reasons
        + gravity.failure_reasons
        + task_fidelity.failure_reasons
    )

    passed = overall >= pass_threshold and len(failure_reasons) == 0

    return SimJudgeVerdict(
        passed=passed,
        overall_score=overall,
        joint_limit=joint_limit,
        collision=collision,
        gravity=gravity,
        task_fidelity=task_fidelity,
        failure_reasons=failure_reasons,
    )
