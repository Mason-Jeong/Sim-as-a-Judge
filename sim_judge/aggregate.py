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
        raise NotImplementedError("Implementation not provided in public release.")

    def summary(self) -> str:
        raise NotImplementedError("Implementation not provided in public release.")


def aggregate(
    joint_limit: JointLimitResult,
    collision: CollisionResult,
    gravity: GravityResult,
    task_fidelity: TaskFidelityResult,
    weights: dict[str, float] | None = None,
    pass_threshold: float = 0.8,
) -> SimJudgeVerdict:
    """Combine all checker results into a final verdict.

    Args:
        joint_limit: Result from JointLimitChecker
        collision: Result from CollisionChecker
        gravity: Result from GravityChecker
        task_fidelity: Result from TaskFidelityChecker
        weights: Per-metric weights (default: equal)
        pass_threshold: Minimum overall score to pass

    Returns:
        SimJudgeVerdict with pass/fail decision and per-metric scores.
    """
    raise NotImplementedError("Implementation not provided in public release.")
