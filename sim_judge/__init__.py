# sim_judge/__init__.py
"""Sim-as-a-Judge: Physics-grounded validation for synthetic robot trajectories."""

from .joint_limit_checker import JointLimitChecker, JointLimitResult
from .collision_checker import CollisionChecker, CollisionResult
from .gravity_checker import GravityChecker, GravityResult
from .task_fidelity_checker import TaskFidelityChecker, TaskFidelityResult
from .aggregate import aggregate, SimJudgeVerdict
from .report import save_json, print_verdict
from .config_loader import (
    load_eval_config,
    load_joint_map,
    load_task_config,
    EvalConfig,
    JointMapConfig,
    TaskConfig,
)

__version__ = "0.1.0"
