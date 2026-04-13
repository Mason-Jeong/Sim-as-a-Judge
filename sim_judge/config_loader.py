"""Load YAML configuration files for Sim-as-a-Judge."""

from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
import yaml


@dataclass(frozen=True)
class EvalConfig:
    """Evaluation configuration from eval_config.yaml."""

    physics_dt: float = 0.001
    collision_force_threshold: float = 1.0
    joint_limits_tolerance: float = 0.05
    gravity_upward_accel_threshold: float = 4.9
    task_fidelity_reach_threshold: float = 0.10
    task_fidelity_grasp_threshold: float = 0.08
    task_fidelity_lift_min_delta: float = 0.05
    task_fidelity_place_threshold: float = 0.10
    metric_weights: dict[str, float] = field(default_factory=lambda: {
        "joint_limit": 0.3,
        "collision": 0.3,
        "gravity": 0.2,
        "task_fidelity": 0.2,
    })
    pass_threshold: float = 0.8


def load_eval_config(path: str) -> EvalConfig:
    """Load evaluation config from a YAML file.

    Args:
        path: Path to eval_config.yaml

    Returns:
        EvalConfig with values from file, defaults for missing keys.

    Raises:
        FileNotFoundError: If the config file does not exist.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(p) as f:
        raw = yaml.safe_load(f) or {}

    physics = raw.get("physics", {})
    collision = raw.get("collision", {})
    joint_limits = raw.get("joint_limits", {})
    gravity = raw.get("gravity", {})
    task_fidelity = raw.get("task_fidelity", {})
    weights = raw.get("metric_weights", None)

    kwargs: dict = {}
    if "dt" in physics:
        kwargs["physics_dt"] = physics["dt"]
    if "force_threshold" in collision:
        kwargs["collision_force_threshold"] = collision["force_threshold"]
    if "tolerance" in joint_limits:
        kwargs["joint_limits_tolerance"] = joint_limits["tolerance"]
    if "upward_accel_threshold" in gravity:
        kwargs["gravity_upward_accel_threshold"] = gravity["upward_accel_threshold"]
    if "reach_threshold" in task_fidelity:
        kwargs["task_fidelity_reach_threshold"] = task_fidelity["reach_threshold"]
    if "grasp_threshold" in task_fidelity:
        kwargs["task_fidelity_grasp_threshold"] = task_fidelity["grasp_threshold"]
    if "lift_min_delta" in task_fidelity:
        kwargs["task_fidelity_lift_min_delta"] = task_fidelity["lift_min_delta"]
    if "place_threshold" in task_fidelity:
        kwargs["task_fidelity_place_threshold"] = task_fidelity["place_threshold"]
    if weights is not None:
        kwargs["metric_weights"] = weights
    if "pass_threshold" in raw:
        kwargs["pass_threshold"] = raw["pass_threshold"]

    return EvalConfig(**kwargs)


@dataclass(frozen=True)
class JointMapConfig:
    """Joint mapping configuration from joint_maps/*.yaml."""

    robot: str
    dof: int
    mapping: dict[int, str]

    def joint_names(self) -> list[str]:
        """Return joint names ordered by parquet index."""
        return [self.mapping[i] for i in sorted(self.mapping.keys())]


def load_joint_map(path: str) -> JointMapConfig:
    """Load joint map from a YAML file.

    Args:
        path: Path to joint map YAML (e.g. joint_maps/g1_inspire.yaml)

    Returns:
        JointMapConfig with robot name, DOF count, and index-to-name mapping.

    Raises:
        FileNotFoundError: If the config file does not exist.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Joint map file not found: {path}")

    with open(p) as f:
        raw = yaml.safe_load(f) or {}

    for required in ("robot", "dof", "mapping"):
        if required not in raw:
            raise ValueError(f"Joint map {path} missing required key: '{required}'")

    mapping = {int(k): str(v) for k, v in raw["mapping"].items()}

    if len(mapping) != raw["dof"]:
        raise ValueError(
            f"Joint map {path}: dof={raw['dof']} but mapping has {len(mapping)} entries"
        )

    return JointMapConfig(
        robot=raw["robot"],
        dof=raw["dof"],
        mapping=mapping,
    )


@dataclass(frozen=True)
class TaskConfig:
    """Task configuration from tasks/*.yaml."""

    task: str
    reach_threshold: float = 0.10
    grasp_threshold: float = 0.08
    lift_min_delta: float = 0.05
    place_threshold: float = 0.10
    place_target_position: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    subtask_weights: dict[str, float] = field(default_factory=lambda: {
        "reach": 0.2,
        "grasp": 0.3,
        "lift": 0.3,
        "place": 0.2,
    })
    success_threshold: float = 0.8


def load_task_config(path: str) -> TaskConfig:
    """Load task config from a YAML file.

    Args:
        path: Path to task config YAML (e.g. tasks/pick_place.yaml)

    Returns:
        TaskConfig with task criteria and thresholds.

    Raises:
        FileNotFoundError: If the config file does not exist.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Task config file not found: {path}")

    with open(p) as f:
        raw = yaml.safe_load(f) or {}

    if "task" not in raw:
        raise ValueError(f"Task config {path} missing required key: 'task'")

    criteria = raw.get("criteria", {})

    kwargs: dict = {
        "task": raw["task"],
        "reach_threshold": criteria.get("reach", {}).get("distance_threshold", 0.10),
        "grasp_threshold": criteria.get("grasp", {}).get("finger_close_threshold", 0.08),
        "lift_min_delta": criteria.get("lift", {}).get("min_height_delta", 0.05),
        "place_threshold": criteria.get("place", {}).get("distance_threshold", 0.10),
        "place_target_position": criteria.get("place", {}).get("target_position", [0.0, 0.0, 0.0]),
    }
    if "subtask_weights" in raw and raw["subtask_weights"]:
        kwargs["subtask_weights"] = raw["subtask_weights"]
    if "success_threshold" in raw:
        kwargs["success_threshold"] = raw["success_threshold"]

    return TaskConfig(**kwargs)
