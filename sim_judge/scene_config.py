"""Scene configuration loader and task detection for Sim-as-a-Judge."""

from __future__ import annotations
import json
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd
import yaml


@dataclass(frozen=True)
class TableConfig:
    """Table placement configuration."""

    usd: str
    prim_path: str
    position: list[float] = field(default_factory=lambda: [0.5, -0.2, 0.0])
    rotation: list[float] = field(default_factory=lambda: [0, 0, -90])
    scale: list[float] = field(default_factory=lambda: [1.0, 1.0, 0.7])


@dataclass(frozen=True)
class ObjectConfig:
    """Single scene object configuration."""

    name: str
    obj_type: str  # DynamicCuboid, FixedCuboid, usd
    prim_path: str
    position: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    rotation: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    scale: list[float] = field(default_factory=lambda: [1.0, 1.0, 1.0])
    color: list[float] | None = None
    size: list[float] | None = None  # for Cuboid types
    mass: float = 0.1
    usd_path: str | None = None  # for usd type


@dataclass(frozen=True)
class SceneConfig:
    """Full scene configuration for a task."""

    task: str
    environment_usd: str
    table: TableConfig | None = None
    objects: list[ObjectConfig] = field(default_factory=list)
    target_object: str | None = None
    ee_prim_path: str = "/World/G1/left_wrist_yaw_link"


def load_scene_config(path: str) -> SceneConfig:
    """Load scene configuration from a YAML file.

    Raises:
        FileNotFoundError: If the config file does not exist.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Scene config not found: {path}")

    with open(p) as f:
        raw = yaml.safe_load(f)

    table = None
    if "table" in raw and raw["table"]:
        t = raw["table"]
        table = TableConfig(
            usd=t["usd"],
            prim_path=t["prim_path"],
            position=t.get("position", [0.5, -0.2, 0.0]),
            rotation=t.get("rotation", [0, 0, -90]),
            scale=t.get("scale", [1.0, 1.0, 0.7]),
        )

    objects = []
    for obj in raw.get("objects", []) or []:
        objects.append(ObjectConfig(
            name=obj["name"],
            obj_type=obj["type"],
            prim_path=obj["prim_path"],
            position=obj.get("position", [0.0, 0.0, 0.0]),
            rotation=obj.get("rotation", [0.0, 0.0, 0.0]),
            scale=obj.get("scale", [1.0, 1.0, 1.0]),
            color=obj.get("color"),
            size=obj.get("size"),
            mass=obj.get("mass", 0.1),
            usd_path=obj.get("usd_path"),
        ))

    target = raw.get("target_object")

    return SceneConfig(
        task=raw["task"],
        environment_usd=raw["environment"]["usd"],
        table=table,
        objects=objects,
        target_object=target,
        ee_prim_path=raw.get("ee_prim_path", "/World/G1/left_wrist_yaw_link"),
    )


def load_tasks_jsonl(path: str) -> dict[int, str]:
    """Load task index → description mapping from tasks.jsonl.

    Returns empty dict if file not found.
    """
    p = Path(path)
    if not p.exists():
        return {}

    tasks: dict[int, str] = {}
    with open(p) as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Malformed JSON in {path} line {lineno}: {exc}") from exc
            tasks[entry["task_index"]] = entry["task"]
    return tasks


def detect_task_from_parquet(
    parquet_path: str,
    tasks_jsonl_path: str,
) -> str:
    """Read task_index from parquet, look up description in tasks.jsonl.

    Returns "unknown" if task_index not found in tasks.jsonl.
    """
    df = pd.read_parquet(parquet_path, columns=["task_index"])
    task_index = int(df["task_index"].iloc[0])

    tasks = load_tasks_jsonl(tasks_jsonl_path)
    return tasks.get(task_index, "unknown")


def resolve_scene_config(
    task_description: str,
    scenes_dir: str,
) -> SceneConfig:
    """Find the scene config YAML that matches the task description.

    Scans all YAML files in scenes_dir. Falls back to default.yaml.

    Raises:
        FileNotFoundError: If no matching config and no default.yaml.
    """
    scenes_path = Path(scenes_dir)

    # Search for matching task
    for yaml_file in sorted(scenes_path.glob("*.yaml")):
        if yaml_file.stem == "default":
            continue
        config = load_scene_config(str(yaml_file))
        if config.task == task_description:
            return config

    # Fallback to default
    default = scenes_path / "default.yaml"
    if default.exists():
        return load_scene_config(str(default))

    raise FileNotFoundError(
        f"No scene config matches task '{task_description}' "
        f"and no default.yaml found in {scenes_dir}"
    )
