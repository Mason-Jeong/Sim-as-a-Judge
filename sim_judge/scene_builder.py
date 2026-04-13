# sim_judge/scene_builder.py
"""Build Isaac Sim scenes from SceneConfig YAML definitions."""

from __future__ import annotations
from .scene_config import SceneConfig


def build_scene(
    world,
    stage,
    scene_config: SceneConfig,
    nucleus_dir: str,
    simulation_app=None,
) -> dict[str, str]:
    """Build an Isaac Sim scene from a SceneConfig.

    Loads environment, table, and objects into the simulation world.
    Supports DynamicCuboid, FixedCuboid, and USD object types.
    Adds full lighting (key, fill, rim, dome).

    Args:
        world: Isaac Sim World instance.
        stage: USD stage.
        scene_config: Loaded SceneConfig with environment, table, objects.
        nucleus_dir: Base URL for Isaac Sim Nucleus assets.
        simulation_app: SimulationApp instance for calling update().

    Returns:
        Dict mapping object names to their prim paths.
    """
    raise NotImplementedError("Implementation not provided in public release.")
