"""Build Isaac Sim scenes from SceneConfig YAML definitions."""

from __future__ import annotations
import numpy as np

from .scene_config import SceneConfig


def build_scene(
    world,
    stage,
    scene_config: SceneConfig,
    nucleus_dir: str,
    simulation_app=None,
) -> dict[str, str]:
    """Build an Isaac Sim scene from a SceneConfig.

    Args:
        world: Isaac Sim World instance.
        stage: USD stage from omni.usd.get_context().get_stage().
        scene_config: Loaded SceneConfig with environment, table, objects.
        nucleus_dir: Base URL for Isaac Sim Nucleus assets.
        simulation_app: SimulationApp instance for calling update().

    Returns:
        Dict mapping object names to their prim paths.
    """
    from omni.isaac.core.utils.stage import add_reference_to_stage
    from pxr import UsdGeom, UsdPhysics, UsdLux, Gf

    def _update(n: int = 3) -> None:
        if simulation_app is not None:
            for _ in range(n):
                simulation_app.update()

    prim_map: dict[str, str] = {}

    # --- Environment ---
    env_loaded = False
    env_candidates = [
        f"{nucleus_dir}/{scene_config.environment_usd}",
        f"{nucleus_dir}/Environments/Simple_Room/simple_room.usd",
        f"{nucleus_dir}/Environments/Grid/default_environment.usd",
        f"{nucleus_dir}/Environments/Simple_Warehouse/warehouse.usd",
        f"{nucleus_dir}/Environments/Simple_Warehouse/warehouse_with_forklifts.usd",
    ]
    for env_usd in env_candidates:
        try:
            add_reference_to_stage(usd_path=env_usd, prim_path="/World/Environment")
            print(f"[Scene] Environment loaded: {env_usd.split('/')[-1]}")
            env_loaded = True
            break
        except Exception:
            continue

    if not env_loaded:
        print("[Scene] No environment USD found, using default ground plane")
        world.scene.add_default_ground_plane()

    # --- Full Lighting Setup ---
    # Key light — main directional light from above-front
    if not stage.GetPrimAtPath("/World/KeyLight").IsValid():
        key = UsdLux.DistantLight.Define(stage, "/World/KeyLight")
        key.CreateIntensityAttr(3500)
        key.CreateAngleAttr(0.53)
        xform = UsdGeom.Xformable(key.GetPrim())
        xform.AddRotateXYZOp().Set(Gf.Vec3f(-50.0, 15.0, 0.0))

    # Fill light — softer, opposite side to reduce harsh shadows
    if not stage.GetPrimAtPath("/World/FillLight").IsValid():
        fill = UsdLux.DistantLight.Define(stage, "/World/FillLight")
        fill.CreateIntensityAttr(2000)
        fill.CreateAngleAttr(1.0)
        xform = UsdGeom.Xformable(fill.GetPrim())
        xform.AddRotateXYZOp().Set(Gf.Vec3f(-35.0, -120.0, 0.0))

    # Rim light — from behind for edge separation
    if not stage.GetPrimAtPath("/World/RimLight").IsValid():
        rim = UsdLux.DistantLight.Define(stage, "/World/RimLight")
        rim.CreateIntensityAttr(1500)
        rim.CreateAngleAttr(0.53)
        xform = UsdGeom.Xformable(rim.GetPrim())
        xform.AddRotateXYZOp().Set(Gf.Vec3f(-20.0, 180.0, 0.0))

    # Dome light — ambient fill to eliminate pure black areas
    if not stage.GetPrimAtPath("/World/DomeLight").IsValid():
        dome = UsdLux.DomeLight.Define(stage, "/World/DomeLight")
        dome.CreateIntensityAttr(500)

    print("[Scene] Full lighting setup (key + fill + rim + dome)")
    _update()

    # --- Table ---
    if scene_config.table is not None:
        t = scene_config.table
        table_usd = f"{nucleus_dir}/{t.usd}"
        try:
            add_reference_to_stage(usd_path=table_usd, prim_path=t.prim_path)
        except Exception as e:
            print(f"[Scene] Table USD failed to load: {e}")
        _update()

        table_prim = stage.GetPrimAtPath(t.prim_path)
        if table_prim.IsValid():
            xform = UsdGeom.Xformable(table_prim)
            xform.ClearXformOpOrder()
            xform.AddTranslateOp().Set(Gf.Vec3d(*t.position))
            xform.AddRotateXYZOp().Set(Gf.Vec3f(*t.rotation))
            xform.AddScaleOp().Set(Gf.Vec3d(*t.scale))
            print(f"[Scene] Table placed at {t.position}")
        prim_map["table"] = t.prim_path

    # --- Objects ---
    for obj in scene_config.objects:
        if obj.obj_type == "DynamicCuboid":
            _build_dynamic_cuboid(world, stage, obj, _update)
        elif obj.obj_type == "FixedCuboid":
            _build_fixed_cuboid(world, stage, obj, _update)
        elif obj.obj_type == "usd":
            _build_usd_object(stage, obj, nucleus_dir, _update)
        else:
            print(f"[Scene] Unknown object type: {obj.obj_type} for {obj.name}")
            continue

        prim_map[obj.name] = obj.prim_path
        print(f"[Scene] Object '{obj.name}' placed at {obj.position}")

    return prim_map


def _build_dynamic_cuboid(world, stage, obj, _update) -> None:
    """Create a DynamicCuboid with physics."""
    from omni.isaac.core.objects import DynamicCuboid
    from pxr import Gf

    color = np.array(obj.color) if obj.color else np.array([0.5, 0.5, 0.5])
    size = obj.size[0] if obj.size else 0.04

    world.scene.add(DynamicCuboid(
        prim_path=obj.prim_path,
        name=obj.name,
        position=np.array(obj.position),
        size=size,
        color=color,
        mass=obj.mass,
    ))
    _update()


def _build_fixed_cuboid(world, stage, obj, _update) -> None:
    """Create a FixedCuboid (static, no physics)."""
    from omni.isaac.core.objects import FixedCuboid

    color = np.array(obj.color) if obj.color else np.array([0.5, 0.5, 0.5])
    size = obj.size[0] if obj.size else 0.04

    world.scene.add(FixedCuboid(
        prim_path=obj.prim_path,
        name=obj.name,
        position=np.array(obj.position),
        size=size,
        color=color,
    ))
    _update()


def _build_usd_object(stage, obj, nucleus_dir, _update) -> None:
    """Load a USD asset and apply physics APIs."""
    from omni.isaac.core.utils.stage import add_reference_to_stage
    from pxr import UsdGeom, UsdPhysics, Gf

    usd_path = f"{nucleus_dir}/{obj.usd_path}" if obj.usd_path else None
    if usd_path is None:
        print(f"[Scene] No usd_path for object '{obj.name}', skipping")
        return

    try:
        add_reference_to_stage(usd_path=usd_path, prim_path=obj.prim_path)
    except Exception as e:
        print(f"[Scene] USD object '{obj.name}' failed to load: {e}")
        return
    _update()

    prim = stage.GetPrimAtPath(obj.prim_path)
    if prim.IsValid():
        xform = UsdGeom.Xformable(prim)
        xform.ClearXformOpOrder()
        xform.AddTranslateOp().Set(Gf.Vec3d(*obj.position))
        if obj.rotation != [0.0, 0.0, 0.0]:
            xform.AddRotateXYZOp().Set(Gf.Vec3f(*obj.rotation))
        xform.AddScaleOp().Set(Gf.Vec3d(*obj.scale))

        if not prim.HasAPI(UsdPhysics.RigidBodyAPI):
            UsdPhysics.RigidBodyAPI.Apply(prim)
        if not prim.HasAPI(UsdPhysics.MassAPI):
            UsdPhysics.MassAPI.Apply(prim).CreateMassAttr(obj.mass)
        if not prim.HasAPI(UsdPhysics.CollisionAPI):
            UsdPhysics.CollisionAPI.Apply(prim)
