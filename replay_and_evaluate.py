"""
replay_and_evaluate.py

Isaac Sim replay engine with integrated Sim-as-a-Judge evaluation.
Based on the test1.py replay pipeline — loads robot USD, parses LeRobot parquet,
replays trajectories with physics stepping, and runs per-frame quality checkers.

Usage:
    python replay_and_evaluate.py \
        --parquet episode_000000.parquet \
        --output-video results/replay.mp4 \
        --output-report results/eval_report.json
"""

import sys
import os
import time
import argparse

# ---------------------------------------------------------
# 1. Runtime environment injection (Isaac Sim requirement)
# ---------------------------------------------------------
os.environ["AGREE_TO_EULA"] = "Y"
os.environ["ACCEPT_EULA"] = "Y"
omni_server = os.environ.get("OMNI_SERVER", "localhost")

sys.argv.extend([
    f"--/persistent/isaac/asset_root/default={omni_server}",
    "--merge-config=/isaac-sim/config/open_endpoint.toml",
    "--allow-root"
])

from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": True})

# ---------------------------------------------------------
# 2. Dependencies
# ---------------------------------------------------------
import omni.kit.pipapi
omni.kit.pipapi.install("pandas")
omni.kit.pipapi.install("pyarrow")
omni.kit.pipapi.install("datasets")
omni.kit.pipapi.install("opencv-python-headless")

import numpy as np
import pandas as pd
from datasets import Dataset
import cv2

from omni.isaac.core import World
from omni.isaac.core.objects import FixedCuboid, DynamicCylinder
from omni.isaac.core.robots import Robot
from omni.isaac.core.utils.stage import add_reference_to_stage
from omni.isaac.core.utils.viewports import set_camera_view
from omni.isaac.sensor import Camera
import omni.usd
from pxr import UsdGeom, UsdPhysics, Usd, Gf, PhysxSchema

# Sim-as-a-Judge checkers
from sim_judge import (
    JointLimitChecker,
    CollisionChecker,
    GravityChecker,
    TaskFidelityChecker,
    aggregate,
    save_json,
    print_verdict,
)


# ---------------------------------------------------------
# Helper: get world-space translation from a USD prim
# ---------------------------------------------------------
def get_prim_world_translate(stage, prim_path: str) -> np.ndarray:
    """Read a prim's world-space position."""
    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        return np.zeros(3)
    xformable = UsdGeom.Xformable(prim)
    world_mtx = xformable.ComputeLocalToWorldTransform(Usd.TimeCode.Default())
    t = world_mtx.ExtractTranslation()
    return np.array([t[0], t[1], t[2]])


# ---------------------------------------------------------
# Joint mapping: numbers of DOF described on parquet → Isaac Sim DOF indices
# ---------------------------------------------------------
EXPECTED_JOINTS = [
    "Please define all the joint name that you use or described on dataset."
]


def main():
    parser = argparse.ArgumentParser(description="Sim-as-a-Judge: Replay and Evaluate")
    parser.add_argument("--parquet", type=str, default="episode_000000.parquet")
    parser.add_argument("--robot-usd", type=str, default=None, help="Robot USD path (auto-detected if not set)")
    parser.add_argument("--output-video", type=str, default="results/replay.mp4")
    parser.add_argument("--output-report", type=str, default="results/eval_report.json")
    parser.add_argument("--fps", type=int, default=30)
    args, _ = parser.parse_known_args()

    world = World(stage_units_in_meters=1.0)
    stage = omni.usd.get_context().get_stage()

    # =========================================================
    # SCENE SETUP
    # =========================================================
    ISAAC_NUCLEUS_DIR = "http://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/4.5/Isaac"
    ISAAC_NUCLEUS_DIRN = "https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/ArchVis"
    TABLE_USD = f"{ISAAC_NUCLEUS_DIR}/Props/PackingTable/packing_table.usd"
    OFFICE_USD = f"{ISAAC_NUCLEUS_DIR}/Environments/Office/office.usd"
    APPLE_USD = f"{ISAAC_NUCLEUS_DIRN}/Residential/Food/Fruit/apple.usd"
    PAN_USD = f"{ISAAC_NUCLEUS_DIRN}/Residential/Kitchen/Kitchenware/Cookware/pan_B.usd"

    APPLE_POS = Gf.Vec3d(0.4, -0.3, 0.7)
    PAN_POS = Gf.Vec3d(0.6, -0.3, 0.7)

    # Environment
    try:
        add_reference_to_stage(usd_path=OFFICE_USD, prim_path="/World/Environment")
        print("[Scene] Environment loaded")
    except Exception as e:
        print(f"[WARN] Environment failed: {e}, using default ground")
        world.scene.add_default_ground_plane()

    # Table
    add_reference_to_stage(usd_path=TABLE_USD, prim_path="/World/PackingTable")
    for _ in range(3):
        simulation_app.update()
    table_prim = stage.GetPrimAtPath("/World/PackingTable")
    if table_prim.IsValid():
        xform = UsdGeom.Xformable(table_prim)
        xform.ClearXformOpOrder()
        xform.AddTranslateOp().Set(Gf.Vec3d(0.5, -0.2, 0.0))
        xform.AddRotateXYZOp().Set(Gf.Vec3f(0.0, 0.0, -90.0))
        xform.AddScaleOp().Set(Gf.Vec3d(1.0, 1.0, 0.7))

    # Pan
    add_reference_to_stage(usd_path=PAN_USD, prim_path="/World/Pan")
    for _ in range(3):
        simulation_app.update()
    pan_prim = stage.GetPrimAtPath("/World/Pan")
    if pan_prim.IsValid():
        xform = UsdGeom.Xformable(pan_prim)
        xform.ClearXformOpOrder()
        xform.AddTranslateOp().Set(PAN_POS)
        xform.AddRotateXYZOp().Set(Gf.Vec3f(0.0, 0.0, 90.0))
        xform.AddScaleOp().Set(Gf.Vec3d(1.0, 1.0, 1.0))
        if not pan_prim.HasAPI(UsdPhysics.CollisionAPI):
            UsdPhysics.CollisionAPI.Apply(pan_prim)

    # Apple (manipulation object)
    add_reference_to_stage(usd_path=APPLE_USD, prim_path="/World/Apple")
    for _ in range(3):
        simulation_app.update()
    apple_prim = stage.GetPrimAtPath("/World/Apple")
    if apple_prim.IsValid():
        xform = UsdGeom.Xformable(apple_prim)
        xform.ClearXformOpOrder()
        xform.AddTranslateOp().Set(APPLE_POS)
        xform.AddScaleOp().Set(Gf.Vec3d(1.0, 1.0, 1.0))
        if not apple_prim.HasAPI(UsdPhysics.RigidBodyAPI):
            UsdPhysics.RigidBodyAPI.Apply(apple_prim)
        if not apple_prim.HasAPI(UsdPhysics.MassAPI):
            UsdPhysics.MassAPI.Apply(apple_prim).CreateMassAttr(0.2)
        if not apple_prim.HasAPI(UsdPhysics.CollisionAPI):
            UsdPhysics.CollisionAPI.Apply(apple_prim)

    # =========================================================
    # ROBOT
    # =========================================================
    current_dir = os.getcwd()
    if args.robot_usd and os.path.exists(args.robot_usd):
        robot_usd_path = args.robot_usd
    else:
        possible = [
            os.path.join(current_dir, "robot_asset_name.usd"),
            os.path.join(current_dir, "robot_asset_name", "robot_asset_name.usd"),
            "/robot_asset_path/robot_asset_name.usd",
        ]
        robot_usd_path = next((p for p in possible if os.path.exists(p)), None)

    if robot_usd_path is None:
        print("[FATAL] Robot USD not found!")
        simulation_app.close()
        sys.exit(1)

    print(f"[System] Robot USD: {robot_usd_path}")
    add_reference_to_stage(usd_path=robot_usd_path, prim_path="/World/G1")
    for _ in range(5):
        simulation_app.update()

    ROBOT_POSITION = np.array([0.0, 0.0, 0.8])
    ROBOT_ORIENT = np.array([1.0, 0.0, 0.0, 0.0])

    robot = world.scene.add(Robot(
        prim_path="/World/robot", name="your_robot_name", position=ROBOT_POSITION
    ))

    # Camera
    RECORD_CAM_PATH = "/World/RecordCamera"
    CAM_EYE = np.array([3.0, -1.5, 2.0])
    CAM_TARGET = np.array([0.2, 0.0, 0.7])
    UsdGeom.Camera.Define(stage, RECORD_CAM_PATH)
    set_camera_view(eye=CAM_EYE, target=CAM_TARGET, camera_prim_path=RECORD_CAM_PATH)

    world.reset()
    print(f"[System] Robot DOF: {robot.num_dof} | DOF names: {robot.dof_names[:5]}...")

    # =========================================================
    # LOAD PARQUET
    # =========================================================
    parquet_path = args.parquet
    if not os.path.exists(parquet_path):
        parquet_path = next(
            (p for p in [
                os.path.join(current_dir, args.parquet),
                f"/isaac-sim/{args.parquet}",
            ] if os.path.exists(p)),
            None,
        )
    if parquet_path is None:
        print(f"[FATAL] Parquet not found: {args.parquet}")
        simulation_app.close()
        sys.exit(1)

    ds = Dataset.from_parquet(parquet_path)
    df = ds.to_pandas()
    print(f"[System] Loaded {len(df)} frames from {parquet_path}")

    # =========================================================
    # JOINT MAPPING
    # =========================================================
    isaac_dof_names = list(robot.dof_names)
    mapping = {}
    for i, name in enumerate(EXPECTED_JOINTS):
        if name in isaac_dof_names:
            mapping[i] = isaac_dof_names.index(name)
    print(f"[System] Mapped {len(mapping)}/{len(EXPECTED_JOINTS)} joints")

    # =========================================================
    # INITIALIZE CHECKERS
    # =========================================================
    n_joints = robot.num_dof
    dt = 1.0 / args.fps

    # Joint limits (arm: ±π, hand: 0~1 as defaults; ideally read from USD)
    mapped_names = [EXPECTED_JOINTS[i] for i in sorted(mapping.keys())]
    lower_limits = np.array([-3.14] * 14 + [0.0] * 12)
    upper_limits = np.array([3.14] * 14 + [1.0] * 12)

    jl_checker = JointLimitChecker(mapped_names, lower_limits, upper_limits, tolerance=0.05)
    col_checker = CollisionChecker(force_threshold=1.0)
    grav_checker = GravityChecker(dt=dt, upward_accel_threshold=4.9)
    task_checker = TaskFidelityChecker(
        target_position=[PAN_POS[0], PAN_POS[1], PAN_POS[2]],
        reach_threshold=0.10,
        grasp_threshold=0.08,
        lift_min_delta=0.05,
        place_threshold=0.10,
    )

    # =========================================================
    # VIDEO WRITER
    # =========================================================
    record_camera = Camera(prim_path=RECORD_CAM_PATH, resolution=(1280, 720))
    record_camera.initialize()

    for _ in range(20):
        world.step(render=True)

    os.makedirs(os.path.dirname(args.output_video) or ".", exist_ok=True)
    writer = cv2.VideoWriter(
        args.output_video, cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (1280, 720)
    )

    # =========================================================
    # REPLAY + EVALUATE (core loop)
    # =========================================================
    print(f"\n[System] Replay + Evaluation started ({len(df)} frames)")

    for idx, row in df.iterrows():
        obs_state = np.array(row["observation.state"], dtype=np.float32)

        # --- Apply action ---
        target_positions = np.zeros(n_joints, dtype=np.float32)
        for parquet_idx, isaac_idx in mapping.items():
            if parquet_idx < len(obs_state):
                target_positions[isaac_idx] = obs_state[parquet_idx]

        robot.set_world_pose(position=ROBOT_POSITION, orientation=ROBOT_ORIENT)
        robot.set_joint_positions(target_positions)
        world.step(render=True)

        # --- Collect physics state ---
        joint_pos = robot.get_joint_positions()
        mapped_joint_pos = np.array([joint_pos[mapping[i]] for i in sorted(mapping.keys())])

        object_pos = get_prim_world_translate(stage, "/World/Apple")

        # EE position (left wrist as end-effector)
        ee_pos = get_prim_world_translate(stage, "/World/G1/left_wrist_yaw_link")
        if np.allclose(ee_pos, 0):
            # Fallback: try alternative prim paths
            for path in ["/World/G1/pelvis/left_wrist_yaw_link",
                         "/World/G1/base_link/left_wrist_yaw_link"]:
                ee_pos = get_prim_world_translate(stage, path)
                if not np.allclose(ee_pos, 0):
                    break

        # Hand closed heuristic: mean of hand joint positions > 0.5
        hand_joints = mapped_joint_pos[14:]  # last 12 = hand
        hand_closed = float(np.mean(hand_joints)) > 0.5

        # --- Run checkers on this frame ---
        jl_checker.check_frame(idx, mapped_joint_pos)
        col_checker.check_frame_no_sensor(idx, joint_pos,
                                           np.full(n_joints, -3.14),
                                           np.full(n_joints, 3.14))
        grav_checker.check_frame(idx, object_pos)
        task_checker.check_frame(idx, ee_pos, object_pos, hand_closed)

        # --- Record video ---
        img_rgba = record_camera.get_rgba()
        if img_rgba is not None and img_rgba.size > 0:
            writer.write(cv2.cvtColor(img_rgba[:, :, :3], cv2.COLOR_RGB2BGR))

        if idx % 100 == 0:
            print(f"  Frame {idx}/{len(df)} | obj={object_pos.round(3)} | ee={ee_pos.round(3)}")

    writer.release()
    print(f"\n[System] Video saved: {args.output_video}")

    # =========================================================
    # AGGREGATE RESULTS
    # =========================================================
    verdict = aggregate(
        joint_limit=jl_checker.result(),
        collision=col_checker.result(),
        gravity=grav_checker.result(),
        task_fidelity=task_checker.result(),
    )

    print_verdict(verdict, episode_name=os.path.basename(parquet_path))

    # Save JSON report
    os.makedirs(os.path.dirname(args.output_report) or ".", exist_ok=True)
    save_json([verdict], args.output_report)

    simulation_app.close()
    print("[System] Done.")


if __name__ == "__main__":
    main()
