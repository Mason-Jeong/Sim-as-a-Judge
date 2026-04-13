"""
replay_and_evaluate.py

Isaac Sim replay engine with integrated Sim-as-a-Judge evaluation.
Based on the test1.py replay pipeline — loads robot USD, parses LeRobot parquet,
replays trajectories with physics stepping, and runs per-frame quality checkers.

Usage (single episode):
    python replay_and_evaluate.py \
        --parquet data/data/chunk-000/episode_000001.parquet \
        --robot-usd assets/g1_29dof/g1-29dof-inspire-base-fix-usd/g1_29dof_with_inspire_rev_1_0.usd \
        --output-video results/replay.mp4 \
        --output-report results/eval_report.json

Usage (batch — all parquet files in a directory):
    python replay_and_evaluate.py \
        --parquet-dir data/data/chunk-000 \
        --robot-usd assets/g1_29dof/g1-29dof-inspire-base-fix-usd/g1_29dof_with_inspire_rev_1_0.usd \
        --output-dir results/batch \
        --output-report results/batch_report.json \
        --no-video
"""

import sys
import os
import argparse
from pathlib import Path

# ---------------------------------------------------------
# 1. Runtime environment injection (Isaac Sim requirement)
# ---------------------------------------------------------
os.environ["AGREE_TO_EULA"] = "Y"
os.environ["ACCEPT_EULA"] = "Y"
omni_server = os.environ.get("OMNI_SERVER", "localhost")

isaac_sim_args = [f"--/persistent/isaac/asset_root/default={omni_server}"]
# Docker container config — only add if it exists
open_endpoint_toml = "/isaac-sim/config/open_endpoint.toml"
if os.path.exists(open_endpoint_toml):
    isaac_sim_args.append(f"--merge-config={open_endpoint_toml}")
isaac_sim_args.append("--allow-root")
sys.argv.extend(isaac_sim_args)

from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": True})

# ---------------------------------------------------------
# 2. Dependencies (pre-installed via: pip install -r requirements.txt)
# ---------------------------------------------------------
import numpy as np
import pandas as pd
from datasets import Dataset
import cv2

from omni.isaac.core import World
from omni.isaac.core.objects import FixedCuboid
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
from sim_judge.config_loader import load_eval_config, load_joint_map, load_task_config
from sim_judge.batch import resolve_parquet_inputs, episode_output_paths
from sim_judge.scene_config import detect_task_from_parquet, resolve_scene_config
from sim_judge.scene_builder import build_scene
from sim_judge.json_data_loader import load_episode_json, detect_data_format


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


def main():
    parser = argparse.ArgumentParser(description="Sim-as-a-Judge: Replay and Evaluate")
    parser.add_argument("--parquet", type=str, default=None,
                        help="Single parquet file to evaluate")
    parser.add_argument("--parquet-dir", type=str, default=None,
                        help="Directory of parquet files to batch-evaluate (overrides --parquet)")
    parser.add_argument("--episode-dir", type=str, default=None,
                        help="JSON episode directory (data.json + colors/ format)")
    parser.add_argument("--robot-usd", type=str, default=None, help="Robot USD path (auto-detected if not set)")
    parser.add_argument("--eval-config", type=str, default="configs/eval_config.yaml",
                        help="Path to eval_config.yaml")
    parser.add_argument("--joint-map", type=str, default="configs/joint_maps/g1_inspire.yaml",
                        help="Path to joint map YAML")
    parser.add_argument("--task-config", type=str, default="configs/pick_place.yaml",
                        help="Path to task config YAML")
    parser.add_argument("--scenes-dir", type=str, default="configs/scenes",
                        help="Directory of scene config YAMLs")
    parser.add_argument("--tasks-jsonl", type=str, default="data/meta/tasks.jsonl",
                        help="Path to tasks.jsonl for task detection")
    parser.add_argument("--output-video", type=str, default="results/replay.mp4",
                        help="Output video path (single mode only)")
    parser.add_argument("--output-report", type=str, default="results/eval_report.json",
                        help="Output report path (single mode) or batch summary path (batch mode)")
    parser.add_argument("--output-dir", type=str, default="results/batch",
                        help="Output directory for batch mode (per-episode subdirs)")
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--no-video", action="store_true",
                        help="Skip video recording (faster batch evaluation)")
    args, _ = parser.parse_known_args()

    # ---------------------------------------------------------
    # Load configurations from YAML
    # ---------------------------------------------------------
    eval_cfg = load_eval_config(args.eval_config)
    joint_map_cfg = load_joint_map(args.joint_map)
    task_cfg = load_task_config(args.task_config)
    EXPECTED_JOINTS = joint_map_cfg.joint_names()
    print(f"[Config] Loaded eval config: {args.eval_config}")
    print(f"[Config] Loaded joint map: {joint_map_cfg.robot} ({joint_map_cfg.dof} DOF)")
    print(f"[Config] Loaded task config: {task_cfg.task}")

    # ---------------------------------------------------------
    # Detect data format and resolve inputs
    # ---------------------------------------------------------
    json_episode = None
    if args.episode_dir is not None:
        # JSON episode mode
        json_episode = load_episode_json(args.episode_dir)
        parquet_files = []
        is_batch = False
        task_description = json_episode.task_goal
        print(f"[System] JSON episode mode: {json_episode.num_frames} frames")
        print(f"[Task] Task from data.json: '{task_description}'")
    else:
        # Parquet mode
        parquet_files = resolve_parquet_inputs(
            parquet=args.parquet, parquet_dir=args.parquet_dir
        )
        is_batch = args.parquet_dir is not None
        if not parquet_files:
            print("[FATAL] No parquet files found.")
            simulation_app.close()
            sys.exit(1)
        print(f"[System] {'Batch' if is_batch else 'Single'} mode: {len(parquet_files)} episode(s) to evaluate")

        first_parquet = str(parquet_files[0])
        task_description = detect_task_from_parquet(first_parquet, args.tasks_jsonl)
        print(f"[Task] Detected task: '{task_description}'")

    world = World(stage_units_in_meters=1.0)
    stage = omni.usd.get_context().get_stage()

    # =========================================================
    # SCENE SETUP (auto-detected from task description)
    # =========================================================
    ISAAC_NUCLEUS_DIR = "https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/4.5/Isaac"

    # Resolve scene config
    scene_cfg = resolve_scene_config(task_description, args.scenes_dir)
    print(f"[Task] Scene config: {scene_cfg.task} ({len(scene_cfg.objects)} objects)")

    # Build scene from config
    prim_map = build_scene(
        world, stage, scene_cfg, ISAAC_NUCLEUS_DIR, simulation_app
    )

    # =========================================================
    # ROBOT
    # =========================================================
    current_dir = os.getcwd()
    if args.robot_usd and os.path.exists(args.robot_usd):
        robot_usd_path = args.robot_usd
    else:
        # Default: G1 29DOF Inspire hand (base-fixed) — matches LeRobot parquet data
        default_usd = os.path.join(
            current_dir, "assets", "g1_29dof", "g1-29dof-inspire-base-fix-usd",
            "g1_29dof_with_inspire_rev_1_0.usd",
        )
        robot_usd_path = default_usd if os.path.exists(default_usd) else None

    if robot_usd_path is None:
        print("[FATAL] Robot USD not found!")
        simulation_app.close()
        sys.exit(1)

    print(f"[System] Robot USD: {robot_usd_path}")
    add_reference_to_stage(usd_path=robot_usd_path, prim_path="/World/G1")
    for _ in range(5):
        simulation_app.update()

    ROBOT_POSITION = np.array([-0.15, 0.0, 0.8])
    ROBOT_ORIENT = np.array([1.0, 0.0, 0.0, 0.0])

    robot = world.scene.add(Robot(
        prim_path="/World/G1", name="G1", position=ROBOT_POSITION
    ))

    # Camera — wide angle to capture full scene
    RECORD_CAM_PATH = "/World/RecordCamera"
    CAM_EYE = np.array([2.5, 0.0, 2.2])
    CAM_TARGET = np.array([0.5, 0.0, 0.5])
    cam_prim = UsdGeom.Camera.Define(stage, RECORD_CAM_PATH)
    cam_prim.CreateHorizontalApertureAttr(36.0)
    cam_prim.CreateVerticalApertureAttr(20.25)
    cam_prim.CreateFocalLengthAttr(18.0)
    set_camera_view(eye=CAM_EYE, target=CAM_TARGET, camera_prim_path=RECORD_CAM_PATH)

    world.reset()
    print(f"[System] Robot DOF: {robot.num_dof}")
    isaac_dof_names = list(robot.dof_names)
    print(f"[System] All DOF names ({len(isaac_dof_names)}):")
    for idx, name in enumerate(isaac_dof_names):
        print(f"  [{idx:2d}] {name}")

    # =========================================================
    # JOINT MAPPING (shared across all episodes)
    # =========================================================
    mapping = {}
    for i, name in enumerate(EXPECTED_JOINTS):
        if name in isaac_dof_names:
            mapping[i] = isaac_dof_names.index(name)
        else:
            print(f"[WARN] Joint '{name}' (parquet idx {i}) not found in USD")
    print(f"[System] Mapped {len(mapping)}/{len(EXPECTED_JOINTS)} joints")

    n_joints = robot.num_dof
    dt = 1.0 / args.fps
    mapped_names = [EXPECTED_JOINTS[i] for i in sorted(mapping.keys())]

    # Joint limits (arm: ±π, hand: 0~1 as defaults; ideally read from USD)
    n_mapped = len(mapped_names)
    N_ARM_JOINTS = min(14, n_mapped)
    N_HAND_JOINTS = n_mapped - N_ARM_JOINTS
    if n_mapped != joint_map_cfg.dof:
        print(f"[WARN] Expected {joint_map_cfg.dof} mapped joints, got {n_mapped}. "
              f"Update configs/joint_maps to match robot USD joint names.")
    lower_limits = np.array([-3.14] * N_ARM_JOINTS + [0.0] * N_HAND_JOINTS)
    upper_limits = np.array([3.14] * N_ARM_JOINTS + [1.0] * N_HAND_JOINTS)

    # =========================================================
    # CAMERA SETUP (shared across all episodes)
    # =========================================================
    record_camera = Camera(prim_path=RECORD_CAM_PATH, resolution=(1280, 720))
    record_camera.initialize()
    for _ in range(20):
        world.step(render=True)

    # =========================================================
    # EPISODE LOOP — evaluate parquet files or JSON episode
    # =========================================================
    all_verdicts = []

    # Build episode list: [(episode_name, data_source), ...]
    if json_episode is not None:
        episode_list = [(Path(args.episode_dir).name, "json")]
    else:
        episode_list = [(os.path.basename(str(p)), str(p)) for p in parquet_files]

    for ep_idx, (episode_name, data_source) in enumerate(episode_list):

        print(f"\n{'=' * 60}")
        print(f"[Episode {ep_idx + 1}/{len(episode_list)}] {episode_name}")
        print(f"{'=' * 60}")

        # --- Determine output paths ---
        if is_batch:
            ep_outputs = episode_output_paths(
                parquet_path=Path(data_source),
                output_dir=args.output_dir,
            )
            video_path = str(ep_outputs["video"])
            report_path = str(ep_outputs["report"])
        else:
            video_path = args.output_video
            report_path = args.output_report

        # --- Load data ---
        if data_source == "json":
            num_frames = json_episode.num_frames
            print(f"[System] JSON episode: {num_frames} frames @ {json_episode.fps} fps")
        else:
            if not os.path.exists(data_source):
                print(f"[WARN] Parquet not found, skipping: {data_source}")
                continue
            ds = Dataset.from_parquet(data_source)
            df = ds.to_pandas()
            num_frames = len(df)
            print(f"[System] Loaded {num_frames} frames")

        # --- Initialize checkers (fresh per episode) ---
        jl_checker = JointLimitChecker(
            mapped_names, lower_limits, upper_limits,
            tolerance=eval_cfg.joint_limits_tolerance,
        )
        col_checker = CollisionChecker(
            force_threshold=eval_cfg.collision_force_threshold,
        )
        grav_checker = GravityChecker(
            dt=dt,
            upward_accel_threshold=eval_cfg.gravity_upward_accel_threshold,
        )
        task_checker = TaskFidelityChecker(
            target_position=task_cfg.place_target_position,
            reach_threshold=task_cfg.reach_threshold,
            grasp_threshold=task_cfg.grasp_threshold,
            lift_min_delta=task_cfg.lift_min_delta,
            place_threshold=task_cfg.place_threshold,
            subtask_weights=task_cfg.subtask_weights,
        )

        # --- Video writer (optional) ---
        writer = None
        if not args.no_video:
            os.makedirs(os.path.dirname(video_path) or ".", exist_ok=True)
            writer = cv2.VideoWriter(
                video_path, cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (1280, 720)
            )

        # --- Reset scene objects for new episode ---
        for obj in scene_cfg.objects:
            obj_prim = stage.GetPrimAtPath(obj.prim_path)
            if obj_prim.IsValid():
                xformable = UsdGeom.Xformable(obj_prim)
                existing_ops = xformable.GetOrderedXformOps()
                for op in existing_ops:
                    if op.GetOpType() == UsdGeom.XformOp.TypeTranslate:
                        op.Set(Gf.Vec3d(*obj.position))
                        break
                else:
                    # No translate op exists — set via ClearXformOpOrder
                    xformable.ClearXformOpOrder()
                    xformable.AddTranslateOp().Set(Gf.Vec3d(*obj.position))

        # --- Replay + evaluate per frame ---
        for idx in range(num_frames):
            # Assemble observation state from data source
            if data_source == "json":
                obs_state = json_episode.get_obs_state(idx)
            else:
                row = df.iloc[idx]
                left_arm = np.array(row["observation.left_arm"], dtype=np.float32)
                right_arm = np.array(row["observation.right_arm"], dtype=np.float32)
                GRIP_MIN, GRIP_MAX = 2.0, 5.0
                left_grip_raw = float(row["observation.left_gripper"])
                right_grip_raw = float(row["observation.right_gripper"])
                left_grip_norm = 1.0 - np.clip((left_grip_raw - GRIP_MIN) / (GRIP_MAX - GRIP_MIN), 0, 1)
                right_grip_norm = 1.0 - np.clip((right_grip_raw - GRIP_MIN) / (GRIP_MAX - GRIP_MIN), 0, 1)
                left_hand = np.full(6, left_grip_norm, dtype=np.float32)
                right_hand = np.full(6, right_grip_norm, dtype=np.float32)
                obs_state = np.concatenate([left_arm, right_arm, left_hand, right_hand])

            target_positions = np.zeros(n_joints, dtype=np.float32)
            for parquet_idx, isaac_idx in mapping.items():
                if parquet_idx < len(obs_state):
                    target_positions[isaac_idx] = obs_state[parquet_idx]

            # Set waist joints for forward lean (if data doesn't provide body joints)
            for wi, wname in enumerate(isaac_dof_names):
                if wname == "waist_pitch_joint":
                    target_positions[wi] = 0.3  # lean forward
                elif wname == "waist_yaw_joint":
                    target_positions[wi] = 0.0
                elif wname == "waist_roll_joint":
                    target_positions[wi] = 0.0

            robot.set_world_pose(position=ROBOT_POSITION, orientation=ROBOT_ORIENT)
            robot.set_joint_positions(target_positions)
            world.step(render=True)

            # Collect physics state
            joint_pos = robot.get_joint_positions()
            mapped_joint_pos = np.array([joint_pos[mapping[i]] for i in sorted(mapping.keys())])
            # Track target object position (from scene config)
            if scene_cfg.target_object:
                object_pos = get_prim_world_translate(stage, scene_cfg.target_object)
            else:
                object_pos = np.zeros(3)

            # End-effector position (from scene config)
            ee_pos = get_prim_world_translate(stage, scene_cfg.ee_prim_path)
            if np.allclose(ee_pos, 0):
                for path in ["/World/G1/pelvis/left_wrist_yaw_link",
                             "/World/G1/base_link/left_wrist_yaw_link"]:
                    ee_pos = get_prim_world_translate(stage, path)
                    if not np.allclose(ee_pos, 0):
                        break

            hand_joints = mapped_joint_pos[N_ARM_JOINTS:]
            # EE convention: higher values = open, lower values = closed
            hand_closed = float(np.mean(hand_joints)) < 0.5

            # Run checkers
            jl_checker.check_frame(idx, mapped_joint_pos)
            col_checker.check_frame_no_sensor(idx, joint_pos,
                                               np.full(n_joints, -3.14),
                                               np.full(n_joints, 3.14))
            grav_checker.check_frame(idx, object_pos)
            task_checker.check_frame(idx, ee_pos, object_pos, hand_closed)

            # Record video
            if writer is not None:
                img_rgba = record_camera.get_rgba()
                if img_rgba is not None and img_rgba.size > 0:
                    writer.write(cv2.cvtColor(img_rgba[:, :, :3], cv2.COLOR_RGB2BGR))

            if idx % 100 == 0:
                print(f"  Frame {idx}/{num_frames} | obj={object_pos.round(3)} | ee={ee_pos.round(3)}")

        if writer is not None:
            writer.release()
            print(f"[System] Video saved: {video_path}")

        # --- Aggregate this episode ---
        verdict = aggregate(
            joint_limit=jl_checker.result(),
            collision=col_checker.result(),
            gravity=grav_checker.result(),
            task_fidelity=task_checker.result(),
            weights=eval_cfg.metric_weights,
            pass_threshold=eval_cfg.pass_threshold,
        )
        all_verdicts.append(verdict)
        print_verdict(verdict, episode_name=episode_name)

        # Save per-episode report
        os.makedirs(os.path.dirname(report_path) or ".", exist_ok=True)
        save_json([verdict], report_path)

    # =========================================================
    # BATCH SUMMARY (if multiple episodes)
    # =========================================================
    if is_batch and all_verdicts:
        batch_report_path = args.output_report
        os.makedirs(os.path.dirname(batch_report_path) or ".", exist_ok=True)
        save_json(all_verdicts, batch_report_path)

        pass_count = sum(1 for v in all_verdicts if v.passed)
        total = len(all_verdicts)
        avg_score = sum(v.overall_score for v in all_verdicts) / total

        print(f"\n{'=' * 60}")
        print(f" BATCH SUMMARY")
        print(f"{'=' * 60}")
        print(f"  Total episodes:  {total}")
        print(f"  Passed:          {pass_count}")
        print(f"  Failed:          {total - pass_count}")
        print(f"  Pass rate:       {pass_count / total:.1%}")
        print(f"  Avg score:       {avg_score:.4f}")
        print(f"  Batch report:    {batch_report_path}")
        print(f"{'=' * 60}")

    simulation_app.close()
    print("[System] Done.")


if __name__ == "__main__":
    main()
