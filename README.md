# Sim-as-a-Judge

**A Physics-Grounded Validation Framework for Synthetic Robot Trajectory Data**

[![arXiv](https://img.shields.io/badge/arXiv-2025.XXXXX-b31b1b.svg)](https://arxiv.org/abs/2025.XXXXX)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.10+-green.svg)](https://www.python.org/)
[![Isaac Sim](https://img.shields.io/badge/Isaac%20Sim-5.0-76B900.svg)](https://developer.nvidia.com/isaac-sim)

<p align="center">
  <img src="docs/figures/pipeline_overview.png" width="800"/>
</p>
## Introduction

Sim-as-a-Judge uses **physics simulation as an automated judge** to evaluate whether synthetic robot trajectories (e.g., from World Foundation Models, teleoperation, or generative pipelines) are physically plausible and task-faithful — before deploying them for policy learning.

## Motivation

In the Physical AI era, data augmentation pipelines (World Foundation Models → action extraction model → Making trajectory) are becoming the primary source of robot training data. But **not all synthetic data is created equal**:

- Trajectories may violate joint limits or physics constraints
- Generated motions may collide with the environment or the robot itself
- Actions may not faithfully execute the intended task

**Sim-as-a-Judge** addresses this by replaying trajectories inside a physics simulator (NVIDIA Isaac Sim with PhysX) and computing quantitative quality metrics — acting as an automated, scalable quality gate for robot data pipelines.

## Key Features

| Metric | Description | Output |
|--------|-------------|--------|
| **Collision Detection** | PhysX-based collision checking between robot links, objects, and environment | Binary pass/fail + contact force magnitude |
| **Joint Limit Violation** | Per-joint position/velocity/torque limit checking against URDF/USD specs | Violation count + severity per joint |
| **Gravity Consistency** | Validates that object/end-effector motions respect gravitational constraints | Gravity violation score |
| **Task Fidelity** | Measures whether the trajectory achieves the intended manipulation goal | Task completion score (0–1) |

## Pipeline

```
┌──────────────────┐     ┌───────────────────┐     ┌──────────────────┐
│  Synthetic Data   │     │   Isaac Sim +      │     │  Quality Report  │
│  (parquet/LeRobot)│────▶│   PhysX Replay     │────▶│  pass/fail +     │
│                   │     │                    │     │  per-metric score│
└──────────────────┘     └───────────────────┘     └──────────────────┘
        │                         │                         │
   action joints           collision check             CSV / JSON
   observation.state       joint limit check           visualization
   timestamps              gravity check               classification
                           task fidelity
```

## Installation

### Prerequisites

- NVIDIA Isaac Sim 5.0 (Isaac Lab)
- Python 3.10+
- CUDA-compatible GPU (16GB+ VRAM)

### Setup

```bash
git clone https://github.com/<your-org>/sim-as-a-judge.git
cd sim-as-a-judge
pip install -r requirements.txt
```

### Isaac Sim Integration

```bash
# Activate Isaac Lab conda environment
conda activate isaaclab

# Install sim-as-a-judge as editable package
pip install -e .
```

## Quick Start

### 1. Evaluate a single trajectory

```bash
python scripts/run_eval.py \
    --trajectory data/sample/episode_000000.parquet \
    --robot-usd assets/robots/unitree_g1_29dof.usd \
    --scene-usd assets/scenes/table_scene.usd \
    --output results/eval_report.json
```

### 2. Batch evaluation

```bash
python scripts/run_eval.py \
    --trajectory-dir data/episodes/ \
    --robot-usd assets/robots/unitree_g1_29dof.usd \
    --scene-usd assets/scenes/table_scene.usd \
    --output results/batch_report.json \
    --num-workers 4
```

### 3. Generate visual report

```bash
python scripts/gen_report.py \
    --input results/batch_report.json \
    --output results/report.html
```

## Data Format

Input trajectories follow the **LeRobot v2 parquet** format:

| Column | Type | Description |
|--------|------|-------------|
| `frame_index` | int64 | Timestep index |
| `episode_index` | int64 | Episode identifier |
| `timestamp` | float64 | Time in seconds |
| `observation.state` | list[float64] | Robot joint state (N-DOF) |
| `action` | list[float64] | Target joint positions (N-DOF) |

### Supported Robot Configurations

| Robot | DOF | Joint Mapping |
|-------|-----|---------------|
| Unitree G1 29-DOF + Inspire Hand | 26D (14 arm + 12 hand) | See `configs/joint_maps/g1_inspire.yaml` |

Custom robots can be added by providing a USD file and joint mapping config.

## Evaluation Metrics

### Collision Detection

Uses Isaac Sim's PhysX collision detection to identify:
- **Self-collision**: robot link-to-link penetration
- **Environment collision**: robot-to-scene object penetration
- **Object collision**: unexpected object-to-object contacts

```python
from sim_judge import CollisionChecker

checker = CollisionChecker(robot_usd, scene_usd)
result = checker.evaluate(trajectory)
# result.collision_count, result.max_contact_force, result.collision_frames
```

### Joint Limit Violation

Validates per-joint constraints from USD/URDF:

```python
from sim_judge import JointLimitChecker

checker = JointLimitChecker(robot_usd)
result = checker.evaluate(trajectory)
# result.violations: {joint_name: {count, max_violation_rad, frames}}
```

### Gravity Consistency

Checks whether free-falling objects and end-effector motions are consistent with gravity:

```python
from sim_judge import GravityChecker

checker = GravityChecker()
result = checker.evaluate(trajectory, object_poses)
# result.gravity_score, result.violation_frames
```

### Task Fidelity

Measures task completion based on configurable success criteria:

```python
from sim_judge import TaskFidelityChecker

checker = TaskFidelityChecker(task_config="configs/tasks/pick_place.yaml")
result = checker.evaluate(trajectory, object_poses)
# result.task_score, result.subtask_scores
```

### Combined Judgment

```python
from sim_judge import SimJudge

judge = SimJudge(
    robot_usd="assets/robots/unitree_g1_29dof.usd",
    scene_usd="assets/scenes/table_scene.usd",
    task_config="configs/tasks/pick_place.yaml",
)

verdict = judge.evaluate("data/episode_000000.parquet")
print(verdict)
# SimJudgeVerdict(
#     passed=False,
#     collision_score=0.95,
#     joint_limit_score=0.88,
#     gravity_score=1.00,
#     task_fidelity_score=0.72,
#     overall_score=0.84,
#     failure_reasons=["joint_limit_violation: left_elbow_joint at frame 142"]
# )
```

## Project Structure

```
sim-as-a-judge/
├── README.md
├── LICENSE                          # Apache 2.0
├── setup.py
├── requirements.txt
│
├── sim_judge/                       # Core library
│   ├── __init__.py
│   ├── collision_checker.py         # PhysX collision detection
│   ├── joint_limit_checker.py       # Joint limit validation
│   ├── gravity_checker.py           # Gravity consistency check
│   ├── task_fidelity_checker.py     # Task completion scoring
│   ├── sim_judge.py                 # Combined judge (main API)
│   ├── trajectory_loader.py         # LeRobot parquet loader
│   ├── robot_loader.py              # USD robot loader
│   ├── visualizer.py                # Result visualization
│   └── utils.py
│
├── configs/
│   ├── eval_config.yaml             # Default evaluation settings
│   ├── joint_maps/
│   │   └── g1_inspire.yaml          # G1 29-DOF joint mapping
│   └── tasks/
│       └── pick_place.yaml          # Pick-place task criteria
│
├── scripts/
│   ├── run_eval.py                  # Single/batch evaluation
│   ├── gen_report.py                # HTML/PDF report generation
│   └── replay_trajectory.py         # Visual replay in Isaac Sim
│
├── assets/
│   ├── robots/                      # Robot USD files
│   └── scenes/                      # Scene USD files
│
├── data/
│   └── sample/                      # Sample trajectories
│       └── episode_000000.parquet
│
├── docs/
│   ├── figures/                     # Paper figures
│   │   ├── pipeline_overview.png
│   │   ├── metric_comparison.png
│   │   └── failure_examples.png
│   └── paper/                       # LaTeX source (optional)
│
├── experiments/                     # Experiment configs & results
│   ├── configs/
│   └── results/
│
└── tests/
    ├── test_collision.py
    ├── test_joint_limits.py
    └── test_task_fidelity.py
```

## Configuration

### Evaluation Config (`configs/eval_config.yaml`)

```yaml
physics:
  dt: 0.005
  gravity: [0, 0, -9.81]
  physx:
    bounce_threshold_velocity: 0.01
    friction_correlation_distance: 0.00625

collision:
  enabled: true
  self_collision: true
  env_collision: true
  contact_force_threshold: 1.0     # N

joint_limits:
  enabled: true
  position_tolerance: 0.05         # rad
  velocity_tolerance: 0.1          # rad/s

gravity:
  enabled: true
  free_fall_tolerance: 0.1         # m/s²

task_fidelity:
  enabled: true
  task: pick_place
  success_threshold: 0.8
```

### Joint Mapping (`configs/joint_maps/g1_inspire.yaml`)

```yaml
robot: unitree_g1_29dof_inspire
dof: 26
mapping:
  # parquet_index: joint_name
  0: left_shoulder_pitch_joint
  1: left_shoulder_roll_joint
  2: left_shoulder_yaw_joint
  3: left_elbow_joint
  4: left_wrist_roll_joint
  5: left_wrist_pitch_joint
  6: left_wrist_yaw_joint
  7: right_shoulder_pitch_joint
  8: right_shoulder_roll_joint
  9: right_shoulder_yaw_joint
  10: right_elbow_joint
  11: right_wrist_roll_joint
  12: right_wrist_pitch_joint
  13: right_wrist_yaw_joint
  14: L_index_proximal_joint
  15: L_middle_proximal_joint
  16: L_ring_proximal_joint
  17: L_pinky_proximal_joint
  18: L_thumb_proximal_pitch_joint
  19: L_thumb_proximal_yaw_joint
  20: R_index_proximal_joint
  21: R_middle_proximal_joint
  22: R_ring_proximal_joint
  23: R_pinky_proximal_joint
  24: R_thumb_proximal_pitch_joint
  25: R_thumb_proximal_yaw_joint
```

## Experiment Reproduction

### Evaluating DreamGen / Cosmos synthetic trajectories

```bash
# 1. Convert synthetic data to LeRobot format (if needed)
python scripts/convert_to_lerobot.py --input <synthetic_data_dir> --output data/synthetic/

# 2. Run evaluation
python scripts/run_eval.py \
    --trajectory-dir data/synthetic/ \
    --robot-usd assets/robots/unitree_g1_29dof.usd \
    --scene-usd assets/scenes/table_scene.usd \
    --output experiments/results/synthetic_eval.json

# 3. Generate comparison report
python scripts/gen_report.py \
    --input experiments/results/synthetic_eval.json \
    --baseline experiments/results/real_eval.json \
    --output experiments/results/comparison_report.html
```

## Citation

If you find this work useful, please cite:

```bibtex
@article{jeong2025simjudge,
    title={Sim-as-a-Judge: Physics-Grounded Validation of Synthetic Robot Trajectories},
    author={Jeong, Yongwoo and Choi, Hyerin},
    journal={arXiv preprint arXiv:2025.XXXXX},
    year={2025}
}
```

## Acknowledgments

I gratefully acknowledge the following projects and resources that made this work possible:

**Simulation & Robot Frameworks**
- NVIDIA Isaac Sim(https://docs.isaacsim.omniverse.nvidia.com/5.1.0/index.html) — Physics simulation platform providing PhysX-based collision detection and articulation support
- Unitree RL Lab(https://github.com/unitreerobotics/unitree_rl_lab) — RL training framework for Unitree robots on Isaac Lab
- Unitree Sim IsaacLab(https://github.com/unitreerobotics/unitree_sim_isaaclab) — Isaac Sim integration for Unitree humanoid robots, including G1 29-DOF pick-and-place task configurations

**Synthetic Data & World Models**
- DreamGen(https://arxiv.org/abs/2505.12705) — Synthetic robot data generation pipeline via video world models, whose DreamGen Bench evaluation framework motivated our physics-grounded validation approach
- GR00T-Dreams(https://github.com/NVIDIA/GR00T-Dreams) — NVIDIA's synthetic data generation pipeline for robot learning using Cosmos world foundation models

Thank you NVIDIA Isaac Sim, Cosmos, and GR00T teams for the simulation and world model foundations, and the Unitree Robotics team for open-sourcing the robot simulation frameworks.

## License

This project is licensed under the Apache License 2.0 — see [LICENSE](LICENSE) for details.
