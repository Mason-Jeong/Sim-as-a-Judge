# Sim-as-a-Judge

**A Physics-Grounded Validation Framework for Synthetic Robot Trajectory Data**

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python](https://img.shields.io/badge/Python-3.10+-green.svg)](https://www.python.org/)
[![Isaac Sim](https://img.shields.io/badge/Isaac%20Sim-5.0%2B-76B900.svg)](https://developer.nvidia.com/isaac-sim)

## 
This repository does not guarantee out-of-the-box execution. Code modifications may be required to run, and all experiments were conducted on the Isaac Sim 5.0 Docker container from https://github.com/unitreerobotics/unitree_rl_lab
## Motivation

In the Physical AI era, data augmentation pipelines — from World Foundation Models (e.g., Cosmos) to action extraction — are becoming the primary source of robot training data. But **not all synthetic data is created equal**:

- Trajectories may violate joint limits or physics constraints
- Generated motions may collide with the environment or the robot itself
- Actions may not faithfully execute the intended manipulation task
- Objects may defy gravity or teleport between frames

**How can we tell good data from bad, before it poisons policy learning?**

Sim-as-a-Judge answers this by **replaying trajectories inside a physics simulator** and computing quantitative quality metrics — acting as an automated, scalable quality gate for robot data pipelines.

## Approach

My core idea is simple: **if a trajectory is physically valid, it should survive replay in a physics simulator without anomalies.**

### Pipeline

```
 Real / Synthetic Data          Isaac Sim Replay              Quality Verdict
┌─────────────────────┐     ┌──────────────────────┐     ┌─────────────────────┐
│  LeRobot Parquet    │     │  Load Robot USD      │     │  Per-frame metrics: │
│  - observation.state│───▶│  Load Scene (table,  │───▶│  ✓ Joint limits     │
│  - action (26D)     │     │    objects, env)     │     │  ✓ Collision        │
│  - timestamps       │     │  Map joints 1:1      │     │  ✓ Gravity          │
│                     │     │  Step physics per    │     │  ✓ Task fidelity    │
│                     │     │    frame             │     │                     │
│                     │     │  Record observations │     │  Overall: PASS /FAIL│
└─────────────────────┘     └──────────────────────┘     └─────────────────────┘
```

### What I Built

1. **Isaac Sim Replay** (`replay_and_evaluate.py`)
   - Loads Unitree G1 USD into a configured scene (table, objects, environment)
   - Parses LeRobot v2 parquet files and maps actions to Isaac Sim joint indices
   - Replays trajectories frame-by-frame with physics stepping, collecting per-frame diagnostic data
   - Records MP4 video for visual inspection

2. **Physics-Grounded Checkers** (integrated into the replay loop)
   - **Joint Limit Checker** — flags frames where joint positions exceed USD-defined limits
   - **Collision Checker** — detects self-collision and environment collision via PhysX contact reports
   - **Gravity Checker** — validates that unsupported objects follow gravitational acceleration
   - **Task Fidelity Checker** — scores whether the trajectory achieves subtask goals (reach, grasp, lift, place)

3. **Report Generator** — produces per-trajectory and batch-level quality reports (JSON + HTML)

### Example
- Example of a trajectory flagged as FAIL by Sim-as-a-Judge.
- The object exhibits upward acceleration inconsistent with gravity, and the end-effector fails to reach the grasp target within the threshold distance (task fidelity failure).
- These anomalies, which are difficult to detect from video alone, are automatically captured by per-frame physics checking during simulation replay.
![Image](https://github.com/user-attachments/assets/2a0385d8-dada-4cc7-be0a-30b33446aa67)
![Image](https://github.com/user-attachments/assets/084e19bc-b673-499b-baf2-98bd4518fff4)

### How Checkers Integrate with the Replay Loop

The key insight is that **all validation happens inside the physics step loop**, not as a separate offline pass:

```python
# Inside the Isaac Sim replay loop (simplified)
for t, row in df.iterrows():
    # 1. Apply action
    robot.set_joint_positions(target_positions)
    world.step(render=True)

    # 2. Read physics state (Isaac Sim returns these after each step)
    joint_pos = robot.get_joint_positions()
    joint_vel = robot.get_joint_velocities()
    contact_forces = contact_sensor.get_current_frame()
    object_pos = get_prim_world_transform(object_prim)
    ee_pos = get_prim_world_transform(ee_prim)

    # 3. Each checker scores this frame
    joint_limit_checker.check_frame(t, joint_pos, joint_vel)
    collision_checker.check_frame(t, contact_forces)
    gravity_checker.check_frame(t, object_pos, dt)
    task_fidelity_checker.check_frame(t, ee_pos, object_pos)

# 4. After replay, aggregate per-frame results into verdict
verdict = SimJudge.aggregate(
    joint_limit_checker.result(),
    collision_checker.result(),
    gravity_checker.result(),
    task_fidelity_checker.result(),
)
```

## Evaluation Metrics

| Metric | What It Checks | Signal Source | Output |
|--------|---------------|---------------|--------|
| **Joint Limits** | Position/velocity within USD-defined bounds | `robot.get_joint_positions()` | Violation count + severity per joint |
| **Collision** | Self-collision, robot↔env, robot↔object | PhysX `ContactSensor` | Contact force magnitude + collision frames |
| **Gravity** | Objects don't float or teleport | Object prim world transform | Gravity violation score |
| **Task Fidelity** | EE reaches object, grasps, lifts, places | EE + object world positions | Subtask completion scores (0–1) |

### Joint Mapping (Parquet Index → Isaac Sim Joint) example

```
Parquet [0:7]   → Left arm:  shoulder_pitch/roll/yaw, elbow, wrist_roll/pitch/yaw
Parquet [7:14]  → Right arm: shoulder_pitch/roll/yaw, elbow, wrist_roll/pitch/yaw
Parquet [14:20] → Left hand:  pinky, ring, middle, index, thumb_pitch, thumb_yaw
etc ... 
```

## Project Structure

```
sim-as-a-judge/
├── README.md
├── requirements.txt                # Python dependencies (non-Isaac Sim)
├── pyproject.toml                  # Package metadata & pip install support
│
├── replay_and_evaluate.py          # Main script: single & batch replay + evaluation
│
├── sim_judge/                      # Core validation package
│   ├── __init__.py
│   ├── joint_limit_checker.py      # Per-frame joint limit validation
│   ├── collision_checker.py        # PhysX contact force analysis
│   ├── gravity_checker.py          # Object gravity consistency
│   ├── task_fidelity_checker.py    # Subtask completion scoring
│   ├── aggregate.py                # Combine checkers → verdict
│   ├── report.py                   # JSON report generation
│   ├── config_loader.py            # YAML config loading (eval, joint map, task)
│   ├── batch.py                    # Batch parquet collection & path resolution
│   └── verify_env.py               # Environment dependency checker
│
├── configs/
│   ├── eval_config.yaml            # Metric thresholds and weights
│   ├── pick_place.yaml             # Task success criteria
│   └── joint_maps/
│       └── g1_inspire.yaml         # Parquet ↔ Isaac Sim joint mapping
│
├── scripts/
│   ├── setup_env.sh                # One-click conda environment setup
│   └── verify_env.py               # Environment verification CLI
│
├── tests/                          # pytest test suite
│   ├── test_config_loader.py
│   ├── test_batch_eval.py
│   ├── test_verify_env.py
│   └── test_replay_integration.py
│
├── assets/                         # Robot USD models
│   └── 29dof/usd/g1_29dof_rev_1_0/
│
├── data/                           # Trajectory dataset (LeRobot v2 parquet)
│   ├── data/chunk-000/             # Episode parquet files
│   └── meta/                       # Dataset metadata
│
├── results/                        # Generated reports (gitignored)
│
└── docs/
    └── figures/
```

## Setup

### Prerequisites

- Python 3.10+
- [Isaac Sim 5.0+](https://developer.nvidia.com/isaac-sim) with PhysX
- NVIDIA GPU with CUDA support
- conda (Miniconda or Anaconda)

### Installation

```bash
# Option 1: Automated setup (recommended)
bash scripts/setup_env.sh

# Option 2: Manual setup
conda activate env_isaaclab
pip install -r requirements.txt
pip install -e .   # install sim_judge package

# Verify environment
python scripts/verify_env.py
```

## Usage

### Single Episode Evaluation

```bash
python replay_and_evaluate.py \
    --parquet data/data/chunk-000/episode_000001.parquet \
    --robot-usd assets/g1_29dof/g1-29dof-inspire-base-fix-usd/g1_29dof_with_inspire_rev_1_0.usd \
    --output-video results/replay.mp4 \
    --output-report results/eval_report.json
```

### Batch Evaluation (all episodes in a directory)

```bash
python replay_and_evaluate.py \
    --parquet-dir data/data/chunk-000 \
    --robot-usd assets/g1_29dof/g1-29dof-inspire-base-fix-usd/g1_29dof_with_inspire_rev_1_0.usd \
    --output-dir results/batch \
    --output-report results/batch_report.json
```

### Batch Evaluation (skip video for speed)

```bash
python replay_and_evaluate.py \
    --parquet-dir data/data/chunk-000 \
    --robot-usd assets/g1_29dof/g1-29dof-inspire-base-fix-usd/g1_29dof_with_inspire_rev_1_0.usd \
    --output-dir results/batch \
    --output-report results/batch_report.json \
    --no-video
```

### CLI Reference

| Argument | Default | Description |
|----------|---------|-------------|
| `--parquet` | — | Single parquet file to evaluate |
| `--parquet-dir` | — | Directory of parquet files (batch mode, overrides `--parquet`) |
| `--robot-usd` | auto-detect | Path to robot USD file |
| `--eval-config` | `configs/eval_config.yaml` | Evaluation thresholds and metric weights |
| `--joint-map` | `configs/joint_maps/g1_inspire.yaml` | Parquet index ↔ Isaac Sim joint mapping |
| `--task-config` | `configs/pick_place.yaml` | Task success criteria |
| `--output-video` | `results/replay.mp4` | Video output path (single mode) |
| `--output-report` | `results/eval_report.json` | JSON report path (single) or batch summary |
| `--output-dir` | `results/batch` | Output directory for batch mode |
| `--fps` | `30` | Video frame rate |
| `--no-video` | `false` | Skip video recording (faster batch runs) |

## Configuration

All evaluation parameters are loaded from YAML config files (no hardcoded values in the replay script).

### `configs/eval_config.yaml`

Controls physics parameters, detection thresholds, metric weights, and pass/fail threshold.

```yaml
collision:
  force_threshold: 1.0    # Newtons — contact force above this triggers collision flag
joint_limits:
  tolerance: 0.01         # rad — allowed overshoot beyond USD-defined limits
gravity:
  upward_accel_threshold: 1.9  # m/s² — flags unsupported upward object acceleration
metric_weights:
  joint_limit: 0.3
  collision: 0.3
  gravity: 0.2
  task_fidelity: 0.2
pass_threshold: 0.8
```

### `configs/joint_maps/g1_inspire.yaml`

Maps parquet column indices to Isaac Sim joint names. Must match your dataset schema.

### `configs/pick_place.yaml`

Defines task-specific success criteria: reach distance, grasp detection, lift height delta, place target position, and subtask weights.

## Example Results

### Single Trajectory Verdict

```
Episode: episode_000000
┌────────────────┬───────┬──────────────────────────────────┐
│ Metric         │ Score │ Details                          │
├────────────────┼───────┼──────────────────────────────────┤
│ Joint Limits   │ 0.97  │ 2 minor violations (left_elbow)  │
│ Collision      │ 1.00  │ No collisions detected           │
│ Gravity        │ 1.00  │ Consistent with 9.81 m/s²        │
│ Task Fidelity  │ 0.85  │ reach:1.0 grasp:0.8 lift:0.9    │
├────────────────┼───────┼──────────────────────────────────┤
│ Overall        │ 0.95  │ PASS ✓                           │
└────────────────┴───────┴──────────────────────────────────┘
```

### Failure Case: Synthetic Trajectory with Gravity Violation

```
Episode: synthetic_042
Verdict: FAIL ✗
Overall: 0.41
Failures:
  - gravity_violation: object levitates at frames 120-145
  - task_fidelity: grasp subtask failed (fingers open during lift)
```

## Testing

```bash
# Run all tests
conda activate env_isaaclab
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=sim_judge --cov-report=term-missing
```

## Acknowledgments

I gratefully acknowledge the following projects and resources that made this work possible:

**Simulation & Robot Frameworks**
- [NVIDIA Isaac Sim](https://docs.isaacsim.omniverse.nvidia.com/5.1.0/index.html) — Physics simulation platform providing PhysX-based collision detection and articulation support
- [Unitree RL Lab](https://github.com/unitreerobotics/unitree_rl_lab) — RL training framework for Unitree robots on Isaac Lab
- [Unitree Sim IsaacLab](https://github.com/unitreerobotics/unitree_sim_isaaclab) — Isaac Sim integration for Unitree humanoid robots, including G1 29-DOF pick-and-place task configurations

**Synthetic Data & World Models**
- [DreamGen](https://arxiv.org/abs/2505.12705) — Synthetic robot data generation pipeline via video world models, whose DreamGen Bench evaluation framework motivated our physics-grounded validation approach
- [GR00T-Dreams](https://github.com/NVIDIA/GR00T-Dreams) — NVIDIA's synthetic data generation pipeline for robot learning using Cosmos world foundation models

Thank you NVIDIA Isaac Sim, Cosmos, and GR00T teams for providing the simulation and world model foundations, and the Unitree Robotics team for open-sourcing the robot simulation frameworks.

## Contributing

Welcome contributions, feedback, and collaboration — if you're working on synthetic data validation, sim-to-real transfer, or Physical AI data pipelines, feel free to open an issue or reach out.
ehdtmxk12@g.hongik.ac.kr

## Disclaimer

This project is under active development. We assume no responsibility for any physical damage, malfunction, or unintended consequences resulting from deploying trajectories validated by this framework directly on real robots. Users are solely responsible for conducting additional safety checks before real-world deployment.


## License

This work is licensed under [CC BY-NC-ND 4.0](https://creativecommons.org/licenses/by-nc-nd/4.0/).

### Third-Party Licenses

