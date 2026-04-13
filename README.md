# Sim-as-a-Judge

**A Physics-Grounded Validation Framework for Synthetic Robot Trajectory Data**

[![License](https://img.shields.io/badge/License-CC%20BY--NC--ND%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc-nd/4.0/)
[![Python](https://img.shields.io/badge/Python-3.10+-green.svg)](https://www.python.org/)
[![Isaac Sim](https://img.shields.io/badge/Isaac%20Sim-5.0%2B-76B900.svg)](https://developer.nvidia.com/isaac-sim)

## Notice

This repository provides the **framework interface and configuration** for Sim-as-a-Judge. Core checker implementations are not included in this public release. Code modifications may be required to run on your environment. All experiments were conducted on the Isaac Sim 5.0+ environment with the [Unitree RL Lab](https://github.com/unitreerobotics/unitree_rl_lab) setup.

## Motivation

In the Physical AI era, data augmentation pipelines — from World Foundation Models (e.g., Cosmos) to action extraction — are becoming the primary source of robot training data. But **not all synthetic data is created equal**:

- Trajectories may violate joint limits or physics constraints
- Generated motions may collide with the environment or the robot itself
- Actions may not faithfully execute the intended manipulation task
- Objects may defy gravity or teleport between frames

**How can we tell good data from bad, before it poisons policy learning?**

Sim-as-a-Judge answers this by **replaying trajectories inside a physics simulator** and computing quantitative quality metrics — acting as an automated, scalable quality gate for robot data pipelines.

## Approach

The core idea is simple: **if a trajectory is physically valid, it should survive replay in a physics simulator without anomalies.**

### Pipeline

```
 Real / Synthetic Data          Isaac Sim Replay              Quality Verdict
┌─────────────────────┐     ┌──────────────────────┐     ┌─────────────────────┐
│  LeRobot Parquet    │     │  Load Robot USD      │     │  Per-frame metrics: │
│  or JSON Episode    │───▶│  Auto-detect task     │───▶│  ✓ Joint limits     │
│  - joint positions  │     │  Build matching scene │     │  ✓ Collision        │
│  - gripper states   │     │  Map joints 1:1      │     │  ✓ Gravity          │
│  - timestamps       │     │  Step physics per    │     │  ✓ Task fidelity    │
│                     │     │    frame             │     │                     │
│                     │     │  Record observations │     │  Overall: PASS /FAIL│
└─────────────────────┘     └──────────────────────┘     └─────────────────────┘
```

### Key Features

1. **Isaac Sim Replay Engine** (`replay_and_evaluate.py`)
   - Loads Unitree G1 USD into a config-driven scene (auto-detected from task description)
   - Supports **LeRobot v2 parquet** and **JSON episode** (data.json + colors/) formats
   - Single episode or **batch evaluation** of entire directories
   - Records MP4 video for visual inspection

2. **Physics-Grounded Checkers** (integrated into the replay loop)
   - **Joint Limit Checker** — flags frames where joint positions exceed USD-defined limits
   - **Collision Checker** — detects self-collision and environment collision via PhysX contact reports
   - **Gravity Checker** — validates that unsupported objects follow gravitational acceleration
   - **Task Fidelity Checker** — scores whether the trajectory achieves subtask goals (reach, grasp, lift, place)

3. **Auto Scene Builder** — detects task type from data and builds matching Isaac Sim scene
   - Task detection from `task_index` (parquet) or `text.goal` (JSON)
   - YAML-driven scene configs: objects, table, environment, lighting
   - Supports DynamicCuboid, FixedCuboid, and USD object types

4. **Report Generator** — produces per-trajectory and batch-level JSON quality reports

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
for t, frame in enumerate(trajectory):
    # 1. Apply action
    robot.set_joint_positions(target_positions)
    world.step(render=True)

    # 2. Read physics state (Isaac Sim returns these after each step)
    joint_pos = robot.get_joint_positions()
    object_pos = get_prim_world_translate(object_prim)
    ee_pos = get_prim_world_translate(ee_prim)

    # 3. Each checker scores this frame
    joint_limit_checker.check_frame(t, joint_pos)
    collision_checker.check_frame(t, contact_forces)
    gravity_checker.check_frame(t, object_pos)
    task_fidelity_checker.check_frame(t, ee_pos, object_pos, hand_closed)

# 4. After replay, aggregate per-frame results into verdict
verdict = aggregate(
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
| **Collision** | Self-collision, robot-env, robot-object | PhysX `ContactSensor` | Contact force magnitude + collision frames |
| **Gravity** | Objects don't float or teleport | Object prim world transform | Gravity violation score |
| **Task Fidelity** | EE reaches object, grasps, lifts, places | EE + object world positions | Subtask completion scores (0-1) |

### Supported Data Formats

| Format | Input | Task Detection |
|--------|-------|----------------|
| **LeRobot v2 Parquet** | `--parquet` or `--parquet-dir` | `task_index` column + `tasks.jsonl` |
| **JSON Episode** | `--episode-dir` | `text.goal` field in `data.json` |

### Joint Mapping (Parquet Index to Isaac Sim Joint)

```
Index [0:7]   -> Left arm:  shoulder_pitch/roll/yaw, elbow, wrist_roll/pitch/yaw
Index [7:14]  -> Right arm: shoulder_pitch/roll/yaw, elbow, wrist_roll/pitch/yaw
Index [14:20] -> Left hand:  pinky, ring, middle, index, thumb_pitch, thumb_yaw
Index [20:26] -> Right hand: pinky, ring, middle, index, thumb_pitch, thumb_yaw
```

## Project Structure

```
sim-as-a-judge/
├── README.md
├── requirements.txt                # Python dependencies (non-Isaac Sim)
├── pyproject.toml                  # Package metadata & pip install support
│
├── replay_and_evaluate.py          # Main entry point (interface reference)
│
├── sim_judge/                      # Core validation package
│   ├── __init__.py
│   ├── joint_limit_checker.py      # Per-frame joint limit validation (interface)
│   ├── collision_checker.py        # PhysX contact force analysis (interface)
│   ├── gravity_checker.py          # Object gravity consistency (interface)
│   ├── task_fidelity_checker.py    # Subtask completion scoring (interface)
│   ├── aggregate.py                # Combine checkers into verdict (interface)
│   ├── report.py                   # JSON report generation (interface)
│   ├── scene_builder.py            # Isaac Sim scene construction (interface)
│   ├── scene_config.py             # Scene YAML loader & task detection
│   ├── config_loader.py            # YAML config loading (eval, joint map, task)
│   ├── batch.py                    # Batch parquet collection & path resolution
│   ├── json_data_loader.py         # JSON episode format loader
│   └── verify_env.py               # Environment dependency checker
│
├── configs/
│   ├── eval_config.yaml            # Metric thresholds and weights
│   ├── pick_place.yaml             # Task success criteria
│   ├── joint_maps/
│   │   └── g1_inspire.yaml         # Parquet index to Isaac Sim joint mapping
│   └── scenes/
│       ├── default.yaml            # Default scene (table only)
│       ├── stack_blocks.yaml       # Block stacking task scene
│       └── pick_place_apple.yaml   # Apple pick-and-place task scene
│
├── scripts/
│   ├── setup_env.sh                # One-click conda environment setup
│   └── verify_env.py               # Environment verification CLI
│
├── assets/                         # Robot USD models & configs
│   └── g1_29dof/                   # Unitree G1 29-DOF variants
│       ├── g1-29dof-inspire-base-fix-usd/
│       ├── g1-29dof_wholebody_inspire/
│       ├── g1-29dof-dex1-base-fix-usd/
│       ├── g1-29dof_wholebody_dex1/
│       ├── g1-29dof-dex3-base-fix-usd/
│       ├── g1-29dof_wholebody_dex3/
│       └── h1_2-26dof-inspire-base-fix-usd/
│
├── data/                           # Trajectory data (not included, see below)
│   ├── episodes/chunk-000/         # LeRobot v2 parquet episodes
│   ├── episode_XXXX/              # JSON episode directories
│   └── meta/                       # Dataset metadata
│
└── results/                        # Generated reports & videos (gitignored)
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

### Data Preparation

Trajectory data is not included in this repository. Place your data under `data/`:
- **Parquet format**: `data/episodes/chunk-000/episode_XXXXXX.parquet`
- **JSON format**: `data/episode_XXXX/data.json` (with `colors/` and optional `tactiles/`)
- **Metadata**: `data/meta/tasks.jsonl`, `data/meta/info.json`

Robot USD assets should be placed under `assets/g1_29dof/` with the corresponding `config.yaml`.

## Usage

### Single Episode (Parquet)

```bash
python replay_and_evaluate.py \
    --parquet data/episodes/chunk-000/episode_000001.parquet \
    --robot-usd path/to/your_robot.usd \
    --output-video results/replay.mp4 \
    --output-report results/eval_report.json
```

### Single Episode (JSON)

```bash
python replay_and_evaluate.py \
    --episode-dir data/episode_0531 \
    --output-video results/replay.mp4 \
    --output-report results/eval_report.json
```

### Batch Evaluation

```bash
python replay_and_evaluate.py \
    --parquet-dir data/episodes/chunk-000 \
    --robot-usd path/to/your_robot.usd \
    --output-dir results/batch \
    --output-report results/batch_report.json \
    --no-video
```

### CLI Reference

| Argument | Default | Description |
|----------|---------|-------------|
| `--parquet` | - | Single parquet file to evaluate |
| `--parquet-dir` | - | Directory of parquet files (batch mode) |
| `--episode-dir` | - | JSON episode directory (data.json + colors/) |
| `--robot-usd` | auto-detect | Path to robot USD file |
| `--eval-config` | `configs/eval_config.yaml` | Evaluation thresholds and metric weights |
| `--joint-map` | `configs/joint_maps/g1_inspire.yaml` | Joint index mapping |
| `--task-config` | `configs/pick_place.yaml` | Task success criteria |
| `--scenes-dir` | `configs/scenes` | Scene config directory |
| `--tasks-jsonl` | `data/meta/tasks.jsonl` | Task index to description mapping |
| `--output-video` | `results/replay.mp4` | Video output path (single mode) |
| `--output-report` | `results/eval_report.json` | JSON report path |
| `--output-dir` | `results/batch` | Output directory for batch mode |
| `--fps` | `30` | Video frame rate |
| `--no-video` | `false` | Skip video recording (faster batch runs) |

## Configuration

All evaluation parameters are loaded from YAML config files.

### `configs/eval_config.yaml`

Controls physics parameters, detection thresholds, metric weights, and pass/fail threshold.

```yaml
collision:
  force_threshold: 1.0    # Newtons
joint_limits:
  tolerance: 0.01         # rad
gravity:
  upward_accel_threshold: 1.9  # m/s²
metric_weights:
  joint_limit: 0.3
  collision: 0.3
  gravity: 0.2
  task_fidelity: 0.2
pass_threshold: 0.8
```

### `configs/scenes/*.yaml`

Scene configs define the Isaac Sim environment for each task type. The framework auto-detects the task from the data and loads the matching scene config.

```yaml
task: "pick up the apple and place it in the blue basket."
environment:
  usd: "Environments/Simple_Warehouse/warehouse.usd"
table:
  usd: "Props/PackingTable/packing_table.usd"
  position: [0.30, 0.0, 0.0]
  rotation: [0, 0, -90]
objects:
  - name: apple
    type: FixedCuboid
    position: [0.117, 0.242, 0.76]
    color: [0.85, 0.1, 0.1]
    size: [0.025, 0.025, 0.025]
target_object: "/World/Apple"
```

### Adding a New Task Scene

1. Add task entry to `data/meta/tasks.jsonl`
2. Create `configs/scenes/<task_name>.yaml` with objects, positions, and physics settings
3. Run — the framework auto-detects the task and builds the matching scene

## Example Results

### Single Trajectory Verdict

```
Episode: episode_000001
Metric            Score  Details
-------------------------------------------------------
Joint Limits       1.00  0 violation frames
Collision          1.00  0 collision frames
Gravity            1.00  0 violation frames
Task Fidelity      0.53  reach:1.0 grasp:1.0 lift:0.0 place:0.13
-------------------------------------------------------
Overall            0.35  FAIL
```

### Failure Case: Synthetic Trajectory with Gravity Violation

```
Episode: synthetic_042
Verdict: FAIL
Overall: 0.41
Failures:
  - gravity_violation: object levitates at frames 120-145
  - task_fidelity: grasp subtask failed (fingers open during lift)
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

## Disclaimer

This project is under active development. We assume no responsibility for any physical damage, malfunction, or unintended consequences resulting from deploying trajectories validated by this framework directly on real robots. Users are solely responsible for conducting additional safety checks before real-world deployment.

## License

This work is licensed under [CC BY-NC-ND 4.0](https://creativecommons.org/licenses/by-nc-nd/4.0/).

### Third-Party Licenses
