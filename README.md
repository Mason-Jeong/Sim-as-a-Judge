# Sim-as-a-Judge

**A Physics-Grounded Validation Framework for Synthetic Robot Trajectory Data**

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.10+-green.svg)](https://www.python.org/)
[![Isaac Sim](https://img.shields.io/badge/Isaac%20Sim-4.5%2B-76B900.svg)](https://developer.nvidia.com/isaac-sim)

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
   - Loads Unitree G1 29-DOF + Inspire Hand USD into a configured scene (table, objects, environment)
   - Parses LeRobot v2 parquet files and maps 26D actions (14 arm + 12 hand) to Isaac Sim joint indices
   - Replays trajectories frame-by-frame with physics stepping, collecting per-frame diagnostic data
   - Records MP4 video for visual inspection
Playing
![Image](https://github.com/user-attachments/assets/2a0385d8-dada-4cc7-be0a-30b33446aa67)

2. **Physics-Grounded Checkers** (integrated into the replay loop)
   - **Joint Limit Checker** — flags frames where joint positions exceed USD-defined limits
   - **Collision Checker** — detects self-collision and environment collision via PhysX contact reports
   - **Gravity Checker** — validates that unsupported objects follow gravitational acceleration
   - **Task Fidelity Checker** — scores whether the trajectory achieves subtask goals (reach, grasp, lift, place)

3. **Report Generator** — produces per-trajectory and batch-level quality reports (JSON + HTML)

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

## Robot & Data Configuration

### Unitree G1 29-DOF + Inspire Hand

| Property | Value |
|----------|-------|
| Total DOF | 49 (29 body + 20 hand) |
| Controlled DOF | 26 (14 arm + 12 hand) |
| Data format | LeRobot v2 parquet |
| Action dim | 26D absolute joint positions |

### Joint Mapping (Parquet Index → Isaac Sim Joint)

```
Parquet [0:7]   → Left arm:  shoulder_pitch/roll/yaw, elbow, wrist_roll/pitch/yaw
Parquet [7:14]  → Right arm: shoulder_pitch/roll/yaw, elbow, wrist_roll/pitch/yaw
Parquet [14:20] → Left hand:  pinky, ring, middle, index, thumb_pitch, thumb_yaw
Parquet [20:26] → Right hand: pinky, ring, middle, index, thumb_pitch, thumb_yaw
```

## Project Structure

```
sim-as-a-judge/
├── README.md
├── LICENSE
├── requirements.txt
│
├── replay_and_evaluate.py          # ★ Main script: replay + evaluation
│
├── sim_judge/                      # Checker modules
│   ├── __init__.py
│   ├── joint_limit_checker.py      # Per-frame joint limit validation
│   ├── collision_checker.py        # PhysX contact force analysis
│   ├── gravity_checker.py          # Object gravity consistency
│   ├── task_fidelity_checker.py    # Subtask completion scoring
│   ├── aggregate.py                # Combine checkers → verdict
│   └── report.py                   # JSON + HTML report generation
│
├── configs/
│   ├── eval_config.yaml            # Metric thresholds and weights
│   ├── joint_maps/
│   │   └── g1_inspire.yaml         # 26D parquet ↔ Isaac Sim mapping
│   └── tasks/
│       └── pick_place.yaml         # Task success criteria
│
├── data/
│   └── sample/
│       └── episode_000000.parquet  # Sample trajectory
│
├── results/                        # Generated reports (gitignored)
│
└── docs/
    └── figures/
        ├── pipeline_overview.png
        ├── metric_comparison.png
        └── failure_examples.png
```

## Usage

### Replay + Evaluate a Trajectory

```bash
# Inside Isaac Sim container or environment
python replay_and_evaluate.py \
    --parquet data/sample/episode_000000.parquet \
    --robot-usd g1_29dof_with_inspire_rev_1_0.usd \
    --output-video results/replay.mp4 \
    --output-report results/eval_report.json
```

### Batch Evaluation

```bash
python replay_and_evaluate.py \
    --parquet-dir data/episodes/ \
    --robot-usd g1_29dof_with_inspire_rev_1_0.usd \
    --output-report results/batch_report.json
```

### Generate HTML Report

```bash
python -m sim_judge.report \
    --input results/batch_report.json \
    --output results/report.html
```

## Example Results

### Single Trajectory Verdict

```
Episode: episode_000000
Frames:  696
Duration: 6.96s

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

This project is licensed under the Apache License 2.0 — see [LICENSE](LICENSE) for details.
