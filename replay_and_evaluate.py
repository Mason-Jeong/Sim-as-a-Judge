"""
replay_and_evaluate.py

Isaac Sim replay engine with integrated Sim-as-a-Judge evaluation.
Loads robot USD, parses LeRobot parquet or JSON episode data,
replays trajectories with physics stepping, and runs per-frame quality checkers.

Usage (single episode — parquet):
    python replay_and_evaluate.py \\
        --parquet data/episodes/chunk-000/episode_000001.parquet \\
        --robot-usd assets/g1_29dof/g1-29dof-inspire-base-fix-usd/g1_29dof_with_inspire_rev_1_0.usd \\
        --output-video results/replay.mp4 \\
        --output-report results/eval_report.json

Usage (single episode — JSON):
    python replay_and_evaluate.py \\
        --episode-dir data/episode_0531 \\
        --output-video results/replay.mp4 \\
        --output-report results/eval_report.json

Usage (batch — all parquet files in a directory):
    python replay_and_evaluate.py \\
        --parquet-dir data/episodes/chunk-000 \\
        --robot-usd assets/g1_29dof/g1-29dof-inspire-base-fix-usd/g1_29dof_with_inspire_rev_1_0.usd \\
        --output-dir results/batch \\
        --output-report results/batch_report.json \\
        --no-video

Supported CLI arguments:
    --parquet           Single parquet file to evaluate
    --parquet-dir       Directory of parquet files (batch mode)
    --episode-dir       JSON episode directory (data.json + colors/ format)
    --robot-usd         Robot USD path (auto-detected if not set)
    --eval-config       Path to eval_config.yaml
    --joint-map         Path to joint map YAML
    --task-config       Path to task config YAML
    --scenes-dir        Directory of scene config YAMLs
    --tasks-jsonl       Path to tasks.jsonl for task detection
    --output-video      Video output path (single mode)
    --output-report     JSON report path
    --output-dir        Output directory for batch mode
    --fps               Video frame rate (default: 30)
    --no-video          Skip video recording

This file is the main entry point. The core implementation is not
provided in this public release. See README.md for the framework
overview and evaluation methodology.
"""

raise NotImplementedError(
    "The replay and evaluation pipeline implementation is not provided "
    "in this public release. This file serves as the interface reference. "
    "See README.md for the framework overview."
)
