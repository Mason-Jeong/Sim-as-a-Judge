"""JSON episode data loader for Sim-as-a-Judge.

Supports the episode directory format used by episode_0531 etc:
  episode_dir/
    data.json       — metadata + per-frame states/actions
    colors/         — camera images (XXXXXX_color_N.jpg)
    tactiles/       — tactile sensor data (.npy)
"""

from __future__ import annotations
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class FrameState:
    """Per-frame observation state extracted from data.json."""

    idx: int
    left_arm: list[float]
    right_arm: list[float]
    left_ee: list[float]
    right_ee: list[float]


@dataclass
class EpisodeData:
    """Loaded JSON episode data."""

    episode_dir: Path
    task_goal: str
    fps: float
    _frames: list[dict]

    @property
    def num_frames(self) -> int:
        return len(self._frames)

    def get_frame_state(self, frame_idx: int) -> FrameState:
        """Extract observation state for a single frame."""
        frame = self._frames[frame_idx]
        states = frame["states"]
        return FrameState(
            idx=frame["idx"],
            left_arm=states["left_arm"]["qpos"],
            right_arm=states["right_arm"]["qpos"],
            left_ee=states["left_ee"]["qpos"],
            right_ee=states["right_ee"]["qpos"],
        )

    def get_obs_state(self, frame_idx: int) -> np.ndarray:
        """Assemble full observation vector: left_arm(7) + right_arm(7) + left_ee(6) + right_ee(6) = 26."""
        fs = self.get_frame_state(frame_idx)
        return np.concatenate([
            fs.left_arm, fs.right_arm, fs.left_ee, fs.right_ee,
        ]).astype(np.float32)


def load_episode_json(episode_dir: str) -> EpisodeData:
    """Load a JSON-format episode directory.

    Args:
        episode_dir: Path to episode directory containing data.json.

    Returns:
        EpisodeData with task goal, FPS, and frame data.

    Raises:
        FileNotFoundError: If directory or data.json doesn't exist.
    """
    p = Path(episode_dir)
    if not p.exists():
        raise FileNotFoundError(f"Episode directory not found: {episode_dir}")

    data_json = p / "data.json"
    if not data_json.exists():
        raise FileNotFoundError(f"data.json not found in {episode_dir}")

    with open(data_json) as f:
        raw = json.load(f)

    task_goal = raw["text"]["goal"]
    fps = raw["info"]["image"]["fps"]
    frames = raw["data"]

    return EpisodeData(
        episode_dir=p,
        task_goal=task_goal,
        fps=fps,
        _frames=frames,
    )


def detect_data_format(path: str) -> str:
    """Auto-detect whether a path is parquet file, parquet dir, or JSON episode dir.

    Returns:
        "json_episode" — directory with data.json
        "parquet" — single .parquet file
        "parquet_dir" — directory containing .parquet files
        "unknown" — cannot determine
    """
    p = Path(path)

    if not p.exists():
        return "unknown"

    # Single parquet file
    if p.is_file() and p.suffix == ".parquet":
        return "parquet"

    # Directory
    if p.is_dir():
        # JSON episode format (has data.json)
        if (p / "data.json").exists():
            return "json_episode"

        # Parquet directory (contains .parquet files)
        if any(p.rglob("*.parquet")):
            return "parquet_dir"

    return "unknown"
