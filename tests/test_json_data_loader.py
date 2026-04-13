"""Tests for JSON episode data loader (episode_0531 format)."""

import pytest
import json
import numpy as np
from pathlib import Path


PROJECT_ROOT = Path(__file__).parent.parent
EPISODE_0531 = PROJECT_ROOT / "data" / "episode_0531"


@pytest.fixture
def sample_data_json(tmp_path):
    """Create a minimal data.json for testing."""
    data = {
        "info": {
            "version": "1.0.0",
            "date": "2025-10-27",
            "author": "unitree",
            "image": {"width": 640, "height": 480, "fps": 60.0},
            "joint_names": {
                "left_arm": ["kLeftShoulderPitch", "kLeftShoulderRoll",
                             "kLeftShoulderYaw", "kLeftElbow",
                             "kLeftWristRoll", "kLeftWristPitch", "kLeftWristyaw"],
                "left_ee": [],
                "right_arm": [],
                "right_ee": [],
                "body": [],
            },
            "tactile_names": {"left_ee": [], "right_ee": []},
            "sim_state": "",
        },
        "text": {
            "goal": "pick up the apple and place it in the blue basket.",
            "desc": "",
            "steps": "",
        },
        "data": [
            {
                "idx": 0,
                "colors": {"color_0": "colors/000000_color_0.jpg"},
                "states": {
                    "left_arm": {"qpos": [-0.476, 0.838, -0.197, 0.525, 0.963, -1.021, 0.839], "qvel": [], "torque": []},
                    "right_arm": {"qpos": [-0.279, -0.626, 0.062, 0.295, -0.680, -0.943, -0.235], "qvel": [], "torque": []},
                    "left_ee": {"qpos": [0.55, 0.60, 0.65, 0.73, 0.66, 0.75], "qvel": [], "torque": []},
                    "right_ee": {"qpos": [0.70, 0.80, 0.81, 0.80, 0.76, 0.64], "qvel": [], "torque": []},
                    "body": {"qpos": []},
                },
                "actions": {
                    "left_arm": {"qpos": [-0.493, 0.855, -0.178, 0.529, 0.931, -1.033, 0.833], "qvel": [], "torque": []},
                    "right_arm": {"qpos": [-0.305, -0.619, 0.062, 0.283, -0.657, -0.935, -0.231], "qvel": [], "torque": []},
                    "left_ee": {"qpos": [0.553, 0.600, 0.653, 0.726, 0.658, 0.758], "qvel": [], "torque": []},
                    "right_ee": {"qpos": [0.701, 0.804, 0.808, 0.797, 0.759, 0.652], "qvel": [], "torque": []},
                    "body": {"qpos": []},
                },
                "tactiles": {"left_ee": "tactiles/tactile_000000_left_ee.npy", "right_ee": "tactiles/tactile_000000_right_ee.npy"},
                "audios": None,
                "sim_state": None,
            },
            {
                "idx": 1,
                "colors": {"color_0": "colors/000001_color_0.jpg"},
                "states": {
                    "left_arm": {"qpos": [-0.500, 0.840, -0.200, 0.530, 0.970, -1.030, 0.850], "qvel": [], "torque": []},
                    "right_arm": {"qpos": [-0.290, -0.630, 0.070, 0.300, -0.690, -0.950, -0.240], "qvel": [], "torque": []},
                    "left_ee": {"qpos": [0.56, 0.61, 0.66, 0.74, 0.67, 0.76], "qvel": [], "torque": []},
                    "right_ee": {"qpos": [0.71, 0.81, 0.82, 0.81, 0.77, 0.65], "qvel": [], "torque": []},
                    "body": {"qpos": []},
                },
                "actions": {
                    "left_arm": {"qpos": [-0.510, 0.850, -0.210, 0.540, 0.960, -1.040, 0.860], "qvel": [], "torque": []},
                    "right_arm": {"qpos": [-0.300, -0.640, 0.080, 0.310, -0.700, -0.960, -0.250], "qvel": [], "torque": []},
                    "left_ee": {"qpos": [0.57, 0.62, 0.67, 0.75, 0.68, 0.77], "qvel": [], "torque": []},
                    "right_ee": {"qpos": [0.72, 0.82, 0.83, 0.82, 0.78, 0.66], "qvel": [], "torque": []},
                    "body": {"qpos": []},
                },
                "tactiles": {"left_ee": "", "right_ee": ""},
                "audios": None,
                "sim_state": None,
            },
        ],
    }
    f = tmp_path / "data.json"
    f.write_text(json.dumps(data))
    return tmp_path  # return episode dir


class TestLoadEpisodeJson:
    """Tests for loading data.json episode format."""

    def test_loads_task_goal(self, sample_data_json):
        from sim_judge.json_data_loader import load_episode_json

        episode = load_episode_json(str(sample_data_json))
        assert episode.task_goal == "pick up the apple and place it in the blue basket."

    def test_loads_frame_count(self, sample_data_json):
        from sim_judge.json_data_loader import load_episode_json

        episode = load_episode_json(str(sample_data_json))
        assert episode.num_frames == 2

    def test_loads_fps(self, sample_data_json):
        from sim_judge.json_data_loader import load_episode_json

        episode = load_episode_json(str(sample_data_json))
        assert episode.fps == 60.0

    def test_raises_on_missing_dir(self):
        from sim_judge.json_data_loader import load_episode_json

        with pytest.raises(FileNotFoundError):
            load_episode_json("/nonexistent/episode")

    def test_raises_on_missing_data_json(self, tmp_path):
        from sim_judge.json_data_loader import load_episode_json

        with pytest.raises(FileNotFoundError, match="data.json"):
            load_episode_json(str(tmp_path))


class TestGetFrameState:
    """Tests for extracting per-frame observation state."""

    def test_frame_returns_left_arm_qpos(self, sample_data_json):
        from sim_judge.json_data_loader import load_episode_json

        episode = load_episode_json(str(sample_data_json))
        state = episode.get_frame_state(0)

        assert len(state.left_arm) == 7
        assert abs(state.left_arm[0] - (-0.476)) < 0.001

    def test_frame_returns_right_arm_qpos(self, sample_data_json):
        from sim_judge.json_data_loader import load_episode_json

        episode = load_episode_json(str(sample_data_json))
        state = episode.get_frame_state(0)

        assert len(state.right_arm) == 7
        assert abs(state.right_arm[0] - (-0.279)) < 0.001

    def test_frame_returns_ee_qpos(self, sample_data_json):
        from sim_judge.json_data_loader import load_episode_json

        episode = load_episode_json(str(sample_data_json))
        state = episode.get_frame_state(0)

        assert len(state.left_ee) == 6
        assert len(state.right_ee) == 6
        assert abs(state.left_ee[0] - 0.55) < 0.001

    def test_assembles_obs_state_vector(self, sample_data_json):
        from sim_judge.json_data_loader import load_episode_json

        episode = load_episode_json(str(sample_data_json))
        obs = episode.get_obs_state(0)

        # left_arm(7) + right_arm(7) + left_ee(6) + right_ee(6) = 26
        assert len(obs) == 26
        assert isinstance(obs, np.ndarray)
        assert obs.dtype == np.float32

    def test_second_frame_differs(self, sample_data_json):
        from sim_judge.json_data_loader import load_episode_json

        episode = load_episode_json(str(sample_data_json))
        obs0 = episode.get_obs_state(0)
        obs1 = episode.get_obs_state(1)

        assert not np.array_equal(obs0, obs1)


class TestDetectDataFormat:
    """Tests for auto-detecting parquet vs JSON episode format."""

    def test_detects_json_episode_dir(self, sample_data_json):
        from sim_judge.json_data_loader import detect_data_format

        fmt = detect_data_format(str(sample_data_json))
        assert fmt == "json_episode"

    def test_detects_parquet_file(self, tmp_path):
        from sim_judge.json_data_loader import detect_data_format
        import pandas as pd

        pq = tmp_path / "episode.parquet"
        pd.DataFrame({"x": [1]}).to_parquet(str(pq))

        fmt = detect_data_format(str(pq))
        assert fmt == "parquet"

    def test_detects_parquet_dir(self, tmp_path):
        from sim_judge.json_data_loader import detect_data_format

        (tmp_path / "episode_000001.parquet").write_bytes(b"fake")

        fmt = detect_data_format(str(tmp_path))
        assert fmt == "parquet_dir"

    def test_returns_unknown_for_empty_dir(self, tmp_path):
        from sim_judge.json_data_loader import detect_data_format

        fmt = detect_data_format(str(tmp_path))
        assert fmt == "unknown"


class TestLoadRealEpisode0531:
    """Integration tests with the actual episode_0531 data."""

    @pytest.mark.skipif(not EPISODE_0531.exists(), reason="episode_0531 not available")
    def test_loads_real_episode(self):
        from sim_judge.json_data_loader import load_episode_json

        episode = load_episode_json(str(EPISODE_0531))
        assert episode.task_goal == "pick up the apple and place it in the blue basket."
        assert episode.num_frames == 503
        assert episode.fps == 60.0

    @pytest.mark.skipif(not EPISODE_0531.exists(), reason="episode_0531 not available")
    def test_real_obs_state_shape(self):
        from sim_judge.json_data_loader import load_episode_json

        episode = load_episode_json(str(EPISODE_0531))
        obs = episode.get_obs_state(0)
        assert len(obs) == 26  # 7+7+6+6
