"""Integration tests for replay_and_evaluate config loading.

These tests verify that the replay script correctly uses config files
instead of hardcoded values. Isaac Sim-dependent tests are excluded.
"""

import pytest
from pathlib import Path


PROJECT_ROOT = Path(__file__).parent.parent


class TestPipApiRemoved:
    """Verify that runtime pip install calls are removed."""

    def test_no_pipapi_install_in_replay(self):
        replay_src = (PROJECT_ROOT / "replay_and_evaluate.py").read_text()
        assert "pipapi.install" not in replay_src, (
            "omni.kit.pipapi.install() calls should be removed. "
            "Dependencies must be pre-installed via requirements.txt."
        )


class TestConfigFilesExist:
    """Verify required config files are present."""

    def test_eval_config_exists(self):
        assert (PROJECT_ROOT / "configs" / "eval_config.yaml").exists()

    def test_joint_map_exists(self):
        assert (PROJECT_ROOT / "configs" / "joint_maps" / "g1_inspire.yaml").exists()

    def test_pick_place_config_exists(self):
        assert (PROJECT_ROOT / "configs" / "pick_place.yaml").exists()

    def test_requirements_txt_exists(self):
        assert (PROJECT_ROOT / "requirements.txt").exists()


class TestConfigLoadFromProject:
    """Load actual project config files and verify values."""

    def test_load_project_eval_config(self):
        from sim_judge.config_loader import load_eval_config

        config = load_eval_config(
            str(PROJECT_ROOT / "configs" / "eval_config.yaml")
        )
        assert config.physics_dt == 0.001
        assert config.collision_force_threshold == 1.0

    def test_load_project_joint_map(self):
        from sim_judge.config_loader import load_joint_map

        joint_map = load_joint_map(
            str(PROJECT_ROOT / "configs" / "joint_maps" / "g1_inspire.yaml")
        )
        assert joint_map.dof == 26
        assert len(joint_map.mapping) == 26
        assert joint_map.joint_names()[0] == "left_shoulder_pitch_joint"

    def test_load_project_task_config(self):
        from sim_judge.config_loader import load_task_config

        task = load_task_config(
            str(PROJECT_ROOT / "configs" / "pick_place.yaml")
        )
        assert task.task == "pick_place"


class TestReplayUsesConfigArgs:
    """Verify replay_and_evaluate.py has config CLI arguments."""

    def test_has_eval_config_arg(self):
        replay_src = (PROJECT_ROOT / "replay_and_evaluate.py").read_text()
        assert "--eval-config" in replay_src, (
            "replay_and_evaluate.py should accept --eval-config argument"
        )

    def test_has_joint_map_arg(self):
        replay_src = (PROJECT_ROOT / "replay_and_evaluate.py").read_text()
        assert "--joint-map" in replay_src, (
            "replay_and_evaluate.py should accept --joint-map argument"
        )

    def test_has_task_config_arg(self):
        replay_src = (PROJECT_ROOT / "replay_and_evaluate.py").read_text()
        assert "--task-config" in replay_src, (
            "replay_and_evaluate.py should accept --task-config argument"
        )
