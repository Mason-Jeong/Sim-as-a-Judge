"""Tests for config loader module — loads YAML configs for Sim-as-a-Judge."""

import pytest
from pathlib import Path


@pytest.fixture
def sample_eval_config(tmp_path):
    """Create a sample eval_config.yaml for testing."""
    config_content = """\
physics:
  dt: 0.001
  gravity: [0, 0, -9.81]

collision:
  force_threshold: 2.0

joint_limits:
  tolerance: 0.05

gravity:
  upward_accel_threshold: 4.9

task_fidelity:
  reach_threshold: 0.10
  grasp_threshold: 0.08
  lift_min_delta: 0.05
  place_threshold: 0.10

metric_weights:
  joint_limit: 0.3
  collision: 0.3
  gravity: 0.2
  task_fidelity: 0.2

pass_threshold: 0.8
"""
    config_file = tmp_path / "eval_config.yaml"
    config_file.write_text(config_content)
    return config_file


@pytest.fixture
def sample_joint_map(tmp_path):
    """Create a sample joint map YAML for testing."""
    config_content = """\
robot: unitree_g1_29dof_inspire
dof: 26
mapping:
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
  14: left_pinky_proximal_joint
  15: left_ring_proximal_joint
  16: left_middle_proximal_joint
  17: left_index_proximal_joint
  18: left_thumb_proximal_pitch_joint
  19: left_thumb_proximal_yaw_joint
  20: right_pinky_proximal_joint
  21: right_ring_proximal_joint
  22: right_middle_proximal_joint
  23: right_index_proximal_joint
  24: right_thumb_proximal_pitch_joint
  25: right_thumb_proximal_yaw_joint
"""
    config_file = tmp_path / "g1_inspire.yaml"
    config_file.write_text(config_content)
    return config_file


@pytest.fixture
def sample_task_config(tmp_path):
    """Create a sample task config YAML for testing."""
    config_content = """\
task: pick_place
criteria:
  reach:
    distance_threshold: 0.10
  grasp:
    finger_close_threshold: 0.08
  lift:
    min_height_delta: 0.05
  place:
    target_position: [0.6, -0.3, 0.7]
    distance_threshold: 0.10
subtask_weights:
  reach: 0.2
  grasp: 0.3
  lift: 0.3
  place: 0.2
success_threshold: 0.8
"""
    config_file = tmp_path / "pick_place.yaml"
    config_file.write_text(config_content)
    return config_file


class TestLoadEvalConfig:
    """Tests for loading eval_config.yaml."""

    def test_loads_valid_config(self, sample_eval_config):
        from sim_judge.config_loader import load_eval_config

        config = load_eval_config(str(sample_eval_config))

        assert config.physics_dt == 0.001
        assert config.collision_force_threshold == 2.0
        assert config.joint_limits_tolerance == 0.05
        assert config.gravity_upward_accel_threshold == 4.9
        assert config.pass_threshold == 0.8

    def test_loads_metric_weights(self, sample_eval_config):
        from sim_judge.config_loader import load_eval_config

        config = load_eval_config(str(sample_eval_config))

        assert config.metric_weights["joint_limit"] == 0.3
        assert config.metric_weights["collision"] == 0.3
        assert config.metric_weights["gravity"] == 0.2
        assert config.metric_weights["task_fidelity"] == 0.2

    def test_loads_task_fidelity_thresholds(self, sample_eval_config):
        from sim_judge.config_loader import load_eval_config

        config = load_eval_config(str(sample_eval_config))

        assert config.task_fidelity_reach_threshold == 0.10
        assert config.task_fidelity_grasp_threshold == 0.08
        assert config.task_fidelity_lift_min_delta == 0.05
        assert config.task_fidelity_place_threshold == 0.10

    def test_raises_on_missing_file(self):
        from sim_judge.config_loader import load_eval_config

        with pytest.raises(FileNotFoundError):
            load_eval_config("/nonexistent/path.yaml")

    def test_returns_defaults_for_missing_keys(self, tmp_path):
        from sim_judge.config_loader import load_eval_config

        minimal = tmp_path / "minimal.yaml"
        minimal.write_text("physics:\n  dt: 0.002\n")

        config = load_eval_config(str(minimal))

        assert config.physics_dt == 0.002
        # Missing keys should have sensible defaults
        assert config.collision_force_threshold == 1.0
        assert config.pass_threshold == 0.8


class TestLoadJointMap:
    """Tests for loading joint map YAML."""

    def test_loads_joint_names(self, sample_joint_map):
        from sim_judge.config_loader import load_joint_map

        joint_map = load_joint_map(str(sample_joint_map))

        assert joint_map.robot == "unitree_g1_29dof_inspire"
        assert joint_map.dof == 26
        assert len(joint_map.mapping) == 26

    def test_mapping_indices_are_integers(self, sample_joint_map):
        from sim_judge.config_loader import load_joint_map

        joint_map = load_joint_map(str(sample_joint_map))

        for idx in joint_map.mapping:
            assert isinstance(idx, int)

    def test_mapping_values_are_strings(self, sample_joint_map):
        from sim_judge.config_loader import load_joint_map

        joint_map = load_joint_map(str(sample_joint_map))

        for name in joint_map.mapping.values():
            assert isinstance(name, str)

    def test_joint_names_list(self, sample_joint_map):
        from sim_judge.config_loader import load_joint_map

        joint_map = load_joint_map(str(sample_joint_map))
        names = joint_map.joint_names()

        assert names[0] == "left_shoulder_pitch_joint"
        assert names[7] == "right_shoulder_pitch_joint"
        assert names[14] == "left_pinky_proximal_joint"
        assert len(names) == 26

    def test_raises_on_missing_file(self):
        from sim_judge.config_loader import load_joint_map

        with pytest.raises(FileNotFoundError):
            load_joint_map("/nonexistent/path.yaml")

    def test_raises_on_missing_required_key(self, tmp_path):
        from sim_judge.config_loader import load_joint_map

        bad = tmp_path / "bad.yaml"
        bad.write_text("robot: test\ndof: 2\n")

        with pytest.raises(ValueError, match="missing required key.*mapping"):
            load_joint_map(str(bad))

    def test_raises_on_dof_mismatch(self, tmp_path):
        from sim_judge.config_loader import load_joint_map

        bad = tmp_path / "mismatch.yaml"
        bad.write_text("robot: test\ndof: 3\nmapping:\n  0: joint_a\n  1: joint_b\n")

        with pytest.raises(ValueError, match="dof=3 but mapping has 2"):
            load_joint_map(str(bad))


class TestLoadTaskConfig:
    """Tests for loading task config YAML."""

    def test_loads_task_name(self, sample_task_config):
        from sim_judge.config_loader import load_task_config

        task = load_task_config(str(sample_task_config))

        assert task.task == "pick_place"

    def test_loads_criteria(self, sample_task_config):
        from sim_judge.config_loader import load_task_config

        task = load_task_config(str(sample_task_config))

        assert task.reach_threshold == 0.10
        assert task.grasp_threshold == 0.08
        assert task.lift_min_delta == 0.05
        assert task.place_threshold == 0.10

    def test_loads_target_position(self, sample_task_config):
        from sim_judge.config_loader import load_task_config

        task = load_task_config(str(sample_task_config))

        assert task.place_target_position == [0.6, -0.3, 0.7]

    def test_loads_subtask_weights(self, sample_task_config):
        from sim_judge.config_loader import load_task_config

        task = load_task_config(str(sample_task_config))

        assert task.subtask_weights["reach"] == 0.2
        assert task.subtask_weights["grasp"] == 0.3

    def test_raises_on_missing_file(self):
        from sim_judge.config_loader import load_task_config

        with pytest.raises(FileNotFoundError):
            load_task_config("/nonexistent/path.yaml")

    def test_raises_on_missing_task_key(self, tmp_path):
        from sim_judge.config_loader import load_task_config

        bad = tmp_path / "notask.yaml"
        bad.write_text("criteria:\n  reach:\n    distance_threshold: 0.1\n")

        with pytest.raises(ValueError, match="missing required key.*task"):
            load_task_config(str(bad))

    def test_uses_default_subtask_weights_when_omitted(self, tmp_path):
        from sim_judge.config_loader import load_task_config

        minimal = tmp_path / "minimal.yaml"
        minimal.write_text("task: pick_place\n")

        task = load_task_config(str(minimal))
        assert task.subtask_weights["reach"] == 0.2
        assert task.subtask_weights["grasp"] == 0.3
