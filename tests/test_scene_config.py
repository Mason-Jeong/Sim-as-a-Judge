"""Tests for scene config loader and task registry."""

import pytest
from pathlib import Path


PROJECT_ROOT = Path(__file__).parent.parent


@pytest.fixture
def sample_scene_config(tmp_path):
    content = """\
task: "Stack the red, green, and yellow blocks."
environment:
  usd: "Environments/Warehouse/warehouse.usd"
table:
  usd: "Props/PackingTable/packing_table.usd"
  prim_path: "/World/Table"
  position: [0.5, -0.2, 0.0]
  rotation: [0, 0, -90]
  scale: [1.0, 1.0, 0.7]
objects:
  - name: red_block
    type: DynamicCuboid
    prim_path: "/World/RedBlock"
    position: [0.35, -0.2, 0.75]
    color: [0.8, 0.1, 0.1]
    size: [0.04, 0.04, 0.04]
    mass: 0.1
  - name: green_block
    type: DynamicCuboid
    prim_path: "/World/GreenBlock"
    position: [0.40, -0.3, 0.75]
    color: [0.1, 0.8, 0.1]
    size: [0.04, 0.04, 0.04]
    mass: 0.1
target_object: "/World/RedBlock"
ee_prim_path: "/World/G1/left_wrist_yaw_link"
"""
    f = tmp_path / "stack_blocks.yaml"
    f.write_text(content)
    return f


@pytest.fixture
def default_scene_config(tmp_path):
    content = """\
task: "default"
environment:
  usd: "Environments/Warehouse/warehouse.usd"
table:
  usd: "Props/PackingTable/packing_table.usd"
  prim_path: "/World/Table"
  position: [0.5, -0.2, 0.0]
  rotation: [0, 0, -90]
  scale: [1.0, 1.0, 0.7]
objects: []
target_object: null
ee_prim_path: "/World/G1/left_wrist_yaw_link"
"""
    f = tmp_path / "default.yaml"
    f.write_text(content)
    return f


@pytest.fixture
def tasks_jsonl(tmp_path):
    content = '{"task_index":0,"task":"Stack the red, green, and yellow blocks."}\n'
    content += '{"task_index":1,"task":"Pick up the apple and place it in the pan."}\n'
    f = tmp_path / "tasks.jsonl"
    f.write_text(content)
    return f


@pytest.fixture
def scenes_dir(tmp_path, sample_scene_config, default_scene_config):
    """Directory with scene configs."""
    scenes = tmp_path / "scenes"
    scenes.mkdir()
    (scenes / "stack_blocks.yaml").write_text(sample_scene_config.read_text())
    (scenes / "default.yaml").write_text(default_scene_config.read_text())
    return scenes


class TestLoadSceneConfig:
    """Tests for loading scene config YAML."""

    def test_loads_task_description(self, sample_scene_config):
        from sim_judge.scene_config import load_scene_config

        config = load_scene_config(str(sample_scene_config))
        assert config.task == "Stack the red, green, and yellow blocks."

    def test_loads_environment(self, sample_scene_config):
        from sim_judge.scene_config import load_scene_config

        config = load_scene_config(str(sample_scene_config))
        assert "Warehouse" in config.environment_usd

    def test_loads_table(self, sample_scene_config):
        from sim_judge.scene_config import load_scene_config

        config = load_scene_config(str(sample_scene_config))
        assert config.table is not None
        assert config.table.prim_path == "/World/Table"
        assert config.table.position == [0.5, -0.2, 0.0]

    def test_loads_objects(self, sample_scene_config):
        from sim_judge.scene_config import load_scene_config

        config = load_scene_config(str(sample_scene_config))
        assert len(config.objects) == 2
        assert config.objects[0].name == "red_block"
        assert config.objects[0].obj_type == "DynamicCuboid"
        assert config.objects[0].mass == 0.1

    def test_loads_object_color(self, sample_scene_config):
        from sim_judge.scene_config import load_scene_config

        config = load_scene_config(str(sample_scene_config))
        assert config.objects[0].color == [0.8, 0.1, 0.1]

    def test_loads_target_object(self, sample_scene_config):
        from sim_judge.scene_config import load_scene_config

        config = load_scene_config(str(sample_scene_config))
        assert config.target_object == "/World/RedBlock"

    def test_loads_ee_prim_path(self, sample_scene_config):
        from sim_judge.scene_config import load_scene_config

        config = load_scene_config(str(sample_scene_config))
        assert config.ee_prim_path == "/World/G1/left_wrist_yaw_link"

    def test_empty_objects_list(self, default_scene_config):
        from sim_judge.scene_config import load_scene_config

        config = load_scene_config(str(default_scene_config))
        assert config.objects == []
        assert config.target_object is None

    def test_raises_on_missing_file(self):
        from sim_judge.scene_config import load_scene_config

        with pytest.raises(FileNotFoundError):
            load_scene_config("/nonexistent/path.yaml")


class TestDetectTaskFromParquet:
    """Tests for detecting task type from parquet data."""

    def test_detects_task_index(self, tasks_jsonl):
        from sim_judge.scene_config import load_tasks_jsonl

        tasks = load_tasks_jsonl(str(tasks_jsonl))
        assert tasks[0] == "Stack the red, green, and yellow blocks."
        assert tasks[1] == "Pick up the apple and place it in the pan."

    def test_returns_empty_for_missing_file(self):
        from sim_judge.scene_config import load_tasks_jsonl

        tasks = load_tasks_jsonl("/nonexistent/tasks.jsonl")
        assert tasks == {}

    def test_detect_task_from_parquet(self, tasks_jsonl, tmp_path):
        from sim_judge.scene_config import detect_task_from_parquet
        import pandas as pd

        # Create a minimal parquet with task_index=0
        df = pd.DataFrame({"task_index": [0, 0, 0], "frame_index": [0, 1, 2]})
        pq_path = tmp_path / "episode.parquet"
        df.to_parquet(str(pq_path))

        task_desc = detect_task_from_parquet(str(pq_path), str(tasks_jsonl))
        assert task_desc == "Stack the red, green, and yellow blocks."

    def test_returns_unknown_for_missing_task_index(self, tasks_jsonl, tmp_path):
        from sim_judge.scene_config import detect_task_from_parquet
        import pandas as pd

        df = pd.DataFrame({"task_index": [99], "frame_index": [0]})
        pq_path = tmp_path / "episode.parquet"
        df.to_parquet(str(pq_path))

        task_desc = detect_task_from_parquet(str(pq_path), str(tasks_jsonl))
        assert task_desc == "unknown"


class TestResolveSceneConfig:
    """Tests for resolving task description to scene config."""

    def test_matches_task_to_scene(self, scenes_dir):
        from sim_judge.scene_config import resolve_scene_config

        config = resolve_scene_config(
            "Stack the red, green, and yellow blocks.",
            str(scenes_dir),
        )
        assert config.task == "Stack the red, green, and yellow blocks."
        assert len(config.objects) == 2

    def test_falls_back_to_default(self, scenes_dir):
        from sim_judge.scene_config import resolve_scene_config

        config = resolve_scene_config(
            "Some unknown task description",
            str(scenes_dir),
        )
        assert config.task == "default"

    def test_raises_if_no_default(self, tmp_path):
        from sim_judge.scene_config import resolve_scene_config

        empty_dir = tmp_path / "empty_scenes"
        empty_dir.mkdir()

        with pytest.raises(FileNotFoundError, match="default"):
            resolve_scene_config("anything", str(empty_dir))
