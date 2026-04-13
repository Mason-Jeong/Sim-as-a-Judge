"""Tests for batch evaluation — collecting and processing multiple parquet files."""

import pytest
from pathlib import Path


PROJECT_ROOT = Path(__file__).parent.parent


class TestCollectParquetFiles:
    """Tests for collecting parquet files from a directory."""

    def test_finds_all_parquet_in_flat_dir(self, tmp_path):
        from sim_judge.batch import collect_parquet_files

        for i in range(5):
            (tmp_path / f"episode_{i:06d}.parquet").write_bytes(b"fake")

        files = collect_parquet_files(str(tmp_path))
        assert len(files) == 5

    def test_finds_parquet_in_nested_dirs(self, tmp_path):
        from sim_judge.batch import collect_parquet_files

        chunk_dir = tmp_path / "chunk-000"
        chunk_dir.mkdir()
        for i in range(3):
            (chunk_dir / f"episode_{i:06d}.parquet").write_bytes(b"fake")

        files = collect_parquet_files(str(tmp_path))
        assert len(files) == 3

    def test_returns_sorted_by_name(self, tmp_path):
        from sim_judge.batch import collect_parquet_files

        (tmp_path / "episode_000003.parquet").write_bytes(b"fake")
        (tmp_path / "episode_000001.parquet").write_bytes(b"fake")
        (tmp_path / "episode_000002.parquet").write_bytes(b"fake")

        files = collect_parquet_files(str(tmp_path))
        names = [f.name for f in files]
        assert names == [
            "episode_000001.parquet",
            "episode_000002.parquet",
            "episode_000003.parquet",
        ]

    def test_returns_empty_for_no_parquets(self, tmp_path):
        from sim_judge.batch import collect_parquet_files

        (tmp_path / "readme.txt").write_text("not a parquet")

        files = collect_parquet_files(str(tmp_path))
        assert files == []

    def test_raises_on_nonexistent_dir(self):
        from sim_judge.batch import collect_parquet_files

        with pytest.raises(FileNotFoundError):
            collect_parquet_files("/nonexistent/directory")

    def test_ignores_non_parquet_files(self, tmp_path):
        from sim_judge.batch import collect_parquet_files

        (tmp_path / "episode_000001.parquet").write_bytes(b"fake")
        (tmp_path / "metadata.json").write_text("{}")
        (tmp_path / "info.csv").write_text("a,b")

        files = collect_parquet_files(str(tmp_path))
        assert len(files) == 1


class TestResolveParquetInputs:
    """Tests for resolving --parquet vs --parquet-dir into a file list."""

    def test_single_parquet_returns_one(self, tmp_path):
        from sim_judge.batch import resolve_parquet_inputs

        pq = tmp_path / "episode_000001.parquet"
        pq.write_bytes(b"fake")

        files = resolve_parquet_inputs(parquet=str(pq), parquet_dir=None)
        assert len(files) == 1
        assert files[0] == pq

    def test_parquet_dir_returns_all(self, tmp_path):
        from sim_judge.batch import resolve_parquet_inputs

        for i in range(4):
            (tmp_path / f"episode_{i:06d}.parquet").write_bytes(b"fake")

        files = resolve_parquet_inputs(parquet=None, parquet_dir=str(tmp_path))
        assert len(files) == 4

    def test_parquet_dir_takes_precedence(self, tmp_path):
        from sim_judge.batch import resolve_parquet_inputs

        pq = tmp_path / "single.parquet"
        pq.write_bytes(b"fake")

        batch_dir = tmp_path / "batch"
        batch_dir.mkdir()
        for i in range(3):
            (batch_dir / f"episode_{i:06d}.parquet").write_bytes(b"fake")

        files = resolve_parquet_inputs(
            parquet=str(pq), parquet_dir=str(batch_dir)
        )
        assert len(files) == 3

    def test_raises_when_neither_provided(self):
        from sim_judge.batch import resolve_parquet_inputs

        with pytest.raises(ValueError, match="parquet.*parquet-dir"):
            resolve_parquet_inputs(parquet=None, parquet_dir=None)

    def test_raises_when_single_file_missing(self):
        from sim_judge.batch import resolve_parquet_inputs

        with pytest.raises(FileNotFoundError):
            resolve_parquet_inputs(
                parquet="/nonexistent/file.parquet", parquet_dir=None
            )


class TestBatchOutputPaths:
    """Tests for generating per-episode output paths."""

    def test_generates_video_path_per_episode(self):
        from sim_judge.batch import episode_output_paths

        paths = episode_output_paths(
            parquet_path=Path("/data/episode_000042.parquet"),
            output_dir="results/batch",
        )
        assert paths["video"] == Path("results/batch/episode_000042/replay.mp4")
        assert paths["report"] == Path("results/batch/episode_000042/eval_report.json")

    def test_uses_parquet_stem_as_subdirectory(self):
        from sim_judge.batch import episode_output_paths

        paths = episode_output_paths(
            parquet_path=Path("/data/chunk-000/episode_000123.parquet"),
            output_dir="results",
        )
        assert "episode_000123" in str(paths["video"])


class TestReplayHasParquetDirArg:
    """Verify replay_and_evaluate.py accepts --parquet-dir."""

    def test_has_parquet_dir_arg(self):
        replay_src = (PROJECT_ROOT / "replay_and_evaluate.py").read_text()
        assert "--parquet-dir" in replay_src

    def test_has_output_dir_arg(self):
        replay_src = (PROJECT_ROOT / "replay_and_evaluate.py").read_text()
        assert "--output-dir" in replay_src
