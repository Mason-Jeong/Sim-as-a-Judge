"""Tests for environment verification module."""

import pytest
import sys
from unittest.mock import patch


class TestCheckDependencies:
    """Tests for dependency checking."""

    def test_all_required_packages_importable(self):
        from sim_judge.verify_env import check_dependencies

        result = check_dependencies()

        assert result.numpy_available is True
        assert result.pandas_available is True
        assert result.yaml_available is True

    def test_reports_missing_package(self):
        from sim_judge.verify_env import check_dependencies, _can_import

        with patch("sim_judge.verify_env._can_import", wraps=_can_import) as mock_import:
            original = _can_import

            def fake_can_import(name: str) -> bool:
                if name == "pyarrow":
                    return False
                return original(name)

            mock_import.side_effect = fake_can_import
            result = check_dependencies()
            assert result.pyarrow_available is False
            assert result.numpy_available is True
            assert "pyarrow" in result.missing()

    def test_result_has_all_fields(self):
        from sim_judge.verify_env import check_dependencies

        result = check_dependencies()

        assert hasattr(result, "numpy_available")
        assert hasattr(result, "pandas_available")
        assert hasattr(result, "pyarrow_available")
        assert hasattr(result, "yaml_available")
        assert hasattr(result, "cv2_available")
        assert hasattr(result, "datasets_available")

    def test_all_ok_returns_true(self):
        from sim_judge.verify_env import check_dependencies

        result = check_dependencies()

        # In env_isaaclab, at minimum numpy/pandas/yaml/cv2 should work
        assert result.numpy_available is True
        assert result.pandas_available is True
        assert result.yaml_available is True


class TestCheckDataFiles:
    """Tests for data file existence checking."""

    def test_finds_existing_parquet(self, tmp_path):
        from sim_judge.verify_env import check_data_files

        parquet = tmp_path / "episode_000001.parquet"
        parquet.write_bytes(b"fake")

        result = check_data_files(str(tmp_path))
        assert result.parquet_count >= 1

    def test_reports_zero_for_empty_dir(self, tmp_path):
        from sim_judge.verify_env import check_data_files

        result = check_data_files(str(tmp_path))
        assert result.parquet_count == 0

    def test_reports_missing_directory(self):
        from sim_judge.verify_env import check_data_files

        result = check_data_files("/nonexistent/data/path")
        assert result.parquet_count == 0
        assert result.directory_exists is False
