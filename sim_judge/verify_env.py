"""Environment verification for Sim-as-a-Judge."""

from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import importlib


@dataclass
class DependencyResult:
    """Result of dependency availability check."""

    numpy_available: bool = False
    pandas_available: bool = False
    pyarrow_available: bool = False
    yaml_available: bool = False
    cv2_available: bool = False
    datasets_available: bool = False

    def all_ok(self) -> bool:
        return all([
            self.numpy_available,
            self.pandas_available,
            self.pyarrow_available,
            self.yaml_available,
            self.cv2_available,
            self.datasets_available,
        ])

    def missing(self) -> list[str]:
        names = {
            "numpy": self.numpy_available,
            "pandas": self.pandas_available,
            "pyarrow": self.pyarrow_available,
            "PyYAML": self.yaml_available,
            "opencv-python-headless": self.cv2_available,
            "datasets": self.datasets_available,
        }
        return [name for name, available in names.items() if not available]


def _can_import(module_name: str) -> bool:
    try:
        importlib.import_module(module_name)
        return True
    except ImportError:
        return False


def check_dependencies() -> DependencyResult:
    """Check that all required Python packages are importable."""
    return DependencyResult(
        numpy_available=_can_import("numpy"),
        pandas_available=_can_import("pandas"),
        pyarrow_available=_can_import("pyarrow"),
        yaml_available=_can_import("yaml"),
        cv2_available=_can_import("cv2"),
        datasets_available=_can_import("datasets"),
    )


@dataclass
class DataFilesResult:
    """Result of data file existence check."""

    directory_exists: bool = False
    parquet_count: int = 0


def check_data_files(data_dir: str) -> DataFilesResult:
    """Check that data directory exists and contains parquet files."""
    p = Path(data_dir)
    if not p.exists():
        return DataFilesResult(directory_exists=False, parquet_count=0)

    parquets = list(p.rglob("*.parquet"))
    return DataFilesResult(
        directory_exists=True,
        parquet_count=len(parquets),
    )
