"""Batch evaluation utilities — collect and resolve parquet file inputs."""

from __future__ import annotations
from pathlib import Path


def collect_parquet_files(directory: str) -> list[Path]:
    """Recursively find all .parquet files in a directory, sorted by name.

    Args:
        directory: Path to search for parquet files.

    Returns:
        Sorted list of Path objects for each .parquet file found.

    Raises:
        FileNotFoundError: If the directory does not exist.
    """
    p = Path(directory)
    if not p.exists():
        raise FileNotFoundError(f"Directory not found: {directory}")

    files = sorted(p.rglob("*.parquet"), key=lambda f: f.name)
    return files


def resolve_parquet_inputs(
    parquet: str | None,
    parquet_dir: str | None,
) -> list[Path]:
    """Resolve CLI inputs into a list of parquet file paths.

    --parquet-dir takes precedence over --parquet when both are provided.

    Args:
        parquet: Single parquet file path (from --parquet).
        parquet_dir: Directory of parquet files (from --parquet-dir).

    Returns:
        List of Path objects to evaluate.

    Raises:
        ValueError: If neither parquet nor parquet_dir is provided.
        FileNotFoundError: If the specified file or directory doesn't exist.
    """
    if parquet_dir is not None:
        return collect_parquet_files(parquet_dir)

    if parquet is not None:
        p = Path(parquet)
        if not p.exists():
            raise FileNotFoundError(f"Parquet file not found: {parquet}")
        return [p]

    raise ValueError(
        "Either --parquet or --parquet-dir must be provided."
    )


def episode_output_paths(
    parquet_path: Path,
    output_dir: str,
) -> dict[str, Path]:
    """Generate per-episode output file paths.

    Creates a subdirectory named after the parquet file stem.

    Args:
        parquet_path: Path to the episode parquet file.
        output_dir: Base output directory.

    Returns:
        Dict with 'video' and 'report' Path values.
    """
    episode_name = parquet_path.stem
    episode_dir = Path(output_dir) / episode_name
    return {
        "video": episode_dir / "replay.mp4",
        "report": episode_dir / "eval_report.json",
    }
