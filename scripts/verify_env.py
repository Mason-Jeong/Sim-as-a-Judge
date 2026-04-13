#!/usr/bin/env python3
"""Verify that the Sim-as-a-Judge environment is correctly configured.

Usage:
    python scripts/verify_env.py
"""

import sys
from pathlib import Path

# Add project root to path so sim_judge is importable
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from sim_judge.verify_env import check_dependencies, check_data_files


def main() -> int:
    print("=" * 50)
    print(" Sim-as-a-Judge Environment Verification")
    print("=" * 50)
    print()

    # Check Python version
    py_ver = sys.version_info
    print(f"Python: {py_ver.major}.{py_ver.minor}.{py_ver.micro}")
    if py_ver < (3, 10):
        print("  [FAIL] Python 3.10+ required")
        return 1
    print("  [OK]")

    # Check dependencies
    print()
    print("Dependencies:")
    deps = check_dependencies()

    checks = [
        ("numpy", deps.numpy_available),
        ("pandas", deps.pandas_available),
        ("pyarrow", deps.pyarrow_available),
        ("PyYAML", deps.yaml_available),
        ("opencv-python-headless", deps.cv2_available),
        ("datasets", deps.datasets_available),
    ]

    all_ok = True
    for name, available in checks:
        status = "[OK]" if available else "[MISSING]"
        print(f"  {name:<25} {status}")
        if not available:
            all_ok = False

    # Check Isaac Sim (optional — may not be importable outside container)
    print()
    print("Isaac Sim:")
    try:
        import isaacsim
        print(f"  isaacsim                    [OK] ({isaacsim.__version__})")
    except (ImportError, AttributeError):
        try:
            import importlib
            importlib.import_module("isaacsim")
            print("  isaacsim                    [OK]")
        except ImportError:
            print("  isaacsim                    [NOT FOUND]")
            print("  (Required for replay — install Isaac Sim in this env)")

    # Check data files
    print()
    print("Data files:")
    data_dir = project_root / "data" / "data" / "chunk-000"
    data_result = check_data_files(str(data_dir))
    if data_result.directory_exists:
        print(f"  Parquet files: {data_result.parquet_count}")
        if data_result.parquet_count == 0:
            print("  [WARN] No parquet files found in data/data/chunk-000/")
    else:
        print(f"  [WARN] Data directory not found: {data_dir}")

    # Check config files
    print()
    print("Config files:")
    configs = [
        "configs/eval_config.yaml",
        "configs/joint_maps/g1_inspire.yaml",
        "configs/pick_place.yaml",
    ]
    for cfg in configs:
        exists = (project_root / cfg).exists()
        status = "[OK]" if exists else "[MISSING]"
        print(f"  {cfg:<40} {status}")

    # Check robot USD
    print()
    print("Robot assets:")
    robot_variants = [
        ("G1 Inspire (wholebody)", "assets/g1_29dof/g1-29dof_wholebody_inspire/g1_29dof_with_inspire_rev_1_0.usd"),
        ("G1 Inspire (base-fix)", "assets/g1_29dof/g1-29dof-inspire-base-fix-usd/g1_29dof_with_inspire_rev_1_0.usd"),
        ("G1 Dex1 (wholebody)", "assets/g1_29dof/g1-29dof_wholebody_dex1/g1_29dof_with_dex1_rev_1_0.usd"),
        ("G1 Dex3 (wholebody)", "assets/g1_29dof/g1-29dof_wholebody_dex3/g1_29dof_with_dex3_rev_1_0.usd"),
    ]
    any_robot_found = False
    for name, rel_path in robot_variants:
        usd_path = project_root / rel_path
        if usd_path.exists():
            size_mb = usd_path.stat().st_size / (1024 * 1024)
            print(f"  {name:<30} [OK] ({size_mb:.0f}MB)")
            any_robot_found = True
        else:
            print(f"  {name:<30} [NOT FOUND]")
    if not any_robot_found:
        print("  [WARN] No robot USD assets found in assets/g1_29dof/")

    print()
    if all_ok:
        print("Result: ALL CHECKS PASSED")
        return 0
    else:
        missing = deps.missing()
        print(f"Result: MISSING PACKAGES: {', '.join(missing)}")
        print(f"  Fix: pip install {' '.join(missing)}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
