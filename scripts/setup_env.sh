#!/usr/bin/env bash
# Setup script for Sim-as-a-Judge conda environment.
# Installs Python dependencies into the existing env_isaaclab conda environment.
#
# Usage:
#   bash scripts/setup_env.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
CONDA_ENV="env_isaaclab"

echo "============================================"
echo " Sim-as-a-Judge Environment Setup"
echo "============================================"
echo ""

# Check conda is available
if ! command -v conda &>/dev/null; then
    echo "[ERROR] conda not found. Install Miniconda/Anaconda first."
    exit 1
fi

# Check env_isaaclab exists
if ! conda env list | grep -q "$CONDA_ENV"; then
    echo "[ERROR] Conda environment '$CONDA_ENV' not found."
    echo "        Create it first with Isaac Sim installed."
    exit 1
fi

echo "[1/3] Installing Python dependencies into '$CONDA_ENV'..."
conda run -n "$CONDA_ENV" pip install -r "$PROJECT_DIR/requirements.txt"

echo ""
echo "[2/3] Installing sim_judge package (editable)..."
conda run -n "$CONDA_ENV" pip install -e "$PROJECT_DIR"

echo ""
echo "[3/3] Verifying environment..."
conda run -n "$CONDA_ENV" python "$PROJECT_DIR/scripts/verify_env.py"

echo ""
echo "============================================"
echo " Setup complete!"
echo " Activate: conda activate $CONDA_ENV"
echo " Run:      python replay_and_evaluate.py --help"
echo "============================================"
