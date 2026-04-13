# sim_judge/report.py
"""Generate JSON reports from SimJudgeVerdict."""

from __future__ import annotations
from .aggregate import SimJudgeVerdict


def save_json(verdicts: list[SimJudgeVerdict], output_path: str):
    """Save evaluation verdicts to a JSON report file."""
    raise NotImplementedError("Implementation not provided in public release.")


def print_verdict(verdict: SimJudgeVerdict, episode_name: str = ""):
    """Print a formatted verdict summary to stdout."""
    raise NotImplementedError("Implementation not provided in public release.")
