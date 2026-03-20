# sim_judge/report.py
"""Generate JSON and HTML reports from SimJudgeVerdict."""

from __future__ import annotations
import json
from pathlib import Path
from .aggregate import SimJudgeVerdict


def save_json(verdicts: list[SimJudgeVerdict], output_path: str):
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    pass_count = sum(1 for v in verdicts if v.passed)
    total = len(verdicts)

    report = {
        "total_trajectories": total,
        "passed": pass_count,
        "failed": total - pass_count,
        "pass_rate": round(pass_count / max(total, 1), 4),
        "avg_overall_score": round(sum(v.overall_score for v in verdicts) / max(total, 1), 4),
        "per_trajectory": [v.to_dict() for v in verdicts],
    }

    with open(output_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"[SimJudge] JSON report saved: {output_path}")
    return report


def print_verdict(verdict: SimJudgeVerdict, episode_name: str = ""):
    print(f"\n{'=' * 55}")
    if episode_name:
        print(f"Episode: {episode_name}")
    print(verdict.summary())
    print(f"{'=' * 55}\n")
