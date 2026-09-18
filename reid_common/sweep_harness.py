"""Shared train->checkpoint->eval->summary-row loop for methods/wicv/run_*.py sweeps."""

from __future__ import annotations

import json
from pathlib import Path

from reid_common.subprocess_utils import run

__all__ = ["run", "resolve_checkpoint", "eval_summary_row"]


def resolve_checkpoint(output_dir: Path) -> Path:
    """Prefer the best-validation checkpoint, falling back to the last epoch."""
    best = output_dir / "model_best.pth"
    if best.exists():
        return best
    return output_dir / "model_last.pth"


def eval_summary_row(
    eval_path: Path,
    per_condition_metrics: list[str] | None = None,
    **extra_fields,
) -> dict:
    """Flatten an evaluate.py eval.json into one summary-table row.

    `extra_fields` (e.g. variant=..., seed=..., protocol=...) are inserted
    first, in the order given, followed by the overall rank1/rank5/mAP, then
    optionally one `{condition}_{metric}` column per requested per-condition
    metric.
    """
    result = json.loads(eval_path.read_text(encoding="utf-8"))
    row = dict(extra_fields)
    row["rank1"] = result["overall"]["rank1"]
    row["rank5"] = result["overall"]["rank5"]
    row["mAP"] = result["overall"]["mAP"]
    if per_condition_metrics:
        for condition, metrics in sorted(result.get("per_condition", {}).items()):
            for metric in per_condition_metrics:
                row[f"{condition}_{metric}"] = metrics[metric]
    return row
