#!/usr/bin/env python3
"""Plot text token length vs edit distance for cutoff content."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import List, Tuple

import matplotlib.pyplot as plt

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Scatter plot of text_token_len vs edit_dist for cutoff content.")
    parser.add_argument(
        "--metric",
        type=Path,
        default=Path("eval_results/pride_square_cutoff_content/1024/dpsk_eval_metric.json"),
        help="Metrics JSON path.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("eval_results/pride_square_cutoff_content/1024/tokenlen_vs_editdist_cutoff_large.pdf"),
        help="Output plot path.",
    )
    return parser.parse_args()


def load_points(path: Path) -> List[Tuple[float, float]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    points: List[Tuple[float, float]] = []
    for entry in data:
        if not isinstance(entry, dict):
            continue
        token_len = entry.get("text_token_len")
        edit_dist = entry.get("edit_dist")
        if token_len is None or edit_dist is None:
            continue
        try:
            x = float(token_len)
            y = float(edit_dist)
        except (TypeError, ValueError):
            continue
        points.append((x, y))
    return points


def main() -> None:
    args = parse_args()
    points = load_points(args.metric)

    fig, ax = plt.subplots(figsize=(10, 6))
    if points:
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        ax.scatter(xs, ys, color="#D65F5F", alpha=0.6, s=20, marker="o", label="Cutoff")
    ax.set_xlabel("text_token_len", fontsize=14, fontweight="bold")
    ax.set_ylabel("edit_dist", fontsize=14, fontweight="bold")
    for tick in ax.get_xticklabels():
        tick.set_fontsize(12)
        tick.set_fontweight("bold")
    for tick in ax.get_yticklabels():
        tick.set_fontsize(12)
        tick.set_fontweight("bold")
    ax.legend(loc="best", fontsize=12, prop={"weight": "bold"})
    fig.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=320)
    logger.info("Saved plot to %s", args.output)


if __name__ == "__main__":
    main()
