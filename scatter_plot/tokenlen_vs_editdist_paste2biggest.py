#!/usr/bin/env python3
"""Plot text token length against edit distance for paste2biggest groups."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import List, Tuple

import matplotlib.pyplot as plt

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
LOGGER = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot text_token_len vs edit_dist for GroupA/B/C paste2biggest.")
    parser.add_argument("--group_a_metric", type=Path, default=Path("eval_results/pride_square_density_experiment/GroupA_paste2biggest/1024/dpsk_eval_metric.json"), help="Metrics JSON for GroupA paste2biggest.")
    parser.add_argument("--group_b_metric", type=Path, default=Path("eval_results/pride_square_density_experiment/GroupB_paste2biggest/1024/dpsk_eval_metric.json"), help="Metrics JSON for GroupB paste2biggest.")
    parser.add_argument("--group_c_metric", type=Path, default=Path("eval_results/pride_square_density_experiment/GroupC_paste2biggest/1024/dpsk_eval_metric.json"), help="Metrics JSON for GroupC paste2biggest.")
    parser.add_argument("--output_path", type=Path, default=Path("eval_results/pride_square_density_experiment/text_token_len_vs_edit_dist_paste2biggest.pdf"), help="Destination plot path.")
    return parser.parse_args()


def load_metric(path: Path) -> List[dict]:
    return json.loads(path.read_text(encoding="utf-8"))


def build_points(metrics: List[dict]) -> List[Tuple[float, float]]:
    points: List[Tuple[float, float]] = []
    for entry in metrics:
        edit_dist = entry.get("edit_dist")
        text_token_len = entry.get("text_token_len")
        if edit_dist is None or text_token_len is None:
            continue
        points.append((float(text_token_len), float(edit_dist)))
    return points


def scatter_points(ax, points: List[Tuple[float, float]], *, color: str, marker: str, label: str, alpha: float = 0.6, size: int = 16) -> None:
    if not points:
        return
    xs = [x for x, _ in points]
    ys = [y for _, y in points]
    ax.scatter(xs, ys, color=color, alpha=alpha, s=size, marker=marker, label=label)


def main() -> None:
    args = parse_args()
    group_a_points = build_points(load_metric(args.group_a_metric))
    group_b_points = build_points(load_metric(args.group_b_metric))
    group_c_points = build_points(load_metric(args.group_c_metric))

    fig, ax = plt.subplots(figsize=(10, 6))
    scatter_points(ax, group_a_points, color="#7fb2d3", marker="o", label="GroupA paste2biggest")
    scatter_points(ax, group_b_points, color="#488f31", marker="^", label="GroupB paste2biggest")
    scatter_points(ax, group_c_points, color="#D65F5F", marker="s", label="GroupC paste2biggest")

    ax.set_xlabel("Text Token Length", fontsize=14, fontweight="bold")
    ax.set_ylabel("Edit Distance", fontsize=14, fontweight="bold")
    for tick in ax.get_xticklabels():
        tick.set_fontsize(14)
        tick.set_fontweight("bold")
    for tick in ax.get_yticklabels():
        tick.set_fontsize(14)
        tick.set_fontweight("bold")
    ax.legend(loc="lower right", fontsize=14, prop={"weight": "bold"})
    fig.tight_layout()

    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output_path, dpi=320)
    LOGGER.info("Saved token length vs edit distance plot to %s", args.output_path)


if __name__ == "__main__":
    main()
