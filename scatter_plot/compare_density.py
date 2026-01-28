#!/usr/bin/env python3
"""Visualize edit distance against resized scaling factor for different density groups."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
LOGGER = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot max(width, height)/1024 vs edit distance for GroupA/B/C.")
    parser.add_argument("--group_a_metric", type=Path, default=Path("eval_results/pride_square_density_experiment/GroupA/1024/dpsk_eval_metric.json"), help="Metrics JSON for GroupA.")
    parser.add_argument("--group_b_metric", type=Path, default=Path("eval_results/pride_square_density_experiment/GroupB/1024/dpsk_eval_metric.json"), help="Metrics JSON for GroupB.")
    parser.add_argument("--group_c_metric", type=Path, default=Path("eval_results/pride_square_density_experiment/GroupC/1024/dpsk_eval_metric.json"), help="Metrics JSON for GroupC.")
    parser.add_argument("--group_a_density", type=Path, default=Path("build_dataset_from_ebook/pride_dataset_square_density_experiment/GroupA/density_record.json"), help="Density record for GroupA.")
    parser.add_argument("--group_b_density", type=Path, default=Path("build_dataset_from_ebook/pride_dataset_square_density_experiment/GroupB/density_record.json"), help="Density record for GroupB.")
    parser.add_argument("--group_c_density", type=Path, default=Path("build_dataset_from_ebook/pride_dataset_square_density_experiment/GroupC/density_record.json"), help="Density record for GroupC.")
    parser.add_argument("--output_path", type=Path, default=Path("eval_results/pride_square_density_experiment/resize_scaling_vs_edit_dist.pdf"), help="Destination plot path.")
    return parser.parse_args()


def load_metric(path: Path) -> List[dict]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_density_map(path: Path) -> Dict[str, Tuple[int, int]]:
    mapping: Dict[str, Tuple[int, int]] = {}
    records = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(records, list):
        return mapping
    for entry in records:
        if not isinstance(entry, dict):
            continue
        sample_id = entry.get("id")
        width = entry.get("width")
        height = entry.get("height")
        if not isinstance(sample_id, str) or width is None or height is None:
            continue
        mapping[sample_id] = (int(width), int(height))
    return mapping


def build_points(metrics: List[dict], density_map: Dict[str, Tuple[int, int]]) -> List[Tuple[float, float]]:
    points: List[Tuple[float, float]] = []
    for entry in metrics:
        sample_id = entry.get("id")
        edit_dist = entry.get("edit_dist")
        if sample_id is None or edit_dist is None:
            continue
        dims = density_map.get(sample_id)
        if dims is None:
            continue
        width, height = dims
        scale = max(width, height) / 1024.0 if max(width, height) else 0.0
        points.append((scale, float(edit_dist)))
    return points


def scatter_points(ax, points: List[Tuple[float, float]], *, color: str, marker: str, label: str, alpha: float = 0.6, size: int = 16) -> None:
    if not points:
        return
    xs = [x for x, _ in points]
    ys = [y for _, y in points]
    ax.scatter(xs, ys, color=color, alpha=alpha, s=size, marker=marker, label=label)


def main() -> None:
    args = parse_args()
    group_a_points = build_points(
        load_metric(args.group_a_metric),
        load_density_map(args.group_a_density),
    )
    group_b_points = build_points(
        load_metric(args.group_b_metric),
        load_density_map(args.group_b_density),
    )
    group_c_points = build_points(
        load_metric(args.group_c_metric),
        load_density_map(args.group_c_density),
    )

    fig, ax = plt.subplots(figsize=(10, 6))
    scatter_points(ax, group_a_points, color="#7fb2d3", marker="o", label="GroupA")
    scatter_points(ax, group_b_points, color="#488f31", marker="^", label="GroupB")
    scatter_points(ax, group_c_points, color="#D65F5F", marker="s", label="GroupC")

    ax.set_xlabel("Resize Scale", fontsize=14, fontweight="bold")
    ax.set_ylabel("Edit Distance", fontsize=14, fontweight="bold")
    for tick in ax.get_xticklabels():
        tick.set_fontsize(14)
        tick.set_fontweight("bold")
    for tick in ax.get_yticklabels():
        tick.set_fontsize(14)
        tick.set_fontweight("bold")
    ax.legend(loc="lower right", fontsize=16, prop={"weight": "bold"})
    fig.tight_layout()
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output_path, dpi=320)
    LOGGER.info("Saved scaling/edit-distance plot to %s", args.output_path)


if __name__ == "__main__":
    main()
