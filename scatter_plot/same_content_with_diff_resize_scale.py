#!/usr/bin/env python3
"""Compare resize scale vs edit distance across base and pasted datasets."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot resize scale (max(width,height)/1024) vs edit distance for pasted vs base datasets.")
    parser.add_argument("--metric_a", type=Path, default=Path("eval_results/pride_square_paste2larger/1024/dpsk_eval_metric.json"), help="Metrics JSON for pasted images (with size suffix in id).")
    parser.add_argument("--metric_b", type=Path, default=Path("eval_results/pride_square/1024/dpsk_eval_metric.json"), help="Metrics JSON for base images (no size suffix).")
    parser.add_argument("--metadata", type=Path, default=Path("build_dataset_from_ebook/pride_dataset_square/pride_square_gt.json"), help="Metadata JSON containing width/height for base images.")
    parser.add_argument("--output", type=Path, default=Path("eval_results/pride_square_paste2larger/1024/resize_scale_vs_edit_distance.pdf"), help="Output plot path.")
    return parser.parse_args()


def load_json_list(path: Path) -> List[dict]:
    return json.loads(path.read_text(encoding="utf-8"))


def build_metadata_size_map(path: Path) -> Dict[str, Tuple[int, int]]:
    records = load_json_list(path)
    mapping: Dict[str, Tuple[int, int]] = {}
    if not isinstance(records, list):
        return mapping
    for entry in records:
        if not isinstance(entry, dict):
            continue
        image_name = str(entry.get("image", "")).strip()
        width = entry.get("width")
        height = entry.get("height")
        if not image_name or not isinstance(width, (int, float)) or not isinstance(height, (int, float)):
            continue
        mapping[Path(image_name).stem] = (int(width), int(height))
    return mapping


def parse_size_from_id(sample_id: str) -> Tuple[str, int, int]:
    """Extract base id and width/height from suffix ..._<w>_<h>."""
    parts = sample_id.rsplit("_", 2)
    if len(parts) != 3:
        raise ValueError(f"Unexpected id format (expected suffix _w_h): {sample_id}")
    base_id, w_str, h_str = parts
    return base_id, int(w_str), int(h_str)


def build_points_from_a(metrics: List[dict]) -> List[Tuple[float, float]]:
    points: List[Tuple[float, float]] = []
    for entry in metrics:
        sample_id = entry.get("id")
        edit_dist = entry.get("edit_dist")
        if not isinstance(sample_id, str) or edit_dist is None:
            continue
        try:
            _, width, height = parse_size_from_id(sample_id)
        except Exception:
            logger.warning("Skip A entry with unexpected id: %s", sample_id)
            continue
        scale = max(width, height) / 1024.0 if max(width, height) else 0.0
        points.append((scale, float(edit_dist)))
    return points


def build_points_from_b(metrics: List[dict], size_map: Dict[str, Tuple[int, int]]) -> List[Tuple[float, float]]:
    points: List[Tuple[float, float]] = []
    for entry in metrics:
        sample_id = entry.get("id")
        edit_dist = entry.get("edit_dist")
        if not isinstance(sample_id, str) or edit_dist is None:
            continue
        dims = size_map.get(Path(sample_id).stem)
        if dims is None:
            logger.warning("Missing size for %s in metadata; skipping.", sample_id)
            continue
        width, height = dims
        scale = max(width, height) / 1024.0 if max(width, height) else 0.0
        points.append((scale, float(edit_dist)))
    return points


def scatter_points(ax, points: List[Tuple[float, float]], *, color: str, marker: str, label: str, alpha: float = 0.6, size: int = 24) -> None:
    if not points:
        return
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    ax.scatter(xs, ys, color=color, alpha=alpha, s=size, marker=marker, label=label)


def main() -> None:
    args = parse_args()
    size_map = build_metadata_size_map(args.metadata)
    points_a = build_points_from_a(load_json_list(args.metric_a))
    points_b = build_points_from_b(load_json_list(args.metric_b), size_map)

    fig, ax = plt.subplots(figsize=(10, 6))
    scatter_points(ax, points_a, color="#D65F5F", marker="o", label="Paste2LargerImage")
    scatter_points(ax, points_b, color="#488f31", marker="^", label="Baseline")

    ax.set_xlabel("Resize Scale (max(width, height)/1024)", fontsize=14, fontweight="bold")
    ax.set_ylabel("Edit Distance", fontsize=14, fontweight="bold")
    for tick in ax.get_xticklabels():
        tick.set_fontsize(12)
        tick.set_fontweight("bold")
    for tick in ax.get_yticklabels():
        tick.set_fontsize(12)
        tick.set_fontweight("bold")
    ax.legend(loc="upper left", fontsize=12, prop={"weight": "bold"})
    fig.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=320)
    logger.info("Saved plot to %s", args.output)


if __name__ == "__main__":
    main()
