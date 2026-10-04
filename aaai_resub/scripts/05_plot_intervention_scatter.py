#!/usr/bin/env python3
"""Plot the law-guided OCR intervention results in a Figure-1-style layout."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


REPO_ROOT = Path(__file__).resolve().parents[2]
AAAI_ROOT = REPO_ROOT / "aaai_resub"
DOMAINS = ("novels", "laws", "economics", "medicine", "newspapers", "letters")
RESOLUTIONS = (512, 640, 1024, 1280)
DOMAIN_LABELS = {
    "novels": "Novels",
    "laws": "Laws",
    "economics": "Economics",
    "medicine": "Medicine",
    "newspapers": "Newspapers",
    "letters": "Letters",
}
DOMAIN_COLORS = {
    "novels": "#4F86B7",
    "laws": "#D48BC4",
    "economics": "#F2A33A",
    "medicine": "#5FA95F",
    "newspapers": "#C96A6A",
    "letters": "#78BAC1",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot four vertical OCR intervention scatter subplots.")
    parser.add_argument("--layout", type=str, default="16_60_-1", help="Intervention-layout directory name.")
    parser.add_argument("--output", type=Path, default=AAAI_ROOT / "figures/fig1_style_intervention_16_60_-1.jpg", help="Destination JPG path.")
    parser.add_argument("--dpi", type=int, default=300, help="JPG resolution in dots per inch.")
    return parser.parse_args()


def load_character_counts(domain: str, resolution: int) -> dict[str, int]:
    metadata_path = AAAI_ROOT / "selected_samples" / domain / str(resolution) / "selected_metadata.json"
    records = json.loads(metadata_path.read_text(encoding="utf-8"))
    counts: dict[str, int] = {}
    for record in records:
        image_name = str(record["image"])
        sample_id = Path(image_name).stem
        char_count = record.get("text_len")
        if char_count is None:
            char_count = len(str(record.get("gt", "")))
        counts[sample_id] = int(char_count)
    if len(counts) != len(records):
        raise ValueError(f"Duplicate image IDs in {metadata_path}")
    return counts


def load_points(domain: str, resolution: int, layout: str) -> list[tuple[float, int]]:
    metric_path = AAAI_ROOT / "eval_results" / domain / str(resolution) / layout / "dpsk_eval_metric.json"
    metrics = json.loads(metric_path.read_text(encoding="utf-8"))
    char_counts = load_character_counts(domain, resolution)
    points: list[tuple[float, int]] = []
    metric_ids: set[str] = set()
    for record in metrics:
        sample_id = str(record["id"])
        if sample_id in metric_ids:
            raise ValueError(f"Duplicate metric ID {sample_id} in {metric_path}")
        metric_ids.add(sample_id)
        if sample_id not in char_counts:
            raise KeyError(f"Metric ID {sample_id} is absent from selected metadata: {metric_path}")
        points.append((float(record["edit_dist"]), char_counts[sample_id]))
    if metric_ids != set(char_counts):
        raise ValueError(
            f"Metric/metadata mismatch for {domain}, R={resolution}: "
            f"metrics={len(metric_ids)}, metadata={len(char_counts)}"
        )
    return points


def main() -> None:
    args = parse_args()
    points_by_resolution: dict[int, dict[str, list[tuple[float, int]]]] = {}
    for resolution in RESOLUTIONS:
        points_by_resolution[resolution] = {
            domain: load_points(domain, resolution, args.layout) for domain in DOMAINS
        }

    plt.rcParams.update({"font.size": 11, "axes.titlesize": 12, "axes.labelsize": 12})
    fig, axes = plt.subplots(nrows=4, ncols=1, figsize=(9, 20), sharex=True)
    for axis, resolution in zip(axes, RESOLUTIONS):
        for domain in DOMAINS:
            points = points_by_resolution[resolution][domain]
            edit_distances, character_counts = zip(*points)
            axis.scatter(
                edit_distances,
                character_counts,
                s=8,
                alpha=0.62,
                color=DOMAIN_COLORS[domain],
                edgecolors="none",
                label=DOMAIN_LABELS[domain],
            )
        axis.set_title(f"Resolution {resolution}", loc="left")
        axis.set_xlim(-0.02, 1.02)
        axis.grid(axis="x", color="#D9D9D9", linewidth=0.6, alpha=0.8)
        axis.set_axisbelow(True)

    axes[-1].set_xlabel("Normalized edit distance")
    fig.text(0.015, 0.5, "Character count", va="center", rotation="vertical", fontsize=12)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.52, 0.995), ncol=3, frameon=False)
    fig.subplots_adjust(top=0.96, bottom=0.05, left=0.1, right=0.98, hspace=0.25)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=args.dpi, format="jpg")
    plt.close(fig)
    point_count = sum(len(points_by_resolution[resolution][domain]) for resolution in RESOLUTIONS for domain in DOMAINS)
    print(f"Wrote {point_count} scatter points to {args.output}")


if __name__ == "__main__":
    main()
