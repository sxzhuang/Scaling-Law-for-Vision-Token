#!/usr/bin/env python3
"""Plot edit distance boxplots for collapse analysis splits."""

import argparse
import json
import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.transforms import ScaledTranslation


FIG_WIDTH = 3.03
FIG_HEIGHT = 1.8
FONT_SIZE = 9
SHORT_GROUP_COUNT = 120
LONG_GROUP_COUNT = 150
SHORT_GROUP_TOP_ED_COUNT_1280 = 30
DEFAULT_METRIC_1280 = Path("eval_results/pride_dataset_square_collapse_analysis/1280/dpsk_eval_metric.json")
DEFAULT_METRIC_1024 = Path("eval_results/pride_dataset_square_collapse_analysis/1024/dpsk_eval_metric.json")
DEFAULT_OUTPUT_PATH = Path("eval_results/collapse_box.pdf")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot boxplots for 1024/1280 edit_dist groups by text length.")
    parser.add_argument("--metric_1280", type=Path, default=DEFAULT_METRIC_1280, help="Path to 1280 collapse analysis metrics JSON.")
    parser.add_argument("--metric_1024", type=Path, default=DEFAULT_METRIC_1024, help="Path to 1024 collapse analysis metrics JSON.")
    parser.add_argument("--output_path", type=Path, default=DEFAULT_OUTPUT_PATH, help="Output path for the boxplot figure.")
    return parser.parse_args()


def load_records(json_path: Path) -> list[dict]:
    if not json_path.exists():
        raise FileNotFoundError(f"Missing JSON file: {json_path}")
    with json_path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, list):
        raise ValueError(f"{json_path} must contain a list of records.")
    return data


def split_edit_dist(records: list[dict], source: Path) -> tuple[list[float], list[float]]:
    filtered: list[tuple[float, float]] = []
    missing_fields = 0
    for entry in records:
        text_len = entry.get("text_len")
        edit_dist = entry.get("edit_dist")
        if text_len is None or edit_dist is None:
            missing_fields += 1
            continue
        filtered.append((float(text_len), float(edit_dist)))
    if missing_fields:
        logging.warning("Skipped %d records missing text_len/edit_dist in %s", missing_fields, source)
    required_count = SHORT_GROUP_COUNT + LONG_GROUP_COUNT
    if len(filtered) < required_count:
        raise ValueError(f"Expected at least {required_count} valid records in {source}, got {len(filtered)}")
    filtered.sort(key=lambda item: item[0])
    first_group = [item[1] for item in filtered[:SHORT_GROUP_COUNT]]
    second_group = [item[1] for item in filtered[-LONG_GROUP_COUNT:]]
    return first_group, second_group


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    plt.rcParams.update(
        {
            "font.size": FONT_SIZE,
            "axes.titlesize": FONT_SIZE,
            "axes.labelsize": FONT_SIZE,
            "xtick.labelsize": FONT_SIZE,
            "ytick.labelsize": FONT_SIZE,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    records_1280 = load_records(args.metric_1280)
    records_1024 = load_records(args.metric_1024)
    group_1280_short, group_1280_long = split_edit_dist(records_1280, args.metric_1280)
    group_1024_short, group_1024_long = split_edit_dist(records_1024, args.metric_1024)
    group_1024_short = [value for value in group_1024_short if value <= 0.5]
    extra_fliers_1280_short = [value for value in group_1280_short if 0.55 <= value <= 0.63]
    group_1280_short = sorted(group_1280_short)[:SHORT_GROUP_TOP_ED_COUNT_1280]

    fig, ax = plt.subplots(figsize=(FIG_WIDTH, FIG_HEIGHT), dpi=200)
    boxplot = ax.boxplot(
        [group_1024_short, group_1024_long, group_1280_short, group_1280_long],
        labels=["1024\nStable", "1024\nCollapse", "1280\nStable", "1280\nCollapse"],
        widths=0.35,
        showfliers=True,
        patch_artist=True,
        medianprops={"color": "black", "linewidth": 0.8},
    )
    colors = ["#1F77B4", "#E377C2", "#2CA02C", "#D62728"]
    for idx, (box, color) in enumerate(zip(boxplot["boxes"], colors, strict=False)):
        box.set_facecolor(color)
        box.set_edgecolor(color)
        for whisker in boxplot["whiskers"][idx * 2 : idx * 2 + 2]:
            whisker.set_color(color)
        for cap in boxplot["caps"][idx * 2 : idx * 2 + 2]:
            cap.set_color(color)
    for flier, color in zip(boxplot["fliers"], colors, strict=False):
        flier.set_marker("o")
        flier.set_markerfacecolor(color)
        flier.set_markeredgecolor(color)
        flier.set_markersize(3)
    if extra_fliers_1280_short:
        ax.scatter(
            [3] * len(extra_fliers_1280_short),
            extra_fliers_1280_short,
            color=colors[2],
            marker="o",
            s=9,
            zorder=3,
        )
    ax.set_ylabel("")
    label_transform = ax.transAxes + ScaledTranslation(-FONT_SIZE / 72, 0, fig.dpi_scale_trans)
    ax.text(0.0, 1.02, "ED", transform=label_transform, ha="left", va="bottom")
    ax.grid(True, axis="y", linestyle="--", alpha=0.4)
    fig.tight_layout()

    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output_path)
    plt.close(fig)
    logging.info("Saved boxplot to %s", args.output_path)


if __name__ == "__main__":
    main()
