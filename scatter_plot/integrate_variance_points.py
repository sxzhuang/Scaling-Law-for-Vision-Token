#!/usr/bin/env python3
"""Integrate high-variance replacements on side-by-side scatter plots."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter

FIG_WIDTH = 5.8
FIG_HEIGHT = 3.0
FONT_SIZE = 8


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Overlay baseline and high-variance metrics on one plot.")
    parser.add_argument("--org_metric_640", type=Path, default=Path("eval_results/pride_square/640/dpsk_eval_metric.json"), help="Baseline metrics JSON for R=640.")
    parser.add_argument("--high_variance_metric_640", type=Path, default=Path("eval_results/pride_square_large_variance/selected_from640/dpsk_eval_metric.json"), help="High-variance metrics JSON for R=640.")
    parser.add_argument("--org_metric_1024", type=Path, default=Path("eval_results/pride_square/1024/dpsk_eval_metric.json"), help="Baseline metrics JSON for R=1024.")
    parser.add_argument("--high_variance_metric_1024", type=Path, default=Path("eval_results/pride_square_large_variance/selected_from1024/dpsk_eval_metric.json"), help="High-variance metrics JSON for R=1024.")
    parser.add_argument("--output_path", type=Path, default=Path("eval_results/pixel_shift_res.pdf"), help="Destination plot path.")
    parser.add_argument("--hard_wall_640", type=float, default=11000.0, help="Token length threshold separating recoverable vs saturated zones for R=640.")
    parser.add_argument("--hard_wall_1024", type=float, default=19150.0, help="Token length threshold separating recoverable vs saturated zones for R=1024.")
    return parser.parse_args()


def normalize_id(sample_id: str) -> str:
    parts = sample_id.split("_")
    if len(parts) >= 3 and parts[-1].isdigit() and parts[-2].isdigit():
        return "_".join(parts[:-2])
    return sample_id


def load_json(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def group_high_variance(records: list[dict]) -> dict[str, list[dict]]:
    grouped: dict[str, list[dict]] = {}
    for entry in records:
        sample_id = entry.get("id")
        edit_dist = entry.get("edit_dist")
        token_len = entry.get("text_token_len")
        if sample_id is None or edit_dist is None or token_len is None:
            continue
        key = normalize_id(sample_id)
        grouped.setdefault(key, []).append(entry)
    return grouped


def format_k_ticks(value: float, _pos: int) -> str:
    abs_value = abs(value)
    if abs_value >= 1000:
        scaled = value / 1000
        if isinstance(scaled, float) and scaled.is_integer():
            return f"{int(scaled)}k"
        return f"{scaled:g}k"
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return f"{value:g}"


def build_plot_data(org_metric: Path, hv_metric: Path) -> tuple[list[tuple[float, float]], list[tuple[float, float]], dict[str, tuple[float, float]], dict[str, tuple[float, float]]]:
    org_records = load_json(org_metric)
    hv_records = load_json(hv_metric)
    hv_groups = group_high_variance(hv_records)
    org_text_len: dict[str, float] = {}
    baseline_points: list[tuple[float, float]] = []
    selected_points: list[tuple[float, float]] = []
    selected_min_points: dict[str, tuple[float, float]] = {}
    overlap_map: dict[str, tuple[float, float]] = {}

    for entry in org_records:
        token_len = entry.get("text_len")
        edit_dist = entry.get("edit_dist")
        sample_id = entry.get("id")
        if token_len is None or edit_dist is None or sample_id is None:
            continue
        key = normalize_id(sample_id)
        point = (float(token_len), float(edit_dist))
        org_text_len[key] = float(token_len)
        if key in hv_groups:
            selected_points.append(point)
            overlap_map[key] = point
        else:
            baseline_points.append(point)

    for key, entries in hv_groups.items():
        base_text_len = org_text_len.get(key)
        if base_text_len is None:
            continue
        best = min(entries, key=lambda item: float(item["edit_dist"]))
        selected_min_points[key] = (base_text_len, float(best["edit_dist"]))

    return baseline_points, selected_points, selected_min_points, overlap_map


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
            "legend.fontsize": FONT_SIZE,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    configs = [
        {
            "res": 640,
            "org_metric": args.org_metric_640,
            "high_variance_metric": args.high_variance_metric_640,
            "hard_wall": args.hard_wall_640,
        },
        {
            "res": 1024,
            "org_metric": args.org_metric_1024,
            "high_variance_metric": args.high_variance_metric_1024,
            "hard_wall": args.hard_wall_1024,
        },
    ]

    fig, axes = plt.subplots(1, 2, figsize=(FIG_WIDTH, FIG_HEIGHT), dpi=200)
    fig.subplots_adjust(left=0.06, right=0.97, bottom=0.22, top=0.75, wspace=0.16)

    for idx, config in enumerate(configs):
        ax = axes[idx]
        baseline_points, selected_points, selected_min_points, overlap_map = build_plot_data(
            config["org_metric"],
            config["high_variance_metric"],
        )

        baseline_x = [x for x, _ in baseline_points]
        baseline_y = [y for _, y in baseline_points]
        ax.scatter(
            baseline_x,
            baseline_y,
            color="#8fa6bf",
            alpha=0.5,
            s=3,
            marker="o",
        )

        selected_x = [x for x, _ in selected_points]
        selected_y = [y for _, y in selected_points]
        ax.scatter(
            selected_x,
            selected_y,
            color="#2CA02C",
            alpha=0.9,
            s=5,
            marker="s",
            edgecolor="white",
            linewidth=0.2,
            zorder=4,
        )

        selected_min_x = [x for x, _ in selected_min_points.values()]
        selected_min_y = [y for _, y in selected_min_points.values()]
        ax.scatter(
            selected_min_x,
            selected_min_y,
            color="#D62728",
            alpha=0.9,
            s=5,
            marker="^",
            edgecolor="white",
            linewidth=0.25,
            zorder=5,
        )

        xmin, xmax = ax.get_xlim()
        ymin, ymax = ax.get_ylim()
        hard_wall = config["hard_wall"]
        ax.axvline(hard_wall, color="#424242", linestyle="--", linewidth=0.3, zorder=0)
        ax.axvspan(hard_wall, xmax, color="#d7ccc8", alpha=0.08, zorder=0)
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(-0.02, 1.02)
        ax.set_yticks([0.0, 0.25, 0.5, 0.75, 1.0])

        for key, orig_point in overlap_map.items():
            hv_point = selected_min_points.get(key)
            if hv_point is None:
                continue
            orig_x, orig_y = orig_point
            hv_x_point, hv_y_point = hv_point
            ax.annotate(
                "",
                xy=(hv_x_point, hv_y_point),
                xytext=(orig_x, orig_y),
                arrowprops=dict(
                    arrowstyle="-",
                    color="#EFB852",
                    lw=0.1,
                    shrinkA=0,
                    shrinkB=0,
                ),
            )

        ax.xaxis.set_major_formatter(FuncFormatter(format_k_ticks))
        ax.tick_params(axis="x", length=2)
        ax.tick_params(axis="y", length=2, pad=1)
        if idx == 0:
            ax.text(
                -0.06,
                1.04,
                "ED",
                transform=ax.transAxes,
                ha="left",
                va="bottom",
                clip_on=False,
            )

        if idx == len(configs) - 1:
            ax.set_xlabel("TL")
            ax.xaxis.set_label_coords(1.03, 0.0)
        else:
            ax.set_xlabel("")

        ax.text(
            0.5,
            1.04,
            f"R={config['res']}",
            transform=ax.transAxes,
            ha="center",
            va="bottom",
            clip_on=False,
        )
        ax.grid(True, linestyle=":", linewidth=0.5, alpha=0.6)

    legend_handles = [
        Line2D(
            [0],
            [0],
            linestyle="none",
            marker="o",
            markersize=4.2,
            markerfacecolor="#8fa6bf",
            markeredgecolor="#8fa6bf",
            label="Unselected",
        ),
        Line2D(
            [0],
            [0],
            linestyle="none",
            marker="s",
            markersize=4.2,
            markerfacecolor="#2CA02C",
            markeredgecolor="#2CA02C",
            label="Original",
        ),
        Line2D(
            [0],
            [0],
            linestyle="none",
            marker="^",
            markersize=4.6,
            markerfacecolor="#D62728",
            markeredgecolor="#D62728",
            label="Min-ED",
        ),
    ]

    fig.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.89),
        ncol=3,
        frameon=False,
        columnspacing=0.8,
        handletextpad=0.35,
        borderaxespad=0.0,
    )

    output_path = args.output_path
    if output_path.suffix.lower() != ".pdf":
        output_path = output_path.with_suffix(".pdf")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)
    logging.info("Saved overlay plot to %s", output_path)


if __name__ == "__main__":
    main()
