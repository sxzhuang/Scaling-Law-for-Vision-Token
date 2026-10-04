#!/usr/bin/env python3
"""Render the intervention data with the visual settings of Figure 1.

The visual constants intentionally mirror
scatter_plot/tokenlen_vs_editdist_square_alltypes.py.  Only the source data
changes: character counts come from selected metadata and edit distances come
from the 16_60_-1 evaluation results.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FormatStrFormatter, FuncFormatter


FIG_WIDTH = 6.3
FIG_HEIGHT = 1.8
FONT_SIZE = 8
RESOLUTIONS = (512, 640, 1024, 1280)
DATASETS = (
    ("economics", "economics"),
    ("laws", "laws"),
    ("letters", "letters"),
    ("medicine", "medicine"),
    ("newspaper", "newspapers"),
    ("pride", "novels"),
)
MARKERS = ("o", "s", "D", "^", "v", "<")
FIXED_COLORS = ("#1F77B4", "#E377C2", "#BCBD22", "#2CA02C", "#D62728", "#86CAD2")
SHADE_BREAKS_BY_RES = {
    512: (3500, 7000),
    640: (5600, 10100),
    1024: (9500, 20300),
    1280: (8600, 21000),
}
SHADE_COLORS = ("#B3E8B8", "#EDD894", "#C1C0C0")
SHADE_ALPHA = 0.18
REPO_ROOT = Path(__file__).resolve().parents[2]
AAAI_ROOT = REPO_ROOT / "aaai_resub"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot 16_60_-1 OCR results in the existing Figure 1 style.")
    parser.add_argument("--layout", type=str, default="16_60_-1", help="Intervention-layout directory name.")
    parser.add_argument("--output", type=Path, default=AAAI_ROOT / "figures/figure1_style_16_60_-1.jpg", help="Destination JPG path.")
    parser.add_argument("--dpi", type=int, default=300, help="JPG resolution in dots per inch.")
    return parser.parse_args()


def format_legend_label(label: str) -> str:
    if label == "pride":
        return "Novels"
    if label == "newspaper":
        return "Newspapers"
    return label[:1].upper() + label[1:]


def format_k_ticks(value: float, _position: int) -> str:
    if abs(value) >= 1000:
        scaled = value / 1000
        return f"{scaled:g}k"
    return f"{value:g}"


def load_character_counts(domain: str, resolution: int) -> dict[str, int]:
    metadata_path = AAAI_ROOT / "selected_samples" / domain / str(resolution) / "selected_metadata.json"
    records = json.loads(metadata_path.read_text(encoding="utf-8"))
    counts: dict[str, int] = {}
    for record in records:
        sample_id = Path(str(record["image"])).stem
        text_len = record.get("text_len")
        counts[sample_id] = int(text_len) if text_len is not None else len(str(record.get("gt", "")))
    if len(counts) != len(records):
        raise ValueError(f"Duplicate sample IDs in {metadata_path}")
    return counts


def load_points(domain: str, resolution: int, layout: str) -> tuple[list[int], list[float]]:
    metric_path = AAAI_ROOT / "eval_results" / domain / str(resolution) / layout / "dpsk_eval_metric.json"
    metrics = json.loads(metric_path.read_text(encoding="utf-8"))
    char_counts = load_character_counts(domain, resolution)
    metric_ids = {str(record["id"]) for record in metrics}
    if metric_ids != set(char_counts):
        raise ValueError(f"Metric/metadata IDs do not match for {domain}, R={resolution}")
    return ([char_counts[str(record["id"])] for record in metrics], [float(record["edit_dist"]) for record in metrics])


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

    fig, axes = plt.subplots(1, 4, figsize=(FIG_WIDTH, FIG_HEIGHT), dpi=args.dpi)
    fig.subplots_adjust(left=0.04, right=0.995, bottom=0.22, top=0.78, wspace=0.06)
    style_map = {
        label: {"domain": domain, "marker": marker, "color": color}
        for (label, domain), marker, color in zip(DATASETS, MARKERS, FIXED_COLORS)
    }

    total_points = 0
    for axis, resolution in zip(axes, RESOLUTIONS):
        for label, style in style_map.items():
            char_counts, edit_distances = load_points(style["domain"], resolution, args.layout)
            total_points += len(char_counts)
            axis.scatter(
                char_counts,
                edit_distances,
                s=2,
                alpha=0.7,
                edgecolors="none",
                marker=style["marker"],
                color=style["color"],
                zorder=2,
            )

        x_min, x_max = axis.get_xlim()
        span_edges = [x_min, *SHADE_BREAKS_BY_RES[resolution], x_max]
        for (start, end), color in zip(zip(span_edges[:-1], span_edges[1:]), SHADE_COLORS):
            if start < end:
                axis.axvspan(start, end, color=color, alpha=SHADE_ALPHA, zorder=0)
        for break_x in SHADE_BREAKS_BY_RES[resolution]:
            if x_min < break_x < x_max:
                axis.axvline(break_x, color="#666666", linewidth=0.4, linestyle="--", zorder=1)
        axis.set_xlim(x_min, x_max)
        axis.xaxis.set_major_formatter(FuncFormatter(format_k_ticks))
        axis.tick_params(axis="x", length=2)
        axis.set_yticks((0.0, 0.2, 0.4, 0.6, 0.8, 1.0))
        axis.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))
        axis.set_ylabel("")
        if resolution == 512:
            axis.text(-0.06, 1.04, "ED", transform=axis.transAxes, ha="left", va="bottom", clip_on=False)
            axis.tick_params(axis="y", length=2, pad=1)
        else:
            axis.tick_params(axis="y", left=False, labelleft=False)
        axis.set_xlabel("")
        axis.text(0.5, 1.04, f"R={resolution}", transform=axis.transAxes, ha="center", va="bottom", clip_on=False)
        axis.grid(True, linestyle=":", linewidth=0.5, alpha=0.6)

    legend_handles = [
        Line2D([0], [0], linestyle="none", marker=style["marker"], markersize=6, markerfacecolor=style["color"], markeredgecolor=style["color"], label=format_legend_label(label))
        for label, style in style_map.items()
    ]
    fig.legend(handles=legend_handles, loc="upper center", bbox_to_anchor=(0.5, 0.985), ncol=6, frameon=False, columnspacing=0.9, handletextpad=0.35, borderaxespad=0.0)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, format="jpg", dpi=args.dpi)
    plt.close(fig)
    logging.info("Saved %d points to %s", total_points, args.output)


if __name__ == "__main__":
    main()
