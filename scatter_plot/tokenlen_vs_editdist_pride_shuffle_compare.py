import argparse
import json
import logging
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator, FuncFormatter, FormatStrFormatter


FIG_WIDTH = 3.25
FIG_HEIGHT = 2.0
FONT_SIZE = 8
MARKERS = ["s", "D"]
COLORS = ["#E377C2", "#BCBD22"]
SERIES = [
    ("Word Shuffle", Path("eval_results/pride_dataset_square_random_phase/640/dpsk_eval_metric.json")),
    ("Char Replace", Path("eval_results/pride_dataset_square_random/640/dpsk_eval_metric.json")),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot Pride 640 ED vs text length for word shuffle and character replacement.")
    parser.add_argument("--output", type=Path, default=Path("eval_results/pride_640_shuffle_compare_ed_vs_text_len.pdf"), help="Output PDF path.")
    return parser.parse_args()


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


def load_points(path: Path) -> tuple[list[int], list[float]]:
    with path.open("r", encoding="utf-8") as handle:
        records = json.load(handle)
    xs = [int(record.get("text_len", record.get("text_token_len", 0)) or 0) for record in records]
    ys = [float(record.get("edit_dist", 0.0) or 0.0) for record in records]
    return xs, ys


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    logging.getLogger("fontTools").setLevel(logging.WARNING)
    logging.getLogger("matplotlib").setLevel(logging.WARNING)

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

    fig, ax = plt.subplots(1, 1, figsize=(FIG_WIDTH, FIG_HEIGHT), dpi=200)
    fig.subplots_adjust(left=0.10, right=0.94, bottom=0.20, top=0.78)

    legend_handles = []
    for index, (label, json_path) in enumerate(SERIES):
        if not json_path.exists():
            logging.warning("Missing file %s", json_path)
            continue
        xs, ys = load_points(json_path)
        color = COLORS[index % len(COLORS)]
        marker = MARKERS[index % len(MARKERS)]
        ax.scatter(xs, ys, s=2, alpha=0.7, edgecolors="none", marker=marker, color=color, zorder=2)
        legend_handles.append(
            Line2D([0], [0], linestyle="none", marker=marker, markersize=6, markerfacecolor=color, markeredgecolor=color, label=label)
        )
        logging.info("Loaded %d points from %s", len(xs), json_path)

    ax.xaxis.set_major_formatter(FuncFormatter(format_k_ticks))
    ax.margins(x=0.04)
    ax.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))
    ax.yaxis.set_major_locator(FixedLocator([0.0, 0.25, 0.50, 0.75, 1.0]))
    ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.text(-0.065, 1.04, "ED", transform=ax.transAxes, ha="left", va="bottom", clip_on=False)
    ax.tick_params(axis="x", length=2)
    ax.tick_params(axis="y", length=2, pad=1)
    ax.grid(True, linestyle=":", linewidth=0.5, alpha=0.6)

    if legend_handles:
        fig.legend(
            handles=legend_handles,
            loc="upper center",
            bbox_to_anchor=(0.5, 0.98),
            ncol=len(legend_handles),
            frameon=False,
            columnspacing=0.9,
            handletextpad=0.35,
            borderaxespad=0.0,
        )

    output_path = args.output.with_suffix(".pdf")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)
    logging.info("Saved plot to %s", output_path)


if __name__ == "__main__":
    main()
