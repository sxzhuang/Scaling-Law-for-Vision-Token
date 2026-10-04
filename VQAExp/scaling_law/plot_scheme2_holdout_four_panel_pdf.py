import argparse
import csv
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter


DEFAULT_FIT_DIR = Path("VQAExp/scaling_law/Fitting/scheme2_resolution_beta_constrained")
DEFAULT_OUTPUT = Path("VQAExp/scaling_law/Fitting/scheme2_resolution_beta_constrained/plots_pdf/qwen_scheme2_resolution_beta_constrained_holdout_20_0_7_four_panel.pdf")
EFFECTIVE_QWEN_RESOLUTIONS = {512: 504, 1280: 1288}


def load_csv(path: Path) -> list[dict]:
    with path.open("r", newline="", encoding="utf-8") as file_handle:
        return list(csv.DictReader(file_handle))


def filter_rows(rows: list[dict], layout: str, resolution: int) -> list[dict]:
    return [row for row in rows if row["layout"] == layout and int(row["resolution"]) == int(resolution)]


def draw_panel(ax, points: list[dict], binned: list[dict], task: str, resolution: int, scatter_size: float) -> None:
    x_values = [int(row["text_len"]) for row in points]
    y_values = [float(row["correct"]) for row in points]
    ax.scatter(x_values, y_values, s=scatter_size, alpha=0.45, edgecolors="none", color="tab:orange", rasterized=True)
    sorted_bins = sorted(binned, key=lambda row: int(row["bin_start"]))
    if sorted_bins:
        bin_x = [float(row["bin_center"]) for row in sorted_bins]
        observed_y = [float(row["observed_acc"]) for row in sorted_bins]
        predicted_y = [float(row["predicted_acc"]) for row in sorted_bins]
        ax.plot(bin_x, observed_y, color="tab:green", marker="o", markersize=2.4, linewidth=1.1, label="Observed")
        ax.plot(bin_x, predicted_y, color="tab:red", marker="o", markersize=2.4, linewidth=1.1, label="Predicted")
    display_resolution = EFFECTIVE_QWEN_RESOLUTIONS.get(resolution, resolution)
    ax.set_title(f"{task.capitalize()} R={display_resolution}", pad=2.0)
    ax.set_xlim(0, 25000)
    ax.set_ylim(-0.05, 1.05)
    ax.set_xticks([0, 5000, 10000, 15000, 20000, 25000])
    ax.xaxis.set_major_formatter(FuncFormatter(lambda value, _: "0" if value == 0 else f"{int(value / 1000)}K"))
    ax.tick_params(axis="y", length=2.0)
    ax.grid(True, linestyle="--", linewidth=0.35, alpha=0.35)


def build_figure(args: argparse.Namespace) -> None:
    plt.rcParams.update(
        {
            "font.size": args.font_size,
            "axes.titlesize": args.font_size,
            "axes.labelsize": args.font_size,
            "xtick.labelsize": args.tick_size,
            "ytick.labelsize": args.tick_size,
            "legend.fontsize": args.legend_size,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    task_data = {}
    for task in ["retrieval", "reasoning"]:
        task_dir = args.fit_dir / task
        task_data[task] = {"points": load_csv(task_dir / "fit_points.csv"), "binned": load_csv(task_dir / "fit_points_binned.csv")}

    panels = [("retrieval", 512), ("retrieval", 1280), ("reasoning", 512), ("reasoning", 1280)]
    fig, axes = plt.subplots(1, 4, figsize=(args.width, args.height), sharex=True, sharey=True)
    for ax, (task, resolution) in zip(axes, panels):
        points = filter_rows(task_data[task]["points"], args.layout, resolution)
        binned = filter_rows(task_data[task]["binned"], args.layout, resolution)
        draw_panel(ax, points, binned, task, resolution, args.scatter_size)

    for ax in axes:
        ax.set_xlabel("")
    axes[0].set_ylabel("Accuracy")
    handles = [
        plt.Line2D([0], [0], marker="o", linestyle="none", color="tab:orange", markersize=3.5, label="Ground-truth correctness"),
        plt.Line2D([0], [0], color="tab:green", marker="o", markersize=3.5, linewidth=1.1, label="Observed Bin ACC"),
        plt.Line2D([0], [0], color="tab:red", marker="o", markersize=3.5, linewidth=1.1, label="Predicted Bin ACC"),
    ]
    fig.legend(handles=handles, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.08))
    fig.tight_layout(rect=(0, 0, 1, 0.9), pad=0.15, w_pad=0.2)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot the constrained scheme-2 held-out layout four-panel PDF for paper figures.")
    parser.add_argument("--fit_dir", type=Path, default=DEFAULT_FIT_DIR, help="Constrained scheme-2 fitting directory.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Output PDF path.")
    parser.add_argument("--layout", type=str, default="20_0_7", help="Held-out layout to plot.")
    parser.add_argument("--width", type=float, default=7.2, help="Figure width in inches, close to ACL double-column text width.")
    parser.add_argument("--height", type=float, default=1.72, help="Figure height in inches.")
    parser.add_argument("--font_size", type=float, default=8.0, help="Main font size.")
    parser.add_argument("--tick_size", type=float, default=7.0, help="Tick font size.")
    parser.add_argument("--legend_size", type=float, default=7.5, help="Legend font size.")
    parser.add_argument("--scatter_size", type=float, default=3.0, help="Scatter marker size.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    build_figure(args)


if __name__ == "__main__":
    main()
