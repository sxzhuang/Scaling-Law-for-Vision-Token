import argparse
import csv
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


FONT_SIZES = [20, 28, 36]
LINE_SPACINGS = [0, 6, 24, 42]
LETTER_SPACINGS = [-1, 0, 7]
RESOLUTIONS = [512, 640, 1024, 1280]
EFFECTIVE_QWEN_RESOLUTIONS = {512: 504, 640: 644, 1024: 1036, 1280: 1288}
DEFAULT_FIT_DIR = Path("VQAExp/scaling_law/Fitting/scheme2")
DEFAULT_OUTPUT_DIR = Path("VQAExp/scaling_law/Fitting/scheme2/plots_pdf")


def iter_layouts() -> list[str]:
    return [f"{font_size}_{line_spacing}_{letter_spacing}" for font_size in FONT_SIZES for line_spacing in LINE_SPACINGS for letter_spacing in LETTER_SPACINGS]


def load_csv(path: Path) -> list[dict]:
    with path.open("r", newline="", encoding="utf-8") as file_handle:
        return list(csv.DictReader(file_handle))


def group_by_layout_resolution(rows: list[dict]) -> dict[tuple[str, int], list[dict]]:
    grouped = {}
    for row in rows:
        grouped.setdefault((row["layout"], int(row["resolution"])), []).append(row)
    return grouped


def draw_panel(ax, points: list[dict], binned: list[dict], title: str, scatter_size: float) -> None:
    if points:
        train_x = [int(row["text_len"]) for row in points if row["split"] == "train"]
        train_y = [float(row["correct"]) for row in points if row["split"] == "train"]
        test_x = [int(row["text_len"]) for row in points if row["split"] == "test"]
        test_y = [float(row["correct"]) for row in points if row["split"] == "test"]
        if train_x:
            ax.scatter(train_x, train_y, s=scatter_size, alpha=0.35, edgecolors="none", color="tab:blue", rasterized=True)
        if test_x:
            ax.scatter(test_x, test_y, s=scatter_size * 1.15, alpha=0.55, edgecolors="none", color="tab:orange", rasterized=True)
    sorted_bins = sorted(binned, key=lambda row: int(row["bin_start"]))
    if sorted_bins:
        bin_x = [float(row["bin_center"]) for row in sorted_bins]
        observed_y = [float(row["observed_acc"]) for row in sorted_bins]
        predicted_y = [float(row["predicted_acc"]) for row in sorted_bins]
        ax.plot(bin_x, observed_y, color="tab:green", marker="o", markersize=1.8, linewidth=0.75)
        ax.plot(bin_x, predicted_y, color="tab:red", marker="o", markersize=1.8, linewidth=0.75)
    ax.set_title(title, pad=1.4)
    ax.set_xlim(0, 25000)
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, linestyle="--", linewidth=0.25, alpha=0.3)


def configure_matplotlib(args: argparse.Namespace) -> None:
    plt.rcParams.update(
        {
            "font.size": args.font_size,
            "axes.titlesize": args.title_size,
            "axes.labelsize": args.font_size,
            "xtick.labelsize": args.tick_size,
            "ytick.labelsize": args.tick_size,
            "legend.fontsize": args.legend_size,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def plot_task_resolution(task: str, resolution: int, point_groups: dict[tuple[str, int], list[dict]], bin_groups: dict[tuple[str, int], list[dict]], args: argparse.Namespace) -> Path:
    layouts = iter_layouts()
    fig, axes = plt.subplots(6, 6, figsize=(args.width, args.height), sharex=True, sharey=True)
    for ax, layout in zip(axes.ravel(), layouts):
        draw_panel(ax, point_groups.get((layout, resolution), []), bin_groups.get((layout, resolution), []), layout, args.scatter_size)
    for ax in axes[-1, :]:
        ax.set_xlabel("Characters")
    for ax in axes[:, 0]:
        ax.set_ylabel("Acc.")
    handles = [
        plt.Line2D([0], [0], marker="o", linestyle="none", color="tab:blue", markersize=3, label="Train samples"),
        plt.Line2D([0], [0], marker="o", linestyle="none", color="tab:orange", markersize=3, label="Held-out samples"),
        plt.Line2D([0], [0], color="tab:green", marker="o", markersize=3, linewidth=0.9, label="Observed bin acc"),
        plt.Line2D([0], [0], color="tab:red", marker="o", markersize=3, linewidth=0.9, label="Predicted bin acc"),
    ]
    fig.legend(handles=handles, loc="upper center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 1.0))
    if task == "retrieval":
        title = f"Capacity-Conditioned Retrieval Fit, R={EFFECTIVE_QWEN_RESOLUTIONS[resolution]}"
    else:
        title = f"Qwen {task.capitalize()} Scheme-2 Scaling Fit, Resolution={resolution}"
    fig.suptitle(title, y=1.002, fontsize=args.suptitle_size)
    fig.tight_layout(rect=(0, 0, 1, 0.975), pad=0.25, h_pad=0.4, w_pad=0.35)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / f"qwen_{task}_scheme2_scaling_fit_res_{resolution}_36_layouts.pdf"
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_task(task: str, args: argparse.Namespace) -> list[Path]:
    task_dir = args.fit_dir / task
    point_groups = group_by_layout_resolution(load_csv(task_dir / "fit_points.csv"))
    bin_groups = group_by_layout_resolution(load_csv(task_dir / "fit_points_binned.csv"))
    return [plot_task_resolution(task, resolution, point_groups, bin_groups, args) for resolution in args.resolutions]


def parse_resolutions(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot scheme-2 Retrieval and Reasoning PDFs with 36 layouts per resolution.")
    parser.add_argument("--fit_dir", type=Path, default=DEFAULT_FIT_DIR, help="Scheme-2 fitting directory.")
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Directory for output PDFs.")
    parser.add_argument("--tasks", type=str, default="retrieval,reasoning", help="Comma-separated tasks to plot.")
    parser.add_argument("--resolutions", type=parse_resolutions, default=RESOLUTIONS, help="Comma-separated resolutions to plot.")
    parser.add_argument("--width", type=float, default=10.0, help="Figure width in inches.")
    parser.add_argument("--height", type=float, default=7.5, help="Figure height in inches.")
    parser.add_argument("--font_size", type=float, default=6.5, help="Main font size.")
    parser.add_argument("--title_size", type=float, default=6.2, help="Subplot title size.")
    parser.add_argument("--tick_size", type=float, default=5.8, help="Tick font size.")
    parser.add_argument("--legend_size", type=float, default=7.0, help="Legend font size.")
    parser.add_argument("--suptitle_size", type=float, default=9.0, help="Figure title size.")
    parser.add_argument("--scatter_size", type=float, default=1.8, help="Scatter marker size.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configure_matplotlib(args)
    tasks = [task.strip() for task in args.tasks.split(",") if task.strip()]
    for task in tasks:
        if task not in {"retrieval", "reasoning"}:
            raise ValueError(f"Unsupported task: {task}")
        for output_path in plot_task(task, args):
            print(output_path)


if __name__ == "__main__":
    main()
