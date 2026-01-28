import argparse
import json
import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter, FormatStrFormatter

FIG_WIDTH = 6.3          # inches (固定宽度)
FIG_HEIGHT = 1.8         # inches (你可按论文版式调整高度，但宽度固定 6.3)
FONT_SIZE = 8            # 固定字号

RESOLUTIONS = [512, 640, 1024, 1280]
MARKERS = ["o", "s", "D", "^", "v", "<", ">", "P", "X", "*", "h", "H", "d", "p"]
# FIXED_COLORS = ["#4C78A8", "#F58518", "#54A24B", "#E45756", "#72B7B2", "#B279A2", "#FF9DA6", "#9D755D", "#BAB0AC", "#1F77B4"]
# FIXED_COLORS = ["#66C2A5", "#FC8D62", "#8DA0CB", "#FFD92F", "#A6D854", "#E78AC3"]
FIXED_COLORS = ["#1F77B4", "#E377C2", "#BCBD22", "#2CA02C", "#D62728", "#86CAD2"]
SHADE_BREAKS = [9500, 20300]
SHADE_BREAKS_BY_RES = {
    512: [3500, 7000],
    640: [5600, 10100],
    1024: SHADE_BREAKS,
    1280: [8600, 21000],
}
SHADE_COLORS = ["#B3E8B8", "#EDD894", "#C1C0C0"]
SHADE_ALPHA = 0.18
# E6A3AD

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot token length vs edit distance for all *_square datasets, grouped by resolution.")
    parser.add_argument("--input_root", type=Path, default=Path("eval_results"), help="Root directory containing *_square subfolders.")
    parser.add_argument("--output", type=Path, default=Path("eval_results/alltypes_tokenlen_vs_editdist.pdf"), help="Output PDF path for the combined plot.")
    return parser.parse_args()


def load_metrics(json_path: Path) -> list[dict]:
    with json_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def collect_data(root: Path) -> dict[int, dict[str, tuple[list[int], list[float]]]]:
    per_res_data: dict[int, dict[str, tuple[list[int], list[float]]]] = {res: {} for res in RESOLUTIONS}

    for square_dir in sorted(root.iterdir()):
        if not square_dir.is_dir() or not square_dir.name.endswith("_square"):
            continue

        dataset_label = square_dir.name[:-7]  # remove "_square"

        for res in RESOLUTIONS:
            json_file = square_dir / str(res) / "dpsk_eval_metric.json"
            if not json_file.exists():
                logging.warning("Missing file %s", json_file)
                continue

            records = load_metrics(json_file)
            xs = [int(rec.get("text_len", 0)) for rec in records]
            ys = [float(rec.get("edit_dist", 0.0)) for rec in records]
            if not xs:
                continue

            per_res_data[res][dataset_label] = (xs, ys)

    return per_res_data


def format_legend_label(label: str) -> str:
    cleaned = label.replace("_", " ").strip()
    if cleaned.lower() == "pride":
        display = "Novels"
    else:
        display = cleaned[:1].upper() + cleaned[1:]
    return display


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
            "pdf.fonttype": 42,  # TrueType，论文里更友好
            "ps.fonttype": 42,
        }
    )

    per_res_data = collect_data(args.input_root)

    dataset_labels = sorted({label for datasets in per_res_data.values() for label in datasets})
    colors = FIXED_COLORS

    style_map: dict[str, dict[str, str]] = {}
    for idx, label in enumerate(dataset_labels):
        style_map[label] = {
            "color": colors[idx % len(colors)],
            "marker": MARKERS[idx % len(MARKERS)],
        }
    legend_labels = {label: format_legend_label(label) for label in dataset_labels}

    # 固定输出尺寸：FIG_WIDTH x FIG_HEIGHT（不要 bbox_inches="tight"）
    fig, axes = plt.subplots(1, 4, figsize=(FIG_WIDTH, FIG_HEIGHT), dpi=200)

    # 手动控制边距，让 4 个子图尽可能宽，同时预留顶部放 legend
    fig.subplots_adjust(
        left=0.04,
        right=0.995,
        bottom=0.22,
        top=0.78,
        wspace=0.06,
    )

    pos_map = {512: 0, 640: 1, 1024: 2, 1280: 3}

    for res, datasets in per_res_data.items():
        if not datasets:
            continue

        idx = pos_map[res]
        ax = axes[idx]

        for label, (xs, ys) in datasets.items():
            style = style_map[label]
            ax.scatter(
                xs,
                ys,
                s=2,
                alpha=0.7,
                edgecolors="none",
                label=label,
                marker=style["marker"],
                color=style["color"],
                zorder=2,
            )

        shade_breaks = SHADE_BREAKS_BY_RES.get(res)
        if shade_breaks:
            x_min, x_max = ax.get_xlim()
            span_edges = [x_min, *shade_breaks, x_max]
            for (start, end), color in zip(zip(span_edges[:-1], span_edges[1:]), SHADE_COLORS):
                if start < end:
                    ax.axvspan(start, end, color=color, alpha=SHADE_ALPHA, zorder=0)
            for break_x in shade_breaks:
                if x_min < break_x < x_max:
                    ax.axvline(break_x, color="#666666", linewidth=0.4, linestyle="--", zorder=1)
            ax.set_xlim(x_min, x_max)

        ax.xaxis.set_major_formatter(FuncFormatter(format_k_ticks))
        ax.tick_params(axis="x", length=2)
        ax.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
        ax.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))
        ax.set_ylabel("")
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
            ax.tick_params(axis="y", length=2, pad=1)
        else:
            ax.tick_params(axis="y", left=False, labelleft=False)

        # 只在右侧最后一个放 TL
        if idx == 3:
            ax.set_xlabel("TL")
            ax.xaxis.set_label_coords(0.96, -0.02)
        else:
            ax.set_xlabel("")

        # 分辨率标注：放在子图上方
        ax.text(
            0.5,
            1.04,
            f"R={res}",
            transform=ax.transAxes,
            ha="center",
            va="bottom",
            clip_on=False,
        )

        ax.grid(True, linestyle=":", linewidth=0.5, alpha=0.6)

    # Legend（放在图内顶部，避免 bbox_inches="tight" 重新裁剪尺寸）
    legend_handles = [
        Line2D(
            [0],
            [0],
            linestyle="none",
            marker=style["marker"],
            markersize=6,
            markerfacecolor=style["color"],
            markeredgecolor=style["color"],
            label=legend_labels[label],
        )
        for label, style in style_map.items()
    ]

    if legend_handles:
        ncol = min(6, len(legend_handles))  # 太多就自动换行
        fig.legend(
            handles=legend_handles,
            labels=[legend_labels[label] for label in style_map.keys()],
            loc="upper center",
            bbox_to_anchor=(0.5, 0.985),   # 关键：y<1，确保在固定画布内
            ncol=ncol,
            frameon=False,
            columnspacing=0.9,
            handletextpad=0.35,
            borderaxespad=0.0,
        )

    output_path = args.output
    if output_path.suffix.lower() != ".pdf":
        output_path = output_path.with_suffix(".pdf")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # 不用 bbox_inches="tight"：保证输出 PDF 的物理尺寸就是 6.3 x FIG_HEIGHT
    fig.savefig(output_path)
    plt.close(fig)
    logging.info("Saved plot to %s", output_path)


if __name__ == "__main__":
    main()
