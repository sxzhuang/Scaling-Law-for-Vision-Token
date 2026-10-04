import argparse
import json
import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


LOGGER = logging.getLogger(__name__)
FONT_SIZES = [20, 28, 36]
LINE_SPACINGS = [0, 6, 24, 42]
LETTER_SPACINGS = [-1, 0, 7]
RESOLUTIONS = [512, 640, 1024, 1280]


def iter_layouts() -> list[str]:
    return [f"{font_size}_{line_spacing}_{letter_spacing}" for font_size in FONT_SIZES for line_spacing in LINE_SPACINGS for letter_spacing in LETTER_SPACINGS]


def load_points(eval_json: Path) -> tuple[list[int], list[float]]:
    with eval_json.open("r", encoding="utf-8") as file_handle:
        data = json.load(file_handle)

    details = data.get("details")
    if not isinstance(details, list):
        raise ValueError(f"{eval_json} must contain a details list.")

    text_lens = []
    contains_all_scores = []
    for item in details:
        text_len = item.get("text_len")
        contains_all = item.get("contains_all")
        if text_len is None or contains_all is None:
            continue
        text_lens.append(int(text_len))
        contains_all_scores.append(float(contains_all))
    return text_lens, contains_all_scores


def plot_resolution(eval_root: Path, output_file: Path, resolution: int) -> None:
    layouts = iter_layouts()
    fig, axes = plt.subplots(6, 6, figsize=(24, 18), sharex=True, sharey=True)
    axes_flat = axes.ravel()

    for ax, layout in zip(axes_flat, layouts):
        eval_json = eval_root / layout / f"reasoning_acc_res_{resolution}.jsonl"
        if not eval_json.is_file():
            ax.set_title(f"{layout}\nmissing", fontsize=9)
            ax.set_xlim(0, 25000)
            ax.set_ylim(-0.05, 1.05)
            ax.grid(True, linestyle="--", linewidth=0.4, alpha=0.35)
            continue

        text_lens, contains_all_scores = load_points(eval_json)
        mean_acc = sum(contains_all_scores) / len(contains_all_scores) if contains_all_scores else 0.0
        ax.scatter(text_lens, contains_all_scores, s=7, alpha=0.45, edgecolors="none")
        ax.set_title(f"{layout}  acc={mean_acc:.3f}", fontsize=9)
        ax.set_xlim(0, 25000)
        ax.set_ylim(-0.05, 1.05)
        ax.grid(True, linestyle="--", linewidth=0.4, alpha=0.35)

    for ax in axes[-1, :]:
        ax.set_xlabel("Characters")
    for ax in axes[:, 0]:
        ax.set_ylabel("contains_all")

    fig.suptitle(f"Reasoning contains_all by character length, resolution={resolution}", fontsize=16)
    fig.tight_layout(rect=(0, 0, 1, 0.975))
    output_file.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_file, dpi=300)
    plt.close(fig)
    LOGGER.info("Saved %s", output_file)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot Reasoning contains_all by character length for 36 layouts and 4 resolutions.")
    parser.add_argument("--eval_root", type=Path, default=Path("VQAExp/eval_results/reasoning/qwen25vl7b"), help="Root directory containing per-layout Reasoning accuracy JSON files.")
    parser.add_argument("--output_dir", type=Path, default=Path("VQAExp/eval_results/reasoning/qwen25vl7b"), help="Directory to save output figures.")
    parser.add_argument("--resolutions", type=int, nargs="+", default=RESOLUTIONS, help="Resolutions to plot.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    for resolution in args.resolutions:
        output_file = args.output_dir / f"reasoning_acc_by_text_len_res_{resolution}.png"
        plot_resolution(args.eval_root, output_file, resolution)


if __name__ == "__main__":
    main()
