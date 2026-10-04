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


def load_points(edit_distance_jsonl: Path) -> tuple[list[int], list[float]]:
    text_lens = []
    edit_distances = []
    with edit_distance_jsonl.open("r", encoding="utf-8") as file_handle:
        for line_number, line in enumerate(file_handle, start=1):
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            text_len = record.get("text_len")
            edit_dist = record.get("edit_dist")
            if text_len is None or edit_dist is None:
                LOGGER.warning("Skip %s:%d because text_len or edit_dist is missing.", edit_distance_jsonl, line_number)
                continue
            text_lens.append(int(text_len))
            edit_distances.append(float(edit_dist))
    return text_lens, edit_distances


def plot_resolution(eval_root: Path, output_file: Path, resolution: int) -> None:
    layouts = iter_layouts()
    fig, axes = plt.subplots(6, 6, figsize=(24, 18), sharex=True, sharey=True)
    axes_flat = axes.ravel()

    for ax, layout in zip(axes_flat, layouts):
        edit_distance_jsonl = eval_root / layout / f"ocr_edit_distance_res_{resolution}.jsonl"
        if not edit_distance_jsonl.is_file():
            ax.set_title(f"{layout}\nmissing", fontsize=9)
            ax.set_xlim(0, 25000)
            ax.set_ylim(-0.05, 1.05)
            ax.grid(True, linestyle="--", linewidth=0.4, alpha=0.35)
            continue

        text_lens, edit_distances = load_points(edit_distance_jsonl)
        mean_edit_distance = sum(edit_distances) / len(edit_distances) if edit_distances else 0.0
        ax.scatter(text_lens, edit_distances, s=7, alpha=0.45, edgecolors="none")
        ax.set_title(f"{layout}  mean={mean_edit_distance:.3f}", fontsize=9)
        ax.set_xlim(0, 25000)
        ax.set_ylim(-0.05, 1.05)
        ax.grid(True, linestyle="--", linewidth=0.4, alpha=0.35)

    for ax in axes[-1, :]:
        ax.set_xlabel("Characters")
    for ax in axes[:, 0]:
        ax.set_ylabel("Edit distance")

    fig.suptitle(f"OCR edit distance by character length, resolution={resolution}", fontsize=16)
    fig.tight_layout(rect=(0, 0, 1, 0.975))
    output_file.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_file, dpi=300)
    plt.close(fig)
    LOGGER.info("Saved %s", output_file)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot OCR edit distance by character length for 36 layouts and 4 resolutions.")
    parser.add_argument("--eval_root", type=Path, default=Path("VQAExp/eval_results/ocr/qwen25vl7b"), help="Root directory containing per-layout OCR edit distance JSONL files.")
    parser.add_argument("--output_dir", type=Path, default=Path("VQAExp/eval_results/ocr/qwen25vl7b/edit_distance_by_text_len_36_layouts"), help="Directory to save output figures.")
    parser.add_argument("--resolutions", type=int, nargs="+", default=RESOLUTIONS, help="Resolutions to plot.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    for resolution in args.resolutions:
        output_file = args.output_dir / f"edit_distance_by_text_len_res_{resolution}.png"
        plot_resolution(args.eval_root, output_file, resolution)


if __name__ == "__main__":
    main()
