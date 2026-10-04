import argparse
import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from cal_metric import calc_contains_all, load_gt_by_id, load_predictions_by_id


LOGGER = logging.getLogger(__name__)
FONT_SIZES = [20, 28, 36]
LINE_SPACINGS = [0, 6, 24, 42]
LETTER_SPACINGS = [-1, 0, 7]
RESOLUTIONS = [512, 640, 1024, 1280]


def iter_layouts() -> list[str]:
    return [f"{font_size}_{line_spacing}_{letter_spacing}" for font_size in FONT_SIZES for line_spacing in LINE_SPACINGS for letter_spacing in LETTER_SPACINGS]


def load_points(gt_by_id: dict[str, dict], predictions_jsonl: Path) -> tuple[list[int], list[float]]:
    predictions_by_id = load_predictions_by_id(predictions_jsonl)
    text_lens = []
    contains_all_scores = []
    for sample_id, sample in sorted(gt_by_id.items()):
        prediction_record = predictions_by_id.get(sample_id)
        if prediction_record is None:
            continue
        text_len = sample.get("text_len")
        if text_len is None:
            continue
        contains_all = calc_contains_all(str(prediction_record["prediction"]), sample["answers"])
        text_lens.append(int(text_len))
        contains_all_scores.append(float(contains_all))
    return text_lens, contains_all_scores


def plot_resolution(gt_by_id: dict[str, dict], predictions_root: Path, output_file: Path, resolution: int) -> None:
    layouts = iter_layouts()
    fig, axes = plt.subplots(6, 6, figsize=(24, 18), sharex=True, sharey=True)
    axes_flat = axes.ravel()

    for ax, layout in zip(axes_flat, layouts):
        predictions_jsonl = predictions_root / layout / f"vqa_predictions_res_{resolution}.jsonl"
        if not predictions_jsonl.is_file():
            ax.set_title(f"{layout}\nmissing", fontsize=9)
            ax.set_xlim(0, 25000)
            ax.set_ylim(-0.05, 1.05)
            ax.grid(True, linestyle="--", linewidth=0.4, alpha=0.35)
            continue

        text_lens, contains_all_scores = load_points(gt_by_id, predictions_jsonl)
        mean_score = sum(contains_all_scores) / len(contains_all_scores) if contains_all_scores else 0.0
        ax.scatter(text_lens, contains_all_scores, s=7, alpha=0.45, edgecolors="none")
        ax.set_title(f"{layout}  mean={mean_score:.3f}", fontsize=9)
        ax.set_xlim(0, 25000)
        ax.set_ylim(-0.05, 1.05)
        ax.grid(True, linestyle="--", linewidth=0.4, alpha=0.35)

    for ax in axes[-1, :]:
        ax.set_xlabel("text_len")
    for ax in axes[:, 0]:
        ax.set_ylabel("contains_all")

    fig.suptitle(f"Retrieval VQA contains_all by text_len, resolution={resolution}", fontsize=16)
    fig.tight_layout(rect=(0, 0, 1, 0.975))
    output_file.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_file, dpi=300)
    plt.close(fig)
    LOGGER.info("Saved %s", output_file)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot Retrieval VQA contains_all by text_len for 36 layouts and 4 resolutions.")
    parser.add_argument("--gt_json", type=Path, default=Path("VQAExp/data/Retrieval/retrieval_cropped_0_25k_600.json"), help="Ground-truth JSON file containing id, answers, and text_len.")
    parser.add_argument("--predictions_root", type=Path, default=Path("VQAExp/vqa_predictions/Qwen/Qwen2.5-VL-7B-Instruct/Retrieval"), help="Root directory containing layout prediction JSONL files.")
    parser.add_argument("--output_dir", type=Path, default=Path("VQAExp/eval_results/retrieval/qwen25vl7b/contains_all_by_text_len_36_layouts"), help="Directory to save output figures.")
    parser.add_argument("--resolutions", type=int, nargs="+", default=RESOLUTIONS, help="Resolutions to plot.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    gt_by_id = load_gt_by_id(args.gt_json)
    for resolution in args.resolutions:
        output_file = args.output_dir / f"contains_all_by_text_len_res_{resolution}.png"
        plot_resolution(gt_by_id, args.predictions_root, output_file, resolution)


if __name__ == "__main__":
    main()
