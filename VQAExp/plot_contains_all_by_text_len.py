import argparse
import json
import logging
from pathlib import Path

import matplotlib.pyplot as plt


LOGGER = logging.getLogger(__name__)


def load_points(input_json: Path) -> tuple[list[int], list[float]]:
    with input_json.open("r", encoding="utf-8") as f:
        data = json.load(f)

    details = data.get("details")
    if not isinstance(details, list):
        raise ValueError("Input JSON must contain a details list.")

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


def plot_scatter(text_lens: list[int], contains_all_scores: list[float], output_file: Path) -> None:
    output_file.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(10, 6))
    plt.scatter(text_lens, contains_all_scores, s=18, alpha=0.7, edgecolors="none")
    plt.xlabel("text_len")
    plt.ylabel("contains_all")
    plt.title("Retrieval VQA contains_all by text length")
    plt.ylim(-0.05, 1.05)
    plt.grid(True, linestyle="--", linewidth=0.5, alpha=0.4)
    plt.tight_layout()
    plt.savefig(output_file, dpi=300)
    plt.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot contains_all versus text_len from Retrieval VQA evaluation details.")
    parser.add_argument("--input_json", type=Path, default=Path("VQAExp/eval_results/retrieval/qwen25vl7b/28_6_0/res640_details.json"), help="Path to evaluation details JSON.")
    parser.add_argument("--output_file", type=Path, default=Path("VQAExp/eval_results/retrieval/qwen25vl7b/28_6_0/res640_contains_all_by_text_len.png"), help="Path to save the scatter plot.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    text_lens, contains_all_scores = load_points(args.input_json)
    if not text_lens:
        raise ValueError("No valid points found in input JSON.")
    plot_scatter(text_lens, contains_all_scores, args.output_file)
    LOGGER.info("Saved scatter plot with %d points to %s", len(text_lens), args.output_file)


if __name__ == "__main__":
    main()
