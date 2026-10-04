import argparse
import csv
import logging
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from datasets import load_dataset


LOGGER = logging.getLogger(__name__)
DEFAULT_REPO_ID = "MLLM-CL/VTCBench"
DEFAULT_SPLIT = "Reasoning"
DEFAULT_BIN_SIZE = 1000


def build_bins(lengths: list[int], bin_size: int) -> list[int]:
    max_length = max(lengths)
    upper = int(math.ceil(max_length / bin_size) * bin_size)
    return list(range(0, upper + bin_size, bin_size))


def save_lengths_csv(lengths: list[int], output_file: Path) -> None:
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with output_file.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["index", "context_char_length"])
        for index, length in enumerate(lengths):
            writer.writerow([index, length])


def plot_histogram(lengths: list[int], output_file: Path, bin_size: int) -> None:
    bins = build_bins(lengths, bin_size)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(12, 6))
    plt.hist(lengths, bins=bins, edgecolor="black", linewidth=0.7)
    plt.xlabel("Context length (characters)")
    plt.ylabel("Number of examples")
    plt.title(f"VTCBench Reasoning Context Lengths (bin size = {bin_size} characters)")
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_file, dpi=200)
    plt.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot a histogram of _context character lengths for the VTCBench Reasoning split.")
    parser.add_argument("--repo_id", type=str, default=DEFAULT_REPO_ID, help="Hugging Face dataset repo id.")
    parser.add_argument("--split", type=str, default=DEFAULT_SPLIT, help="Dataset split to analyze.")
    parser.add_argument("--bin_size", type=int, default=DEFAULT_BIN_SIZE, help="Histogram bin size in characters.")
    parser.add_argument("--output_png", type=Path, default=Path("VQAExp/figures/reasoning_context_length_histogram.png"), help="Path to save the histogram PNG.")
    parser.add_argument("--output_csv", type=Path, default=Path("VQAExp/figures/reasoning_context_lengths.csv"), help="Path to save per-example character lengths.")
    parser.add_argument("--cache_dir", type=Path, default=None, help="Optional Hugging Face cache directory.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    LOGGER.info("Loading %s split from %s", args.split, args.repo_id)
    dataset = load_dataset(args.repo_id, split=args.split, cache_dir=str(args.cache_dir) if args.cache_dir else None)
    contexts = dataset["_context"]
    lengths = [len(context) for context in contexts]

    if not lengths:
        raise ValueError("No contexts found in the dataset.")

    LOGGER.info("Examples: %d", len(lengths))
    LOGGER.info("Min characters: %d", min(lengths))
    LOGGER.info("Max characters: %d", max(lengths))
    LOGGER.info("Mean characters: %.2f", sum(lengths) / len(lengths))

    save_lengths_csv(lengths, args.output_csv)
    plot_histogram(lengths, args.output_png, args.bin_size)

    LOGGER.info("Saved lengths CSV to %s", args.output_csv)
    LOGGER.info("Saved histogram to %s", args.output_png)


if __name__ == "__main__":
    main()
