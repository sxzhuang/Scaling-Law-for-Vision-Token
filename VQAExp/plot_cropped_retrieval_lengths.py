import argparse
import json
import logging
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


LOGGER = logging.getLogger(__name__)
DEFAULT_BIN_SIZE = 1000


def load_lengths(input_file: Path) -> list[int]:
    with input_file.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return [int(sample["text_len"]) for sample in data]


def build_bins(lengths: list[int], bin_size: int, max_length: int | None) -> list[int]:
    if max_length is None:
        max_length = int(math.ceil(max(lengths) / bin_size) * bin_size)
    return list(range(0, max_length + bin_size, bin_size))


def plot_histogram(lengths: list[int], output_file: Path, bin_size: int, max_length: int | None) -> None:
    bins = build_bins(lengths, bin_size, max_length)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(12, 6))
    plt.hist(lengths, bins=bins, edgecolor="black", linewidth=0.7)
    plt.xlabel("Cropped context length (characters)")
    plt.ylabel("Number of examples")
    plt.title(f"Cropped Retrieval Context Lengths (bin size = {bin_size} characters)")
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_file, dpi=200)
    plt.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot text_len distribution for the cropped Retrieval JSON dataset.")
    parser.add_argument("--input_file", type=Path, default=Path("VQAExp/data/Retrieval/retrieval_cropped_0_25k_600.json"), help="Input cropped Retrieval JSON file.")
    parser.add_argument("--output_png", type=Path, default=Path("VQAExp/data/Retrieval/retrieval_cropped_0_25k_600_length_histogram.png"), help="Path to save histogram PNG.")
    parser.add_argument("--bin_size", type=int, default=DEFAULT_BIN_SIZE, help="Histogram bin size in characters.")
    parser.add_argument("--max_length", type=int, default=25000, help="Maximum histogram length.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    lengths = load_lengths(args.input_file)
    if not lengths:
        raise ValueError("No samples found in input JSON.")

    LOGGER.info("Samples: %d", len(lengths))
    LOGGER.info("Min characters: %d", min(lengths))
    LOGGER.info("Max characters: %d", max(lengths))
    LOGGER.info("Mean characters: %.2f", sum(lengths) / len(lengths))

    plot_histogram(lengths, args.output_png, args.bin_size, args.max_length)
    LOGGER.info("Saved histogram to %s", args.output_png)


if __name__ == "__main__":
    main()
