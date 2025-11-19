#!/usr/bin/env python3
"""Plot text_token_len distributions for all *_square datasets."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Scan *_square folders for JSON files and plot text_token_len distributions.")
    parser.add_argument("--root", type=Path, default=Path("build_dataset_from_ebook"), help="Root directory containing *_square folders.")
    parser.add_argument("--output", type=Path, default=Path("text_token_len_distribution.png"), help="Path to save the generated plot.")
    parser.add_argument("--bins", type=int, default=50, help="Number of histogram bins for each dataset.")
    return parser.parse_args()


def iter_square_dirs(root: Path) -> Iterable[Path]:
    if not root.exists():
        raise FileNotFoundError(f"Root directory not found: {root}")
    for path in sorted(root.iterdir()):
        if path.is_dir() and path.name.endswith("_square"):
            yield path


def load_token_lengths(json_path: Path) -> list[int]:
    data = json.loads(json_path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"Expected a list at {json_path}, got {type(data).__name__}")
    lengths: list[int] = []
    for item in data:
        if not isinstance(item, dict):
            continue
        value = item.get("text_token_len")
        if isinstance(value, (int, float)):
            lengths.append(int(value))
    return lengths


def collect_distributions(root: Path) -> dict[str, list[int]]:
    results: dict[str, list[int]] = {}
    for square_dir in iter_square_dirs(root):
        json_files = sorted(square_dir.glob("*.json"))
        if not json_files:
            continue
        label = square_dir.name.split("_")[0]
        all_lengths: list[int] = []
        for json_file in json_files:
            try:
                all_lengths.extend(load_token_lengths(json_file))
            except Exception as exc:
                print(f"Skipping {json_file} due to error: {exc}")
        if all_lengths:
            results[label] = all_lengths
    return results


def compute_histograms(distributions: dict[str, list[int]], bin_count: int) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    if not distributions:
        raise RuntimeError("No text_token_len values were found in any *_square JSON files.")
    all_values = [value for values in distributions.values() for value in values]
    if not all_values:
        raise RuntimeError("No text_token_len values available to plot.")
    data_min = min(all_values)
    data_max = max(all_values)
    if data_min == data_max:
        data_min -= 0.5
        data_max += 0.5
    edges = np.linspace(data_min, data_max, bin_count + 1)
    histograms: dict[str, np.ndarray] = {}
    for label, values in distributions.items():
        counts, _ = np.histogram(values, bins=edges)
        histograms[label] = counts
    return edges, histograms


def plot_distributions(distributions: dict[str, list[int]], bins: int, output: Path) -> None:
    edges, histograms = compute_histograms(distributions, bins)
    bin_centers = (edges[:-1] + edges[1:]) / 2
    bin_width = (edges[1] - edges[0]) * 0.9
    labels = list(histograms.keys())
    num_labels = len(labels)
    bar_width = bin_width / max(num_labels, 1)

    plt.figure(figsize=(14, 7))
    for idx, label in enumerate(labels):
        offsets = bin_centers - (bin_width / 2) + idx * bar_width
        plt.bar(
            offsets,
            histograms[label],
            width=bar_width,
            label=label,
            edgecolor="black",
            linewidth=0.4,
            alpha=0.65,
        )

    for boundary in edges[1:-1]:
        plt.axvline(boundary, color="gray", linestyle="--", linewidth=0.4, alpha=0.7)

    xtick_labels = [f"{int(edges[i])}-{int(edges[i + 1])}" for i in range(len(edges) - 1)]
    plt.xticks(bin_centers, xtick_labels, rotation=45, ha="right")
    plt.xlabel("text_token_len ranges")
    plt.ylabel("Count per bin")
    plt.title("text_token_len Distribution across *_square datasets")
    plt.legend()
    plt.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output)
    plt.close()
    print(f"Saved distribution plot to {output}")


def main() -> None:
    args = parse_args()
    distributions = collect_distributions(args.root)
    plot_distributions(distributions, args.bins, args.output)


if __name__ == "__main__":
    main()


# python utils/plot_text_token_len_distribution.py --root build_dataset_from_ebook --output token_len.png --bins 10