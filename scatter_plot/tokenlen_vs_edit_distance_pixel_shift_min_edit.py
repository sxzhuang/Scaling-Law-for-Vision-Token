#!/usr/bin/env python3
"""Plot edit distance trends for grid-shifted samples."""

from __future__ import annotations

import argparse
import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
LOGGER = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot edit distance vs. padded shift grid coordinates.")
    parser.add_argument("--metrics_json", type=Path, default=Path("eval_results/pride_square_large_variance/selected_from1024/dpsk_eval_metric.json"), help="Path to dpsk_eval_metric.json.")
    parser.add_argument("--output_path", type=Path, default=Path("eval_results/pride_square_large_variance/selected_from1024/token_len_vs_min_edit_distance.png"), help="Path to save the plot.")
    return parser.parse_args()

def load_metrics(path: Path) -> Dict[str, List[Tuple[int, float, float]]]:
    records = json.loads(path.read_text(encoding="utf-8"))
    grouped: Dict[str, List[Tuple[int, float, float]]] = defaultdict(list)
    for entry in records:
        sample_id = entry.get("id")
        edit_dist = entry.get("edit_dist")
        token_len = entry.get("text_token_len")
        if sample_id is None or edit_dist is None or token_len is None:
            continue
        parts = sample_id.split("_")
        if len(parts) < 2:
            continue
        prefix = "_".join(parts[:-2])
        shift_value = float(parts[-2]) * 100 + float(parts[-1])
        grouped[prefix].append(
            (int(token_len), shift_value, float(edit_dist))
        )
    return grouped


def plot_groups(grouped_data: Dict[str, List[Tuple[int, float, float]]], output_path: Path) -> None:
    plt.figure(figsize=(10, 6))
    for prefix, entries in grouped_data.items():
        best_entry = min(entries, key=lambda item: item[2])
        token_len, _, edit_dist = best_entry
        plt.scatter(token_len, edit_dist)

    plt.xlabel("Text Token Length")
    plt.ylabel("Edit Distance (min per group)")
    plt.title("Minimum Edit Distance per Shifted Group")
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300)
    LOGGER.info("Saved plot to %s", output_path)


def main() -> None:
    args = parse_args()
    grouped = load_metrics(args.metrics_json)
    if not grouped:
        raise RuntimeError("No valid entries found in the metrics JSON.")
    plot_groups(grouped, args.output_path)


if __name__ == "__main__":
    main()
