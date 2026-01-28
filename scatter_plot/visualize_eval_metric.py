#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


METRIC_KEYS = ["bleu", "f_measure", "precision", "recall", "edit_dist"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize OCR evaluation metrics.")
    parser.add_argument("--input", type=Path, default=Path("eval_results/pride_square/OpenGVLab/InternVL3_5-8B-896/dpsk_eval_metric.json"), help="Path to the per-record metric JSON.")
    parser.add_argument("--output", type=Path, default=Path("InternVL3_5-8B-896_eval_metric.png"), help="Destination path for the scatter plot image.")
    return parser.parse_args()


def load_metrics(path: Path) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(f"Metric JSON not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("Metric JSON must be a list of records.")
    return data


def plot_metrics(records: list[dict], output_path: Path) -> None:
    text_lengths = [record.get("text_len", 0) for record in records]
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), sharey=False)
    axes_flat = axes.flatten()

    for ax, key in zip(axes_flat, METRIC_KEYS):
        values = [record.get(key, 0) for record in records]
        ax.scatter(text_lengths, values, s=10, alpha=0.6)
        ax.set_title(key.replace("_", " ").title())
        ax.set_xlabel("Text Length")
        ax.set_ylabel(key.upper())
        ax.grid(True, linestyle="--", alpha=0.3)

    # Hide the unused sixth subplot (bottom-right)
    if len(axes_flat) > len(METRIC_KEYS):
        axes_flat[-1].axis("off")

    fig.suptitle("OCR Evaluation Metrics vs. Text Token Length", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    records = load_metrics(args.input)
    plot_metrics(records, args.output)


if __name__ == "__main__":
    main()
