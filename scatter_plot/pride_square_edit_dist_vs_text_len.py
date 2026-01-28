#!/usr/bin/env python3
import argparse
import json
import logging
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


DEFAULT_INPUT_DIR = Path("eval_results/pride_square")
DEFAULT_OUTPUT_PATH = Path("scatter_plot/pride_square_edit_dist_vs_text_len.png")
DEFAULT_NCOLS = 3


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot edit_dist (y) vs text_len (x) for each resolution in pride_square.")
    parser.add_argument("--input_dir", type=Path, default=DEFAULT_INPUT_DIR, help="Directory containing resolution subfolders with dpsk_eval_metric.json.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH, help="Output PNG path for the figure.")
    parser.add_argument("--ncols", type=int, default=DEFAULT_NCOLS, help="Number of subplot columns in the grid.")
    return parser.parse_args()


def _sorted_subdirs(input_dir: Path) -> list[Path]:
    subdirs = [path for path in input_dir.iterdir() if path.is_dir()]

    def sort_key(path: Path) -> tuple[int, object]:
        name = path.name
        if name.isdigit():
            return (0, int(name))
        return (1, name)

    return sorted(subdirs, key=sort_key)


def _load_records(json_path: Path) -> list[dict]:
    with json_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"{json_path} must contain a list of records.")
    return data


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    input_dir: Path = args.input_dir
    subdirs = _sorted_subdirs(input_dir)
    if not subdirs:
        raise FileNotFoundError(f"No subdirectories found under {input_dir}")

    ncols = max(1, args.ncols)
    nrows = math.ceil(len(subdirs) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.5 * ncols, 3.5 * nrows), dpi=200, squeeze=False, constrained_layout=True)

    for idx, subdir in enumerate(subdirs):
        ax = axes[idx // ncols][idx % ncols]
        json_path = subdir / "dpsk_eval_metric.json"
        if not json_path.exists():
            logging.warning("Missing metrics file: %s", json_path)
            ax.set_title(subdir.name)
            ax.axis("off")
            continue

        records = _load_records(json_path)
        text_lengths = []
        edit_dists = []
        missing_fields = 0
        for record in records:
            text_len = record.get("text_len")
            edit_dist = record.get("edit_dist")
            if text_len is None or edit_dist is None:
                missing_fields += 1
                continue
            text_lengths.append(text_len)
            edit_dists.append(edit_dist)

        if missing_fields:
            logging.warning("Skipped %d records missing text_len or edit_dist in %s", missing_fields, json_path)

        if text_lengths and edit_dists:
            ax.scatter(text_lengths, edit_dists, s=10, alpha=0.6)
        else:
            ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)

        ax.set_title(subdir.name)
        ax.set_xlabel("text_len")
        ax.set_ylabel("edit_dist")
        ax.grid(True, linestyle="--", alpha=0.3)

    total_axes = nrows * ncols
    for idx in range(len(subdirs), total_axes):
        axes[idx // ncols][idx % ncols].axis("off")

    output_path: Path = args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    logging.info("Saved figure to %s", output_path)


if __name__ == "__main__":
    main()
