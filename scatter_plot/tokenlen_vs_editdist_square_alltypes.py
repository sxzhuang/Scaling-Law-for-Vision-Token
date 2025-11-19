import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


RESOLUTIONS = [512, 640, 1024, 1280]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot token length vs edit distance for all *_square datasets, grouped by resolution.")
    parser.add_argument("--input_root", type=Path, default=Path("eval_results"), help="Root directory containing *_square subfolders.")
    parser.add_argument("--output", type=Path, default=Path("eval_results/alltypes_tokenlen_vs_editdist.png"), help="Output PNG path for the combined plot.")
    return parser.parse_args()


def load_metrics(json_path: Path) -> list[dict]:
    with json_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def collect_data(root: Path) -> dict[int, Dict[str, List[int]]]:
    per_res_data: dict[int, Dict[str, List[int]]] = {res: {} for res in RESOLUTIONS}
    for square_dir in sorted(root.iterdir()):
        if not square_dir.is_dir() or not square_dir.name.endswith("_square"):
            continue
        dataset_label = square_dir.name[:-7]  # remove "_square"
        for res in RESOLUTIONS:
            json_file = square_dir / str(res) / "dpsk_eval_metric.json"
            if not json_file.exists():
                logging.warning("Missing file %s", json_file)
                continue
            records = load_metrics(json_file)
            xs = [int(rec.get("text_token_len", 0)) for rec in records]
            ys = [float(rec.get("edit_dist", 0.0)) for rec in records]
            if not xs:
                continue
            current = per_res_data.setdefault(res, {})
            current[dataset_label] = (xs, ys)
    return per_res_data


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    per_res_data = collect_data(args.input_root)
    fig, axes = plt.subplots(2, 2, figsize=(18, 12), dpi=200, constrained_layout=True)
    pos_map = {512: (0, 0), 640: (0, 1), 1024: (1, 0), 1280: (1, 1)}

    for res, datasets in per_res_data.items():
        if not datasets:
            continue
        row, col = pos_map[res]
        ax = axes[row][col]
        for label, (xs, ys) in datasets.items():
            ax.scatter(xs, ys, s=16, alpha=0.7, edgecolors="none", label=label)
        ax.set_title(f"Resolution {res}")
        ax.set_xlabel("text_token_len")
        ax.set_ylabel("edit_dist")
        ax.grid(True, linestyle=":", linewidth=0.5, alpha=0.6)
        ax.legend()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(args.output, bbox_inches="tight")
    logging.info("Saved plot to %s", args.output)


if __name__ == "__main__":
    main()
