import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


RESOLUTIONS = [512, 640, 1024, 1280]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot text_token_len (x) vs edit_dist (y) for near-square dataset across resolutions.")
    parser.add_argument("--input_dir", type=Path, default=Path("eval_results/pride_square"), help="Directory containing resolution subfolders with dpsk_eval_metric.json.")
    parser.add_argument("--output", type=Path, default=Path("eval_results/pride_square/exp2_tokenlen_vs_editdist.png"), help="Output PNG path for the combined plot.")
    return parser.parse_args()


def load_metrics(json_path: Path) -> List[dict]:
    with json_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    input_dir: Path = args.input_dir
    output_path: Path = args.output

    # Per-resolution: list of (token_len, edit_dist)
    per_res_points: Dict[int, Tuple[List[int], List[float]]] = {}
    all_x: List[int] = []
    all_y: List[float] = []

    for res in RESOLUTIONS:
        json_file = input_dir / str(res) / "dpsk_eval_metric.json"
        if not json_file.exists():
            logging.warning(f"Missing file: {json_file}")
            continue
        records = load_metrics(json_file)
        xs, ys = [], []
        for rec in records:
            xs.append(int(rec.get("text_token_len", 0)))
            ys.append(float(rec.get("edit_dist", 0.0)))
        if xs:
            per_res_points[res] = (xs, ys)
            all_x.extend(xs)
            all_y.extend(ys)

    if not per_res_points:
        raise FileNotFoundError(f"No metrics found under {input_dir}")

    # Shared axes ranges for comparability
    x_min, x_max = (min(all_x), max(all_x)) if all_x else (0, 1)
    y_min, y_max = (min(all_y), max(all_y)) if all_y else (0.0, 1.0)

    fig, axes = plt.subplots(2, 2, figsize=(16, 12), dpi=200, constrained_layout=True)
    pos_map = {512: (0, 0), 640: (0, 1), 1024: (1, 0), 1280: (1, 1)}

    for res in RESOLUTIONS:
        if res not in per_res_points:
            continue
        r, c = pos_map[res]
        ax = axes[r][c]
        xs, ys = per_res_points[res]
        ax.scatter(xs, ys, s=16, alpha=0.8, edgecolors="none")
        ax.set_title(f"Resolution {res}")
        ax.set_xlabel("text_token_len")
        ax.set_ylabel("edit_dist")
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.grid(True, linestyle=":", linewidth=0.5, alpha=0.6)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, bbox_inches="tight")
    logging.info(f"Saved plot to {output_path}")


if __name__ == "__main__":
    main()

