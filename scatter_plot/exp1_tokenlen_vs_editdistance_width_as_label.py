import argparse
import json
import logging
import re
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


RESOLUTIONS = [512, 640, 1024, 1280]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot 2D scatter: token_len vs edit_dist across resolutions, colored by width.")
    parser.add_argument("--input_dir", type=Path, default=Path("eval_results/pride_same_content_diff_shape"), help="Directory containing resolution subfolders with dpsk_eval_metric.json.")
    parser.add_argument("--output", type=Path, default=Path("eval_results/pride_same_content_diff_shape/exp1_2d_tokenlen_editdist.png"), help="Output PNG path for the combined plot.")
    return parser.parse_args()


def load_metrics(json_path: Path) -> List[dict]:
    with json_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def parse_id_width(sample_id: str) -> int:
    m = re.match(r".+_(\d+)$", sample_id)
    if not m:
        raise ValueError(f"Unrecognized id format: {sample_id}")
    return int(m.group(1))


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    input_dir: Path = args.input_dir
    output_path: Path = args.output

    # Load per-resolution data, grouped by width
    per_res_points: Dict[int, Dict[int, Tuple[List[int], List[float]]]] = {}
    all_x, all_y = [], []
    all_widths: List[int] = []
    for res in RESOLUTIONS:
        json_file = input_dir / str(res) / "dpsk_eval_metric.json"
        if not json_file.exists():
            logging.warning(f"Missing file: {json_file}")
            continue
        records = load_metrics(json_file)
        width_map: Dict[int, Tuple[List[int], List[float]]] = {}
        for rec in records:
            token_len = int(rec.get("text_token_len", 0))
            edit_dist = float(rec.get("edit_dist", 0.0))
            sid = rec.get("id", "")
            try:
                width = parse_id_width(sid)
            except ValueError:
                logging.warning(f"Skip record with unrecognized id: {sid}")
                continue
            xs, ys = width_map.setdefault(width, ([], []))
            xs.append(token_len)
            ys.append(edit_dist)
            all_x.append(token_len)
            all_y.append(edit_dist)
            all_widths.append(width)
        per_res_points[res] = width_map

    if not per_res_points:
        raise FileNotFoundError(f"No metrics found under {input_dir}")

    # Axis ranges for consistency across subplots
    x_min, x_max = (min(all_x), max(all_x)) if all_x else (0, 1)
    y_min, y_max = (min(all_y), max(all_y)) if all_y else (0.0, 1.0)

    # Consistent color map by width across all subplots
    unique_widths = sorted(set(all_widths))
    cmap = plt.get_cmap("tab20")
    color_map: Dict[int, Tuple[float, float, float, float]] = {w: cmap(i % cmap.N) for i, w in enumerate(unique_widths)}

    # Create figure with 4 subplots (2x2)
    fig, axes = plt.subplots(2, 2, figsize=(16, 12), dpi=200)
    pos_map = {512: (0, 0), 640: (0, 1), 1024: (1, 0), 1280: (1, 1)}

    for res in RESOLUTIONS:
        if res not in per_res_points:
            continue
        r, c = pos_map[res]
        ax = axes[r][c]
        width_map = per_res_points[res]
        handles = []
        labels = []
        # Plot per width
        for w in sorted(width_map.keys()):
            xs, ys = width_map[w]
            h = ax.scatter(xs, ys, s=12, color=color_map[w], label=str(w), alpha=0.8)
            handles.append(h)
            labels.append(str(w))
        ax.set_title(f"Resolution {res}")
        ax.set_xlabel("token_len")
        ax.set_ylabel("edit_dist")
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.grid(True, linestyle=":", linewidth=0.5, alpha=0.6)
        if handles:
            ax.legend(handles=handles, labels=labels, title="width", fontsize=8, title_fontsize=9, frameon=False)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path)
    logging.info(f"Saved plot to {output_path}")


if __name__ == "__main__":
    main()
