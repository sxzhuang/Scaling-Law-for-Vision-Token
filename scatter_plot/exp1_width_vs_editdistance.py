import argparse
import json
import logging
import re
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable


RESOLUTIONS = [512, 640, 1024, 1280]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot width (x) vs edit_dist (y) across resolutions with heat-colored points (token_len) and per-group connecting lines.")
    parser.add_argument("--input_dir", type=Path, default=Path("eval_results/pride_same_content_diff_shape"), help="Directory containing resolution subfolders with dpsk_eval_metric.json.")
    parser.add_argument("--output", type=Path, default=Path("eval_results/pride_same_content_diff_shape/exp1_width_vs_editdist.png"), help="Output PNG path for the combined plot.")
    return parser.parse_args()


def load_metrics(json_path: Path) -> List[dict]:
    with json_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def parse_group_and_width(sample_id: str) -> Tuple[str, int]:
    m = re.match(r"(.+_)(\d+)$", sample_id)
    if not m:
        raise ValueError(f"Unrecognized id format: {sample_id}")
    return m.group(1), int(m.group(2))


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    input_dir: Path = args.input_dir
    output_path: Path = args.output

    # Per-resolution: group prefix -> list of (width, edit_dist, token_len)
    per_res_groups: Dict[int, Dict[str, List[Tuple[int, float, int]]]] = {}
    all_widths: List[int] = []
    all_edit: List[float] = []
    all_token: List[int] = []

    for res in RESOLUTIONS:
        json_file = input_dir / str(res) / "dpsk_eval_metric.json"
        if not json_file.exists():
            logging.warning(f"Missing file: {json_file}")
            continue
        records = load_metrics(json_file)
        groups: Dict[str, List[Tuple[int, float, int]]] = {}
        for rec in records:
            sid = rec.get("id", "")
            try:
                prefix, width = parse_group_and_width(sid)
            except ValueError:
                logging.warning(f"Skip record with unrecognized id: {sid}")
                continue
            edit_dist = float(rec.get("edit_dist", 0.0))
            token_len = int(rec.get("text_token_len", 0))
            groups.setdefault(prefix, []).append((width, edit_dist, token_len))
            all_widths.append(width)
            all_edit.append(edit_dist)
            all_token.append(token_len)
        per_res_groups[res] = groups

    if not per_res_groups:
        raise FileNotFoundError(f"No metrics found under {input_dir}")

    # Axis and color normalization shared across subplots
    x_min, x_max = (min(all_widths), max(all_widths)) if all_widths else (0, 1)
    y_min, y_max = (min(all_edit), max(all_edit)) if all_edit else (0.0, 1.0)
    t_min, t_max = (min(all_token), max(all_token)) if all_token else (0, 1)
    cmap = plt.get_cmap("viridis")
    norm = Normalize(vmin=t_min, vmax=t_max)

    # Create figure with 4 subplots
    fig, axes = plt.subplots(2, 2, figsize=(16, 12), dpi=200, constrained_layout=True)
    pos_map = {512: (0, 0), 640: (0, 1), 1024: (1, 0), 1280: (1, 1)}

    for res in RESOLUTIONS:
        if res not in per_res_groups:
            continue
        r, c = pos_map[res]
        ax = axes[r][c]
        groups = per_res_groups[res]

        # Plot scatter for all points to ensure consistent color mapping
        xs_all, ys_all, cs_all = [], [], []
        for prefix, items in groups.items():
            for (w, e, t) in items:
                xs_all.append(w)
                ys_all.append(e)
                cs_all.append(t)
        if xs_all:
            ax.scatter(xs_all, ys_all, c=cs_all, cmap=cmap, norm=norm, s=16, alpha=0.9, edgecolors="none")

        # Connect points within each group by width order
        for prefix, items in groups.items():
            items_sorted = sorted(items, key=lambda t: t[0])
            x_line = [t[0] for t in items_sorted]
            y_line = [t[1] for t in items_sorted]
            ax.plot(x_line, y_line, color="#666666", linewidth=1.0, alpha=0.7)

        ax.set_title(f"Resolution {res}")
        ax.set_xlabel("width")
        ax.set_ylabel("edit_dist")
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.grid(True, linestyle=":", linewidth=0.5, alpha=0.6)

    # Shared colorbar for token_len
    sm = ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    fig.colorbar(sm, ax=axes, label="token_len", location="right", shrink=0.9, pad=0.02)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    # With constrained_layout, avoid calling tight_layout to prevent overlap warnings
    plt.savefig(output_path, bbox_inches="tight")
    logging.info(f"Saved plot to {output_path}")


if __name__ == "__main__":
    main()
