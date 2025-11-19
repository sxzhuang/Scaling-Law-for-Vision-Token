import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable


RESOLUTIONS = [512, 640, 1024, 1280]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot resolution (x) vs edit_dist (y), connect same-id points across resolutions, color by token_len.")
    parser.add_argument("--input_dir", type=Path, default=Path("eval_results/pride_square"), help="Directory containing 512/640/1024/1280 subfolders with dpsk_eval_metric.json.")
    parser.add_argument("--output", type=Path, default=Path("eval_results/pride_square/exp3_resolution_vs_editdist.png"), help="Output PNG path for the plot.")
    return parser.parse_args()


def load_metrics(json_path: Path) -> List[dict]:
    with json_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    input_dir: Path = args.input_dir
    output_path: Path = args.output

    # Map: id -> list of (resolution, edit_dist, token_len)
    id_series: Dict[str, List[Tuple[int, float, int]]] = {}
    all_tokens: List[int] = []
    all_edits: List[float] = []

    for res in RESOLUTIONS:
        json_file = input_dir / str(res) / "dpsk_eval_metric.json"
        if not json_file.exists():
            logging.warning(f"Missing file: {json_file}")
            continue
        records = load_metrics(json_file)
        for rec in records:
            sid = rec.get("id", "")
            edit_dist = float(rec.get("edit_dist", 0.0))
            token_len = int(rec.get("text_token_len", 0))
            id_series.setdefault(sid, []).append((res, edit_dist, token_len))
            all_tokens.append(token_len)
            all_edits.append(edit_dist)

    if not id_series:
        raise FileNotFoundError(f"No metrics found under {input_dir}")

    # Prepare color normalization by token_len
    t_min, t_max = (min(all_tokens), max(all_tokens)) if all_tokens else (0, 1)
    cmap = plt.get_cmap("viridis")
    norm = Normalize(vmin=t_min, vmax=t_max)

    fig, ax = plt.subplots(figsize=(10, 7), dpi=200, constrained_layout=True)

    # Scatter all points first for consistent colorbar
    xs_all, ys_all, cs_all = [], [], []
    for sid, series in id_series.items():
        for (res, e, t) in series:
            xs_all.append(res)
            ys_all.append(e)
            cs_all.append(t)

    if xs_all:
        sc = ax.scatter(xs_all, ys_all, c=cs_all, cmap=cmap, norm=norm, s=18, alpha=0.9, edgecolors="none")
        sm = ScalarMappable(norm=norm, cmap=cmap)
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax, pad=0.02)
        cbar.set_label("text_token_len")

    # Connect points for each id across resolutions
    for sid, series in id_series.items():
        series_sorted = sorted(series, key=lambda t: t[0])
        x_line = [t[0] for t in series_sorted]
        y_line = [t[1] for t in series_sorted]
        ax.plot(x_line, y_line, color="#666666", linewidth=0.9, alpha=0.6)

    ax.set_xlabel("resolution")
    ax.set_ylabel("edit_dist")
    ax.set_title("Resolution vs Edit Distance (color by text_token_len)")
    ax.set_xticks(RESOLUTIONS)
    if all_edits:
        ax.set_ylim(min(all_edits), max(all_edits))
    ax.grid(True, linestyle=":", linewidth=0.5, alpha=0.6)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, bbox_inches="tight")
    logging.info(f"Saved plot to {output_path}")


if __name__ == "__main__":
    main()

