#!/usr/bin/env python3
"""Plot edit_distance vs token_len scatter plots grouped by layout parameters."""

from __future__ import annotations

import argparse
import csv
import logging
from collections import defaultdict
from math import ceil
from pathlib import Path


LOGGER = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Group rows by (font_size, line_spacing, letter_spacing) and plot edit_distance vs token_len per group.")
    parser.add_argument("--input_csv", type=Path, default=Path("data_analysis/data4analysis.csv"), help="Path to the CSV file.")
    parser.add_argument("--output_jpg", type=Path, default=Path("data_analysis/edit_distance_vs_token_len_grouped.jpg"), help="Path to save the output JPG.")
    parser.add_argument("--group_cols", type=str, default="font_size,line_spacing,letter_spacing", help="Comma-separated group-by columns.")
    parser.add_argument("--x_col", type=str, default="token_len", help="Column name used as x-axis.")
    parser.add_argument("--y_col", type=str, default="edit_distance", help="Column name used as y-axis.")
    parser.add_argument("--ncols", type=int, default=6, help="Number of subplots per row.")
    parser.add_argument("--point_size", type=float, default=4.0, help="Scatter point size.")
    parser.add_argument("--alpha", type=float, default=0.35, help="Scatter alpha.")
    parser.add_argument("--max_points_per_group", type=int, default=0, help="Optional cap per group (0 means no cap).")
    return parser.parse_args()


def _safe_float(value: str) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _sort_group_key(group_key: tuple[str, ...]) -> tuple:
    def cast_one(value: str) -> float | str:
        number = _safe_float(value)
        return number if number is not None else value

    return tuple(cast_one(value) for value in group_key)


def read_grouped_points(csv_path: Path, group_cols: list[str], x_col: str, y_col: str, max_points_per_group: int) -> tuple[list[str], dict[tuple[str, ...], tuple[list[float], list[float]]]]:
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    grouped_x: dict[tuple[str, ...], list[float]] = defaultdict(list)
    grouped_y: dict[tuple[str, ...], list[float]] = defaultdict(list)

    skipped_non_numeric = 0
    skipped_missing = 0
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"CSV has no header: {csv_path}")
        for needed in [x_col, y_col, *group_cols]:
            if needed not in reader.fieldnames:
                raise KeyError(f"Missing column '{needed}' in CSV header: {reader.fieldnames}")
        for row in reader:
            if row is None:
                continue
            key = tuple((row.get(col) or "").strip() for col in group_cols)
            x_raw = (row.get(x_col) or "").strip()
            y_raw = (row.get(y_col) or "").strip()
            if not x_raw or not y_raw:
                skipped_missing += 1
                continue
            x_value = _safe_float(x_raw)
            y_value = _safe_float(y_raw)
            if x_value is None or y_value is None:
                skipped_non_numeric += 1
                continue
            if max_points_per_group > 0 and len(grouped_x[key]) >= max_points_per_group:
                continue
            grouped_x[key].append(x_value)
            grouped_y[key].append(y_value)

    groups: dict[tuple[str, ...], tuple[list[float], list[float]]] = {}
    for key in grouped_x:
        groups[key] = (grouped_x[key], grouped_y[key])

    LOGGER.info("Loaded %d groups from %s.", len(groups), csv_path)
    if skipped_missing:
        LOGGER.info("Skipped %d rows due to missing %s/%s.", skipped_missing, x_col, y_col)
    if skipped_non_numeric:
        LOGGER.info("Skipped %d rows due to non-numeric %s/%s.", skipped_non_numeric, x_col, y_col)
    return group_cols, groups


def plot_groups(group_cols: list[str], groups: dict[tuple[str, ...], tuple[list[float], list[float]]], x_col: str, y_col: str, ncols: int, point_size: float, alpha: float, output_jpg: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError("matplotlib is required for plotting; please `pip install matplotlib` in your environment.") from exc

    if not groups:
        raise RuntimeError("No data points available to plot (after filtering missing/non-numeric rows).")
    if ncols <= 0:
        raise ValueError("--ncols must be a positive integer.")

    group_keys = sorted(groups.keys(), key=_sort_group_key)
    n_groups = len(group_keys)
    nrows = ceil(n_groups / ncols)

    fig_width = max(12.0, ncols * 3.1)
    fig_height = max(4.0, nrows * 2.7)
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(fig_width, fig_height), squeeze=False)

    for idx, group_key in enumerate(group_keys):
        row_idx = idx // ncols
        col_idx = idx % ncols
        ax = axes[row_idx][col_idx]
        xs, ys = groups[group_key]
        ax.scatter(xs, ys, s=point_size, alpha=alpha, edgecolors="none")
        title_parts = [f"{name}={value}" for name, value in zip(group_cols, group_key)]
        ax.set_title(", ".join(title_parts), fontsize=8)
        if row_idx == nrows - 1:
            ax.set_xlabel(x_col, fontsize=8)
        if col_idx == 0:
            ax.set_ylabel(y_col, fontsize=8)
        ax.tick_params(axis="both", labelsize=7)
        ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.25)

    for idx in range(n_groups, nrows * ncols):
        axes[idx // ncols][idx % ncols].axis("off")

    fig.suptitle(f"{y_col} vs {x_col} grouped by {', '.join(group_cols)}", fontsize=12)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.96))
    output_jpg.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_jpg, dpi=200, format="jpg")
    plt.close(fig)
    LOGGER.info("Saved figure to %s", output_jpg)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args()
    group_cols = [col.strip() for col in args.group_cols.split(",") if col.strip()]
    resolved_group_cols, groups = read_grouped_points(args.input_csv, group_cols, args.x_col, args.y_col, args.max_points_per_group)
    plot_groups(resolved_group_cols, groups, args.x_col, args.y_col, args.ncols, args.point_size, args.alpha, args.output_jpg)


if __name__ == "__main__":
    main()
