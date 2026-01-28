#!/usr/bin/env python3
"""Fit per-group k in patch_density = k * text_len * resolution^-2."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd


logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
LOGGER = logging.getLogger(__name__)

DEFAULT_INPUT_CSV = Path("data_analysis/data4analysis.csv")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fit k in patch_density = k * text_len * resolution^-2 per group.")
    parser.add_argument("--input_csv", type=Path, default=DEFAULT_INPUT_CSV, help="Input CSV containing patch_density and text_len.")
    return parser.parse_args()


def fit_group_k(group_df: pd.DataFrame) -> tuple[float, int]:
    token_len = group_df["text_len"].to_numpy(dtype=float)
    patch_density = group_df["patch_density"].to_numpy(dtype=float)
    resolution = group_df["resolution"].to_numpy(dtype=float)
    valid_mask = np.isfinite(token_len) & np.isfinite(patch_density) & np.isfinite(resolution)
    valid_mask &= token_len > 0.0
    token_len = token_len[valid_mask]
    patch_density = patch_density[valid_mask]
    resolution = resolution[valid_mask]
    if token_len.size == 0:
        return float("nan"), 0
    x = token_len / (resolution ** 2)
    denom = float(np.sum(x ** 2))
    if denom <= 0.0:
        return float("nan"), int(token_len.size)
    k = float(np.sum(x * patch_density) / denom)
    return k, int(token_len.size)


def main() -> None:
    args = parse_args()
    if not args.input_csv.exists():
        raise FileNotFoundError(f"Input CSV not found: {args.input_csv}")
    df = pd.read_csv(args.input_csv)
    required_cols = ["patch_density", "text_len", "resolution", "font_size", "line_spacing", "letter_spacing"]
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    group_cols = ["resolution", "font_size", "line_spacing", "letter_spacing"]
    print("resolution,font_size,line_spacing,letter_spacing,k_r2")
    for key, group_df in df.groupby(group_cols, observed=True):
        k, n_points = fit_group_k(group_df)
        resolution = float(key[0])
        k_r2 = float("nan") if not np.isfinite(k) else float(k / (resolution ** 2))
        print(f"{resolution},{float(key[1])},{float(key[2])},{float(key[3])},{k_r2}")
    LOGGER.info("Processed %d groups", df.groupby(group_cols, observed=True).ngroups)


if __name__ == "__main__":
    main()
