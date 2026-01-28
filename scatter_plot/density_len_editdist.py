#!/usr/bin/env python3
"""Report patch density mean around L_break."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
LOGGER = logging.getLogger(__name__)

DEFAULT_PIECEWISE_PARAMS_PATH = Path("data_analysis/law_relation/piecewise_fit_params.csv")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Report patch density mean for token lengths near L_break.")
    parser.add_argument("--input_csv", type=Path, default=Path("data_analysis/data4analysis.csv"), help="CSV file containing patch_density, token_len, resolution, font_size, line_spacing, letter_spacing.")
    parser.add_argument("--piecewise_params_path", type=Path, default=DEFAULT_PIECEWISE_PARAMS_PATH, help="CSV containing L_break per group.")
    return parser.parse_args()


def build_piecewise_map(df: pd.DataFrame) -> dict[tuple[float, float, float, float], float]:
    required = ["resolution", "font_size", "line_spacing", "letter_spacing", "L_break"]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in piecewise params: {missing}")
    mapping = {}
    for _, row in df.iterrows():
        key = (
            float(row["resolution"]),
            float(row["font_size"]),
            float(row["line_spacing"]),
            float(row["letter_spacing"]),
        )
        mapping[key] = float(row["L_break"])
    return mapping


def report_patch_density_near_l_break(
    df: pd.DataFrame,
    piecewise_map: dict[tuple[float, float, float, float], float],
    window: float = 10.0,
) -> None:
    group_cols = ["resolution", "font_size", "line_spacing", "letter_spacing"]
    print("resolution,font_size,line_spacing,letter_spacing,patch_density_mean")
    for key, group_df in df.groupby(group_cols, observed=True):
        key_floats = tuple(float(val) for val in key)
        l_break_val = piecewise_map.get(key_floats)
        if l_break_val is None:
            LOGGER.warning("Missing L_break for group %s; skipping.", key_floats)
            continue
        if not np.isfinite(l_break_val):
            LOGGER.warning("Invalid L_break for group %s; skipping.", key_floats)
            continue
        token_len = group_df["token_len"].to_numpy(dtype=float)
        patch_density = group_df["patch_density"].to_numpy(dtype=float)
        valid_mask = np.isfinite(token_len) & np.isfinite(patch_density)
        token_len = token_len[valid_mask]
        patch_density = patch_density[valid_mask]
        if token_len.size == 0:
            mean_val = float("nan")
        else:
            window_mask = (token_len >= l_break_val - window) & (token_len <= l_break_val + window)
            selected = patch_density[window_mask]
            mean_val = float(np.mean(selected)) if selected.size else float("nan")
        print(f"{key_floats[0]},{key_floats[1]},{key_floats[2]},{key_floats[3]},{mean_val}")


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.input_csv)

    required = ["patch_density", "token_len", "resolution", "font_size", "line_spacing", "letter_spacing"]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    piecewise_df = pd.read_csv(args.piecewise_params_path)
    piecewise_map = build_piecewise_map(piecewise_df)
    report_patch_density_near_l_break(df, piecewise_map)


if __name__ == "__main__":
    main()
