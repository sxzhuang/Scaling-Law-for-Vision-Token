import argparse
import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot b2 values (y) against resolution (x) from piecewise_fit_params.csv.")
    parser.add_argument("--input_csv", type=Path, default=Path("data_analysis/law_relation_alltype/piecewise_fit_params.csv"), help="Path to piecewise_fit_params.csv.")
    parser.add_argument("--lr_csv", type=Path, default=Path("data_analysis/data_preprocess_res_alltype/data_preprocess_high_variance.csv"), help="Path to data_preprocess_high_variance.csv.")
    parser.add_argument("--output", type=Path, default=Path("scatter_plot/b2_vs_resolution.png"), help="Output PNG path for the plot.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    if not args.input_csv.exists():
        raise FileNotFoundError(f"CSV file not found: {args.input_csv}")

    df = pd.read_csv(args.input_csv)
    required_cols = ["resolution", "b2"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df["resolution"] = df["resolution"].astype(float)
    df["b2"] = df["b2"].astype(float)
    df = df.dropna(subset=["resolution", "b2"])

    if not args.lr_csv.exists():
        raise FileNotFoundError(f"CSV file not found: {args.lr_csv}")

    lr_df = pd.read_csv(args.lr_csv)
    lr_required = ["resolution", "L", "R"]
    lr_missing = [c for c in lr_required if c not in lr_df.columns]
    if lr_missing:
        raise ValueError(f"Missing required columns: {lr_missing}")
    lr_df["resolution"] = lr_df["resolution"].astype(float)
    lr_df["L"] = lr_df["L"].astype(float)
    lr_df["R"] = lr_df["R"].astype(float)
    lr_df = lr_df.dropna(subset=["resolution", "L", "R"])

    fig, axes = plt.subplots(1, 3, figsize=(16, 5), dpi=200, constrained_layout=True)

    axes[0].scatter(df["resolution"].to_numpy(), df["b2"].to_numpy(), s=18, alpha=0.75, edgecolors="none")
    axes[0].set_xlabel("resolution")
    axes[0].set_ylabel("b2")
    axes[0].set_title("b2 vs Resolution")
    axes[0].grid(True, linestyle=":", linewidth=0.5, alpha=0.6)

    axes[1].scatter(lr_df["resolution"].to_numpy(), lr_df["L"].to_numpy(), s=18, alpha=0.75, edgecolors="none")
    axes[1].set_xlabel("resolution")
    axes[1].set_ylabel("L")
    axes[1].set_title("L vs Resolution")
    axes[1].grid(True, linestyle=":", linewidth=0.5, alpha=0.6)

    axes[2].scatter(lr_df["resolution"].to_numpy(), lr_df["R"].to_numpy(), s=18, alpha=0.75, edgecolors="none")
    axes[2].set_xlabel("resolution")
    axes[2].set_ylabel("R")
    axes[2].set_title("R vs Resolution")
    axes[2].grid(True, linestyle=":", linewidth=0.5, alpha=0.6)

    xticks = sorted(
        set(df["resolution"].dropna().unique().tolist() + lr_df["resolution"].dropna().unique().tolist())
    )
    if xticks:
        for ax in axes:
            ax.set_xticks(xticks)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(args.output, bbox_inches="tight")
    logging.info("Saved plot to %s", args.output)


if __name__ == "__main__":
    main()
