"""Plot edit distance scaling law for font_size=28, line_spacing=6, letter_spacing=0."""

import argparse
import json
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.special import expit

DEFAULT_CSV_PATH = Path("data_analysis/data4analysis.csv")
DEFAULT_SUMMARY_DIR = Path("data_analysis/em_fiting_res/shared_parameters")
DEFAULT_OUTPUT_PATH = Path("data_analysis/scaling_law_plot.png")
DEFAULT_FONT_SIZE = 28
DEFAULT_LINE_SPACING = 6
DEFAULT_LETTER_SPACING = 0
DEFAULT_EPS = 1e-6
DEFAULT_PLOT_DPI = 200


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def parse_summary(summary: dict) -> dict[str, float]:
    gating_params = summary.get("gating_params", {})
    beta_params = summary.get("beta_params", {})
    low_ed = beta_params.get("low_ed", {})
    high_ed = beta_params.get("high_ed", {})
    alpha0 = float(low_ed["alpha"])
    beta0 = float(low_ed["beta"])
    alpha1 = float(high_ed["alpha"])
    beta1 = float(high_ed["beta"])
    mean0 = float(low_ed.get("mean", alpha0 / (alpha0 + beta0)))
    mean1 = float(high_ed.get("mean", alpha1 / (alpha1 + beta1)))
    return {
        "w0": float(gating_params["w0"]),
        "a": float(gating_params["a"]),
        "alpha": float(gating_params["alpha"]),
        "alpha0": alpha0,
        "beta0": beta0,
        "alpha1": alpha1,
        "beta1": beta1,
        "mean0": mean0,
        "mean1": mean1,
    }


def load_shared_parameters(summary_dir: Path, logger: logging.Logger) -> tuple[dict[str, float], list[int]]:
    summary_paths: list[Path] = []
    for item in summary_dir.iterdir():
        if item.is_dir():
            summary_path = item / "em_fit_summary.json"
            if summary_path.is_file():
                summary_paths.append(summary_path)
    if not summary_paths:
        raise FileNotFoundError(f"No em_fit_summary.json found under {summary_dir}")
    summary_paths = sorted(summary_paths, key=lambda path: path.parent.name)

    params: dict[str, float] | None = None
    resolutions: list[int] = []
    for summary_path in summary_paths:
        with summary_path.open("r", encoding="utf-8") as f:
            summary = json.load(f)
        resolution = summary.get("resolution")
        if resolution is None:
            try:
                resolution = int(summary_path.parent.name)
            except ValueError as exc:
                raise ValueError(f"Cannot determine resolution for {summary_path}") from exc
        resolutions.append(int(resolution))
        current_params = parse_summary(summary)
        if params is None:
            params = current_params
        else:
            mismatches = {
                key: (current_params[key], params[key])
                for key in (
                    "w0",
                    "a",
                    "alpha",
                    "alpha0",
                    "beta0",
                    "alpha1",
                    "beta1",
                )
                if not np.isclose(current_params[key], params[key], rtol=1e-5, atol=1e-8)
            }
            if mismatches:
                logger.warning("Parameter mismatch for resolution=%s: %s", resolution, mismatches)
    if params is None:
        raise ValueError("No parameters loaded from summaries.")
    return params, sorted(set(resolutions))


def load_plot_data(
    csv_path: Path,
    resolutions: list[int],
    font_size: int,
    line_spacing: int,
    letter_spacing: int,
    eps: float,
    logger: logging.Logger,
) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    required_cols = {
        "resolution",
        "text_len",
        "edit_distance",
        "character_width",
        "letter_spacing",
        "character_height",
        "line_spacing",
        "font_size",
    }
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    df = df[
        (df["font_size"] == font_size)
        & (df["line_spacing"] == line_spacing)
        & (df["letter_spacing"] == letter_spacing)
    ].copy()
    if resolutions:
        df = df[df["resolution"].isin(resolutions)].copy()
    if df.empty:
        raise ValueError("No rows found for requested font/spacing settings.")
    df = df.replace([np.inf, -np.inf], np.nan).dropna(
        subset=[
            "text_len",
            "edit_distance",
            "character_width",
            "letter_spacing",
            "character_height",
            "line_spacing",
            "font_size",
            "resolution",
        ]
    )
    text_len = df["text_len"].astype(float).to_numpy()
    char_width = df["character_width"].astype(float).to_numpy()
    letter_spacing_vals = df["letter_spacing"].astype(float).to_numpy()
    char_height = df["character_height"].astype(float).to_numpy()
    line_spacing_vals = df["line_spacing"].astype(float).to_numpy()
    resolution_vals = df["resolution"].astype(float).to_numpy()
    vision_token_num = (resolution_vals / 64.0) ** 2
    g_vals = text_len / vision_token_num
    numerator = (char_width + letter_spacing_vals) * text_len
    denominator = char_height + line_spacing_vals
    v_vals = np.divide(
        numerator,
        denominator,
        out=np.full_like(text_len, np.nan),
        where=denominator != 0,
    )
    v_vals = np.maximum(v_vals, float(eps))
    g_vals = np.maximum(g_vals, float(eps))
    log_v = np.log(v_vals)
    log_g = np.log(g_vals)
    mask = np.isfinite(log_v) & np.isfinite(log_g)
    df = df.loc[mask].copy()
    df["vision_token_num"] = vision_token_num[mask]
    df["logV"] = log_v[mask]
    df["logG"] = log_g[mask]
    logger.info("Loaded %d rows for plot.", len(df))
    return df


def plot_scaling_law(
    df: pd.DataFrame,
    params: dict[str, float],
    output_path: Path,
    plot_dpi: int,
) -> None:
    df = df.copy()
    df["Z"] = params["w0"] + params["a"] * df["logV"].astype(float) + params["alpha"] * df["logG"].astype(float)

    fig, ax = plt.subplots(figsize=(8, 6))
    cmap = plt.get_cmap("tab10")
    for idx, (resolution, group) in enumerate(df.groupby("resolution", sort=True)):
        color = cmap(idx % cmap.N)
        ax.scatter(
            group["Z"],
            group["edit_distance"],
            s=18,
            alpha=0.7,
            color=color,
            label=f"resolution={resolution}",
        )

    z_min = float(df["Z"].min())
    z_max = float(df["Z"].max())
    z_grid = np.linspace(z_min, z_max, 200)
    pi = expit(z_grid)
    expected_ed = (1.0 - pi) * params["mean0"] + pi * params["mean1"]
    ax.plot(z_grid, expected_ed, color="black", linewidth=2, label="E[ED|Z]")

    ax2 = ax.twinx()
    ax2.plot(z_grid, pi, color="tab:red", linestyle="--", linewidth=2, label="pi(Z)")
    ax2.set_ylabel("pi(Z)")
    ax2.set_ylim(0.0, 1.0)

    ax.set_xlabel("Z")
    ax.set_ylabel("Edit distance")
    ax.set_title("Scaling Law for font=28, line=6, letter=0")

    handles, labels = ax.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(handles + handles2, labels + labels2, loc="best", fontsize=8)

    fig.tight_layout()
    fig.savefig(output_path, dpi=int(plot_dpi))
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot scaling law for font_size=28, line_spacing=6, letter_spacing=0.")
    parser.add_argument("--csv_path", type=Path, default=DEFAULT_CSV_PATH, help="Path to input CSV file.")
    parser.add_argument("--summary_dir", type=Path, default=DEFAULT_SUMMARY_DIR, help="Directory containing shared-parameter summaries.")
    parser.add_argument("--output_path", type=Path, default=DEFAULT_OUTPUT_PATH, help="Path to save output plot.")
    parser.add_argument("--font_size", type=int, default=DEFAULT_FONT_SIZE, help="Font size to filter.")
    parser.add_argument("--line_spacing", type=int, default=DEFAULT_LINE_SPACING, help="Line spacing to filter.")
    parser.add_argument("--letter_spacing", type=int, default=DEFAULT_LETTER_SPACING, help="Letter spacing to filter.")
    parser.add_argument("--eps", type=float, default=DEFAULT_EPS, help="Epsilon for clipping V/G values.")
    parser.add_argument("--plot_dpi", type=int, default=DEFAULT_PLOT_DPI, help="DPI for saved plot.")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")
    logger = logging.getLogger("plot_28_6_0")

    params, resolutions = load_shared_parameters(args.summary_dir, logger)
    df = load_plot_data(
        args.csv_path,
        resolutions,
        args.font_size,
        args.line_spacing,
        args.letter_spacing,
        args.eps,
        logger,
    )

    ensure_dir(args.output_path.parent)
    plot_scaling_law(df, params, args.output_path, args.plot_dpi)
    logger.info("Saved plot to %s", args.output_path)


if __name__ == "__main__":
    main()
