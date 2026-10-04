"""Plot edit distance scaling law for font_size=28, line_spacing=6, letter_spacing=0 with per-resolution w0."""

import argparse
import json
import logging
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
from scipy.special import expit

DEFAULT_CSV_PATH = Path("data_analysis/data4analysis.csv")
DEFAULT_SUMMARY_DIR = Path("data_analysis/em_fiting_res/shared_exceptw")
DEFAULT_OUTPUT_PATH = Path("eval_results/scaling_law_plot.pdf")
DEFAULT_FONT_SIZE = 28
DEFAULT_LINE_SPACING = 6
DEFAULT_LETTER_SPACING = 0
DEFAULT_EPS = 1e-6
DEFAULT_PLOT_DPI = 200
DEFAULT_PLOT_WIDTH = 3.25
DEFAULT_PLOT_HEIGHT = DEFAULT_PLOT_WIDTH * 0.75
DEFAULT_PLOT_FONT_SIZE = 7.5
DEFAULT_Z_MIN = -8.0
DEFAULT_Z_MAX = 13.0
DEFAULT_Z_PADDING = 1.0
DEFAULT_BOUNDARY_ZS = (-2.0, 3.4)
Z_BIN_WIDTH = 1.0
BOUNDARY_BIN_SAMPLE_RATIO = 0.5
OTHER_BIN_SAMPLE_RATIO = 0.25
RANDOM_SEED = 0
SCATTER_ALPHA = 1.0
SCATTER_SIZE = 1.0
SCATTER_EDGE_COLOR = "none"
SCATTER_LINEWIDTH = 0.0
LINE_MARKER_LINEWIDTH = 0.45
BOUNDARY_COLOR = "#9E9E9E"
CURVE_COLOR = "#D62728"
LEGEND_CURVE_LINEWIDTH = 1.2
LEGEND_BOUNDARY_LINEWIDTH = 1.2


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


def load_parameters(summary_dir: Path, logger: logging.Logger) -> tuple[dict[int, dict[str, float]], dict[str, float]]:
    summary_paths: list[Path] = []
    for item in summary_dir.iterdir():
        if item.is_dir():
            summary_path = item / "em_fit_summary.json"
            if summary_path.is_file():
                summary_paths.append(summary_path)
    if not summary_paths:
        raise FileNotFoundError(f"No em_fit_summary.json found under {summary_dir}")
    summary_paths = sorted(summary_paths, key=lambda path: path.parent.name)

    per_resolution_params: dict[int, dict[str, float]] = {}
    shared_params: dict[str, float] | None = None
    for summary_path in summary_paths:
        with summary_path.open("r", encoding="utf-8") as f:
            summary = json.load(f)
        resolution = summary.get("resolution")
        if resolution is None:
            try:
                resolution = int(summary_path.parent.name)
            except ValueError as exc:
                raise ValueError(f"Cannot determine resolution for {summary_path}") from exc
        current_params = parse_summary(summary)
        per_resolution_params[int(resolution)] = current_params
        if shared_params is None:
            shared_params = current_params
        else:
            mismatches = {
                key: (current_params[key], shared_params[key])
                for key in (
                    "a",
                    "alpha",
                    "alpha0",
                    "beta0",
                    "alpha1",
                    "beta1",
                )
                if not np.isclose(current_params[key], shared_params[key], rtol=1e-5, atol=1e-8)
            }
            if mismatches:
                logger.warning("Shared parameter mismatch for resolution=%s: %s", resolution, mismatches)

    if shared_params is None:
        raise ValueError("No parameters loaded from summaries.")
    shared_only = {
        "a": shared_params["a"],
        "alpha": shared_params["alpha"],
        "mean0": shared_params["mean0"],
        "mean1": shared_params["mean1"],
    }
    return per_resolution_params, shared_only


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


def boundary_neighbor_bins(boundary_zs: tuple[float, ...], bin_width: float) -> set[int]:
    bins: set[int] = set()
    for boundary_z in boundary_zs:
        right_bin = int(np.floor(boundary_z / bin_width))
        bins.add(right_bin - 1)
        bins.add(right_bin)
    return bins


def sample_by_z_bin(group: pd.DataFrame, resolution: int) -> pd.DataFrame:
    if group.empty:
        return group
    boundary_bins = boundary_neighbor_bins(DEFAULT_BOUNDARY_ZS, Z_BIN_WIDTH)
    sampled_groups = []
    group = group.copy()
    group["_z_bin"] = np.floor(group["Z"].astype(float) / Z_BIN_WIDTH).astype(int)
    for z_bin, bin_group in group.groupby("_z_bin", sort=True):
        sample_ratio = BOUNDARY_BIN_SAMPLE_RATIO if int(z_bin) in boundary_bins else OTHER_BIN_SAMPLE_RATIO
        if sample_ratio >= 1.0:
            sampled_bin = bin_group
        else:
            random_state = RANDOM_SEED + int(resolution) * 1009 + (int(z_bin) + 10000)
            sampled_bin = bin_group.sample(frac=sample_ratio, random_state=random_state)
        sampled_groups.append(sampled_bin)
    sampled = pd.concat(sampled_groups, axis=0).sort_index()
    return sampled.drop(columns=["_z_bin"])


def plot_scaling_law(
    df: pd.DataFrame,
    per_resolution_params: dict[int, dict[str, float]],
    shared_params: dict[str, float],
    output_path: Path,
    plot_dpi: int,
    scatter_ratio: float,
) -> None:
    plt.rcParams.update(
        {
            "font.size": DEFAULT_PLOT_FONT_SIZE,
            "axes.titlesize": DEFAULT_PLOT_FONT_SIZE,
            "axes.labelsize": DEFAULT_PLOT_FONT_SIZE,
            "xtick.labelsize": DEFAULT_PLOT_FONT_SIZE,
            "ytick.labelsize": DEFAULT_PLOT_FONT_SIZE,
            "legend.fontsize": DEFAULT_PLOT_FONT_SIZE,
        }
    )
    df = df.copy()
    df["w0"] = df["resolution"].map(lambda res: per_resolution_params[int(res)]["w0"])
    df["Z"] = df["w0"] + shared_params["a"] * df["logV"].astype(float) + shared_params["alpha"] * df["logG"].astype(float)

    fig, ax = plt.subplots(figsize=(DEFAULT_PLOT_WIDTH, DEFAULT_PLOT_HEIGHT))
    colors = ["#4F86B7", "#D48BC4", "#F2A33A", "#5FA95F", "#C96A6A", "#78BAC1"]
    markers = ["o", "^", "+", "v", "<", "P", "X", "*"]
    grouped = list(df.groupby("resolution", sort=True))
    for idx, (resolution, group) in enumerate(grouped):
        group = sample_by_z_bin(group, int(resolution))
        if scatter_ratio < 1.0:
            group = group.sample(frac=scatter_ratio, random_state=RANDOM_SEED)
        color = colors[idx % len(colors)]
        marker = markers[idx % len(markers)]
        is_line_marker = marker in {"+", "x", "1", "2", "3", "4", "|", "_"}
        scatter_kwargs = {
            "s": SCATTER_SIZE,
            "alpha": SCATTER_ALPHA,
            "color": color,
            "marker": marker,
            "linewidth": LINE_MARKER_LINEWIDTH if is_line_marker else SCATTER_LINEWIDTH,
            "label": f"R = {resolution}",
        }
        if not is_line_marker:
            scatter_kwargs["edgecolors"] = SCATTER_EDGE_COLOR
        ax.scatter(group["Z"], group["edit_distance"], **scatter_kwargs)

    z_min = DEFAULT_Z_MIN - DEFAULT_Z_PADDING
    z_max = DEFAULT_Z_MAX + DEFAULT_Z_PADDING
    z_grid = np.linspace(z_min, z_max, 200)
    pi = expit(z_grid)
    expected_ed = (1.0 - pi) * shared_params["mean0"] + pi * shared_params["mean1"]
    ax.plot(z_grid, expected_ed, color=CURVE_COLOR, linewidth=2, label="E[D|Z]")

    for boundary_z in DEFAULT_BOUNDARY_ZS:
        ax.axvline(
            boundary_z,
            color=BOUNDARY_COLOR,
            linestyle="--",
            linewidth=0.7,
            zorder=1,
        )
    ax.set_xlim(z_min, z_max)
    ed_min = float(df["edit_distance"].min())
    ed_max = float(df["edit_distance"].max())
    ed_range = ed_max - ed_min
    ed_margin = ed_range * 0.05 if ed_range > 0 else max(abs(ed_max) * 0.05, 0.1)
    left_bottom = min(ed_min, 0.0) - ed_margin
    left_top = ed_max + ed_margin
    if left_top < 1.0:
        left_top = 1.0 + ed_margin
    if left_bottom >= 0:
        left_bottom = -ed_margin
    ax.set_ylim(left_bottom, left_top)
    right_ticks = [0.0, 0.25, 0.5, 0.75, 1.0]
    ax.set_yticks(right_ticks)
    ax.set_yticklabels(["0.00", "0.25", "0.50", "0.75", "1.00"])

    ax.set_xlabel("Capacity pressure")
    ax.set_ylabel("Edit distance")
    ax.text(
        1.01,
        -0.055,
        "Z",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=DEFAULT_PLOT_FONT_SIZE,
        clip_on=False,
    )
    ax.tick_params(axis="both", which="both", length=1.5, width=0.5)
    ax.grid(False)

    handles, labels = ax.get_legend_handles_labels()
    legend_items = [
        Line2D([0], [0], color=CURVE_COLOR, linewidth=LEGEND_CURVE_LINEWIDTH)
        if label == "E[D|Z]"
        else handle
        for handle, label in zip(handles, labels)
    ]
    boundary_handle = Line2D(
        [0],
        [0],
        color=BOUNDARY_COLOR,
        linestyle="--",
        linewidth=LEGEND_BOUNDARY_LINEWIDTH,
    )
    legend_items.append(boundary_handle)
    legend_labels = labels + ["Boundary"]
    scatter_legend_size = 6
    for handle in legend_items:
        if hasattr(handle, "set_sizes"):
            handle.set_sizes([scatter_legend_size])

    fig.legend(
        legend_items,
        legend_labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=min(3, len(legend_labels)),
        fontsize=DEFAULT_PLOT_FONT_SIZE,
        handletextpad=0.3,
        columnspacing=0.6,
        handlelength=1.8,
        frameon=False,
    )

    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.85))
    fig.savefig(output_path, dpi=int(plot_dpi))
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot scaling law for font_size=28, line_spacing=6, letter_spacing=0 with per-resolution w0.")
    parser.add_argument("--csv_path", type=Path, default=DEFAULT_CSV_PATH, help="Path to input CSV file.")
    parser.add_argument("--summary_dir", type=Path, default=DEFAULT_SUMMARY_DIR, help="Directory containing shared-parameter summaries.")
    parser.add_argument("--output_path", type=Path, default=DEFAULT_OUTPUT_PATH, help="Path to save output plot.")
    parser.add_argument("--font_size", type=int, default=DEFAULT_FONT_SIZE, help="Font size to filter.")
    parser.add_argument("--line_spacing", type=int, default=DEFAULT_LINE_SPACING, help="Line spacing to filter.")
    parser.add_argument("--letter_spacing", type=int, default=DEFAULT_LETTER_SPACING, help="Letter spacing to filter.")
    parser.add_argument("--eps", type=float, default=DEFAULT_EPS, help="Epsilon for clipping V/G values.")
    parser.add_argument("--plot_dpi", type=int, default=DEFAULT_PLOT_DPI, help="DPI for saved plot.")
    parser.add_argument("--scatter_ratio", type=float, default=1.0, help="Fraction of points to plot per resolution (0 < ratio <= 1).")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")
    logger = logging.getLogger("plot_28_6_0_exceptw")

    per_resolution_params, shared_params = load_parameters(args.summary_dir, logger)
    resolutions = sorted(per_resolution_params.keys())
    if not (0.0 < args.scatter_ratio <= 1.0):
        raise ValueError("scatter_ratio must be in (0, 1].")
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
    plot_scaling_law(df, per_resolution_params, shared_params, args.output_path, args.plot_dpi, args.scatter_ratio)
    logger.info("Saved plot to %s", args.output_path)


if __name__ == "__main__":
    main()
