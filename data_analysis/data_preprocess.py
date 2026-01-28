"""Preprocess OCR edit-distance data with L/R-based filtering."""

import argparse
import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

BIN_WIDTH = 100
Q_FOR_THRESHOLD = 0.90
Q_LOW_FOR_VARIANCE = 0.10
OUTPUT_DIR = Path("data_analysis/data_preprocess_res")
FILTER_IQR_SCALE = 0.4
CHARACTER_SIZE_BY_FONT = {
    20.0: (10.0369, 19.0),
    28.0: (14.0533, 26.0),
    36.0: (18.0624, 33.0),
}


@dataclass
class IntervalResult:
    L: float
    R: float
    bin_stats: pd.DataFrame
    var_threshold: float


def compute_bin_quantiles(
    text_len: np.ndarray,
    edit_distance: np.ndarray,
    bin_width: int = BIN_WIDTH,
    q: float = Q_FOR_THRESHOLD,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return bin left edges, q_high, q_low, q_diff, and counts."""
    if text_len.size == 0:
        return (
            np.array([], dtype=float),
            np.array([], dtype=float),
            np.array([], dtype=float),
            np.array([], dtype=float),
            np.array([], dtype=int),
        )

    token_vals = text_len.astype(float)
    edit_vals = edit_distance.astype(float)
    bin_idx = np.floor_divide(token_vals.astype(int), int(bin_width)).astype(int)
    df_tmp = pd.DataFrame({"bin": bin_idx, "E": edit_vals})

    q_high = df_tmp.groupby("bin")["E"].quantile(q)
    q_low = df_tmp.groupby("bin")["E"].quantile(Q_LOW_FOR_VARIANCE)
    if q_high.empty:
        return (
            np.array([], dtype=float),
            np.array([], dtype=float),
            np.array([], dtype=float),
            np.array([], dtype=float),
            np.array([], dtype=int),
        )

    bmin = int(q_high.index.min())
    bmax = int(q_high.index.max())
    full_bins = pd.Index(range(bmin, bmax + 1), name="bin")
    q_high = q_high.reindex(full_bins).fillna(0.0)
    q_low = q_low.reindex(full_bins).fillna(0.0)
    count_by_bin = df_tmp.groupby("bin")["E"].size().reindex(full_bins).fillna(0).astype(int)

    left_edges = full_bins.values.astype(float) * float(bin_width)
    q_high_values = q_high.values.astype(float)
    q_low_values = q_low.values.astype(float)
    q_diff = q_high_values - q_low_values
    return left_edges, q_high_values, q_low_values, q_diff, count_by_bin.values.astype(int)


def detect_high_variance_interval(
    df: pd.DataFrame,
    token_col: str = "text_len",
    y_col: str = "edit_distance",
    bin_width: int = BIN_WIDTH,
    left_iqr_thr: float = 0.1,
    right_iqr_thr: float = 0.2,
    peak_iqr_thr: float = 0.5,
) -> IntervalResult:
    """Detect L/R using fixed-width bins and quantile-based thresholds."""
    d = df[[token_col, y_col]].dropna().copy()
    d = d[(d[token_col] > 0) & np.isfinite(d[y_col])]
    if len(d) < 10:
        raise ValueError("Not enough valid data points for interval detection.")

    left_edges, q_high, q_low, q_diff, counts = compute_bin_quantiles(
        text_len=d[token_col].to_numpy(),
        edit_distance=d[y_col].to_numpy(),
        bin_width=bin_width,
        q=Q_FOR_THRESHOLD,
    )
    if q_high.size == 0:
        empty_stats = pd.DataFrame(columns=["token_median", "token_min", "token_max", "count", "y_iqr"])
        return IntervalResult(L=float("nan"), R=float("nan"), bin_stats=empty_stats, var_threshold=float(peak_iqr_thr))

    token_min = left_edges
    token_max = left_edges + float(bin_width)
    token_median = left_edges + 0.5 * float(bin_width)
    stats = pd.DataFrame(
        {
            "token_median": token_median,
            "token_min": token_min,
            "token_max": token_max,
            "count": counts,
            "q_high": q_high,
            "q_low": q_low,
            "y_iqr": q_diff,
        }
    )

    avg_y_iqr = stats["y_iqr"].rolling(window=5, center=True, min_periods=1).mean().to_numpy()
    avg_y_iqr = np.nan_to_num(avg_y_iqr, nan=0.0, posinf=0.0, neginf=0.0)
    if avg_y_iqr.size == 0:
        return IntervalResult(L=float("nan"), R=float("nan"), bin_stats=stats, var_threshold=float(peak_iqr_thr))

    peak_idx = int(np.argmax(avg_y_iqr))
    peak_val = float(avg_y_iqr[peak_idx])

    left_boundary = float(stats.loc[0, "token_max"])
    search_start = peak_idx
    high_iqr_thr = 0.15
    while True:
        found = False
        for i in range(search_start, -1, -1):
            start = i - 2
            if start < 0:
                continue
            if np.all(avg_y_iqr[start:i + 1] < left_iqr_thr):
                left_boundary = float(stats.loc[i, "token_max"])
                found = True
                break
        if not found:
            break
        left_high_idx = np.where(avg_y_iqr[:i] > high_iqr_thr)[0]
        if left_high_idx.size == 0:
            break
        search_start = int(left_high_idx[-1])

    right_boundary = float(stats.loc[len(avg_y_iqr) - 1, "token_median"])
    for i in range(peak_idx, len(avg_y_iqr)):
        end = i + 3
        if end > len(avg_y_iqr):
            continue
        if np.all(avg_y_iqr[i:end] < right_iqr_thr):
            right_boundary = float(stats.loc[i, "token_median"])
            break

    return IntervalResult(L=left_boundary, R=right_boundary, bin_stats=stats, var_threshold=float(peak_iqr_thr))


def filter_by_lr_bins(
    df: pd.DataFrame,
    L: float,
    R: float,
    token_col: str = "text_len",
    y_col: str = "edit_distance",
    bin_width: int = BIN_WIDTH,
    scale: float = FILTER_IQR_SCALE,
) -> pd.DataFrame:
    """Filter bins within [L, R] and drop all bins to the right of R."""
    if not np.isfinite(L) or not np.isfinite(R):
        return df.copy()

    left_edges, _, q_low, q_diff, _ = compute_bin_quantiles(
        text_len=df[token_col].to_numpy(),
        edit_distance=df[y_col].to_numpy(),
        bin_width=bin_width,
        q=Q_FOR_THRESHOLD,
    )
    if left_edges.size == 0:
        return df.copy()

    thresholds = {int(edge): float(q_low[i] + scale * q_diff[i]) for i, edge in enumerate(left_edges)}

    d = df.copy()
    bin_left = (d[token_col].astype(int) // int(bin_width)) * int(bin_width)
    thr_series = bin_left.map(lambda b: thresholds.get(int(b), float("inf")))

    right_mask = bin_left > R
    mid_mask = (bin_left >= L) & (bin_left <= R)
    keep = ~right_mask
    keep &= ~(mid_mask & (d[y_col].astype(float) > thr_series))
    return d.loc[keep].copy()


def drop_top_ed_points(df: pd.DataFrame, count: int, y_col: str = "edit_distance") -> pd.DataFrame:
    """Drop the top-N points by edit distance within a group."""
    if count <= 0 or df.empty:
        return df.copy()
    n_drop = min(count, len(df))
    return df.sort_values(y_col, ascending=True).iloc[:-n_drop].copy()


def add_information_density(df: pd.DataFrame) -> pd.DataFrame:
    if "patch_density" not in df.columns:
        return df.copy()
    info = pd.Series(np.nan, index=df.index, dtype=float)
    for font_size, (char_width, char_height) in CHARACTER_SIZE_BY_FONT.items():
        mask = df["font_size"].astype(float) == float(font_size)
        if not mask.any():
            continue
        denom = (char_width + df.loc[mask, "letter_spacing"].astype(float)) * (
            char_height + df.loc[mask, "line_spacing"].astype(float)
        )
        base = (char_width * char_height) / denom
        info.loc[mask] = base * df.loc[mask, "patch_density"].astype(float)
    out = df.copy()
    out["information_density"] = info
    return out


def format_resolution_tag(resolution: float) -> str:
    if abs(resolution - round(resolution)) < 1e-6:
        return str(int(round(resolution)))
    return str(resolution).replace(".", "p")


def group_by_resolution(df: pd.DataFrame) -> dict[float, dict[tuple[float, float, float], pd.DataFrame]]:
    grouped: dict[float, dict[tuple[float, float, float], pd.DataFrame]] = {}
    for (resolution, font_size, line_spacing, letter_spacing), group_df in df.groupby(
        ["resolution", "font_size", "line_spacing", "letter_spacing"], observed=True
    ):
        key = (float(font_size), float(line_spacing), float(letter_spacing))
        grouped.setdefault(float(resolution), {})[key] = group_df.copy()
    return grouped


def plot_by_resolution(
    groups: dict[float, dict[tuple[float, float, float], pd.DataFrame]],
    out_dir: Path,
    ncols: int,
    prefix: str,
    lr_map: Optional[dict[tuple[float, float, float, float], tuple[float, float]]] = None,
    x_max_by_resolution: Optional[dict[float, float]] = None,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for resolution, res_groups in sorted(groups.items()):
        items = sorted(res_groups.items(), key=lambda item: item[0])
        count = len(items)
        ncols = max(1, ncols)
        nrows = max(1, math.ceil(count / ncols))
        fig, axes = plt.subplots(nrows, ncols, figsize=(4.0 * ncols, 3.2 * nrows), squeeze=False)
        axes_flat = axes.ravel()
        x_max = None
        if x_max_by_resolution is not None:
            x_max = x_max_by_resolution.get(float(resolution))
        for idx, (params, group_df) in enumerate(items):
            ax = axes_flat[idx]
            text_lens = group_df["text_len"].astype(float).to_numpy()
            edit_distances = group_df["edit_distance"].astype(float).to_numpy()
            if text_lens.size > 0:
                sorted_idx = np.argsort(text_lens)
                text_lens = text_lens[sorted_idx]
                edit_distances = edit_distances[sorted_idx]
            ax.scatter(text_lens, edit_distances, s=12, alpha=0.6, color="steelblue")
            left_edges, _, _, q_diff, _ = compute_bin_quantiles(
                text_len=group_df["text_len"].to_numpy(),
                edit_distance=group_df["edit_distance"].to_numpy(),
                bin_width=BIN_WIDTH,
                q=Q_FOR_THRESHOLD,
            )
            if left_edges.size > 0:
                avg_y_iqr = pd.Series(q_diff).rolling(window=5, center=True, min_periods=1).mean().to_numpy()
                ax.plot(left_edges, avg_y_iqr, color="green", linewidth=1)
            if lr_map is not None:
                key = (float(resolution), float(params[0]), float(params[1]), float(params[2]))
                if key in lr_map:
                    l_val, r_val = lr_map[key]
                    ax.axvline(l_val, color="darkorange", linestyle="--", linewidth=1.2)
                    ax.axvline(r_val, color="darkorange", linestyle="--", linewidth=1.2)
            ax.set_title(f"fs={params[0]}, ls={params[1]}, cs={params[2]}")
            ax.set_xlabel("text_len")
            ax.set_ylabel("edit_distance")
            ax.set_ylim(0.0, 1.0)
            if x_max is not None and np.isfinite(x_max):
                ax.set_xlim(0.0, x_max)
        for ax in axes_flat[count:]:
            ax.axis("off")
        fig.suptitle(f"resolution={resolution}")
        fig.tight_layout()
        fig_path = out_dir / f"{prefix}_resolution_{format_resolution_tag(resolution)}.png"
        fig.savefig(fig_path, dpi=300)
        plt.close(fig)


def plot_pd_by_resolution(
    groups: dict[float, dict[tuple[float, float, float], pd.DataFrame]],
    out_dir: Path,
    ncols: int,
    prefix: str,
    lr_map: Optional[dict[tuple[float, float, float, float], tuple[float, float]]] = None,
    x_max_by_resolution: Optional[dict[float, float]] = None,
    raw_groups: Optional[dict[float, dict[tuple[float, float, float], pd.DataFrame]]] = None,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for resolution, res_groups in sorted(groups.items()):
        items = sorted(res_groups.items(), key=lambda item: item[0])
        count = len(items)
        ncols = max(1, ncols)
        nrows = max(1, math.ceil(count / ncols))
        fig, axes = plt.subplots(nrows, ncols, figsize=(4.0 * ncols, 3.2 * nrows), squeeze=False)
        axes_flat = axes.ravel()
        x_max = None
        if x_max_by_resolution is not None:
            x_max = x_max_by_resolution.get(float(resolution))
        for idx, (params, group_df) in enumerate(items):
            ax = axes_flat[idx]
            patch_density = group_df["patch_density"].astype(float).to_numpy()
            edit_distances = group_df["edit_distance"].astype(float).to_numpy()
            if patch_density.size > 0:
                sorted_idx = np.argsort(patch_density)
                patch_density = patch_density[sorted_idx]
                edit_distances = edit_distances[sorted_idx]
            ax.scatter(patch_density, edit_distances, s=12, alpha=0.6, color="steelblue")
            if lr_map is not None:
                key = (float(resolution), float(params[0]), float(params[1]), float(params[2]))
                if key in lr_map:
                    l_val, r_val = lr_map[key]
                    raw_group_df = group_df
                    if raw_groups is not None:
                        raw_group_df = raw_groups.get(float(resolution), {}).get(params, group_df)
                    text_len_vals = raw_group_df["text_len"].astype(float)
                    l_mask = (text_len_vals >= (l_val - 50)) & (text_len_vals <= (l_val + 50))
                    r_mask = (text_len_vals >= (r_val - 50)) & (text_len_vals <= (r_val + 50))
                    l_pd = raw_group_df.loc[l_mask, "patch_density"].astype(float).mean()
                    r_pd = raw_group_df.loc[r_mask, "patch_density"].astype(float).mean()
                    if np.isfinite(l_pd):
                        ax.axvline(l_pd, color="darkorange", linestyle="--", linewidth=1.2)
                    if np.isfinite(r_pd):
                        ax.axvline(r_pd, color="darkorange", linestyle="--", linewidth=1.2)
            ax.set_title(f"fs={params[0]}, ls={params[1]}, cs={params[2]}")
            ax.set_xlabel("patch_density")
            ax.set_ylabel("edit_distance")
            ax.set_ylim(0.0, 1.0)
            if x_max is not None and np.isfinite(x_max):
                ax.set_xlim(0.0, x_max)
        for ax in axes_flat[count:]:
            ax.axis("off")
        fig.suptitle(f"resolution={resolution}")
        fig.tight_layout()
        fig_path = out_dir / f"{prefix}_resolution_{format_resolution_tag(resolution)}.png"
        fig.savefig(fig_path, dpi=300)
        plt.close(fig)


def plot_info_by_resolution(
    groups: dict[float, dict[tuple[float, float, float], pd.DataFrame]],
    out_dir: Path,
    ncols: int,
    prefix: str,
    lr_map: Optional[dict[tuple[float, float, float, float], tuple[float, float]]] = None,
    x_max_by_resolution: Optional[dict[float, float]] = None,
    raw_groups: Optional[dict[float, dict[tuple[float, float, float], pd.DataFrame]]] = None,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for resolution, res_groups in sorted(groups.items()):
        items = sorted(res_groups.items(), key=lambda item: item[0])
        count = len(items)
        ncols = max(1, ncols)
        nrows = max(1, math.ceil(count / ncols))
        fig, axes = plt.subplots(nrows, ncols, figsize=(4.0 * ncols, 3.2 * nrows), squeeze=False)
        axes_flat = axes.ravel()
        x_max = None
        if x_max_by_resolution is not None:
            x_max = x_max_by_resolution.get(float(resolution))
        for idx, (params, group_df) in enumerate(items):
            ax = axes_flat[idx]
            info_density = group_df["information_density"].astype(float).to_numpy()
            edit_distances = group_df["edit_distance"].astype(float).to_numpy()
            if info_density.size > 0:
                sorted_idx = np.argsort(info_density)
                info_density = info_density[sorted_idx]
                edit_distances = edit_distances[sorted_idx]
            ax.scatter(info_density, edit_distances, s=12, alpha=0.6, color="steelblue")
            if lr_map is not None:
                key = (float(resolution), float(params[0]), float(params[1]), float(params[2]))
                if key in lr_map:
                    l_val, r_val = lr_map[key]
                    raw_group_df = group_df
                    if raw_groups is not None:
                        raw_group_df = raw_groups.get(float(resolution), {}).get(params, group_df)
                    text_len_vals = raw_group_df["text_len"].astype(float)
                    l_mask = (text_len_vals >= (l_val - 50)) & (text_len_vals <= (l_val + 50))
                    r_mask = (text_len_vals >= (r_val - 50)) & (text_len_vals <= (r_val + 50))
                    l_info = raw_group_df.loc[l_mask, "information_density"].astype(float).mean()
                    r_info = raw_group_df.loc[r_mask, "information_density"].astype(float).mean()
                    if np.isfinite(l_info):
                        ax.axvline(l_info, color="darkorange", linestyle="--", linewidth=1.2)
                    if np.isfinite(r_info):
                        ax.axvline(r_info, color="darkorange", linestyle="--", linewidth=1.2)
            ax.set_title(f"fs={params[0]}, ls={params[1]}, cs={params[2]}")
            ax.set_xlabel("information_density")
            ax.set_ylabel("edit_distance")
            ax.set_ylim(0.0, 1.0)
            if x_max is not None and np.isfinite(x_max):
                ax.set_xlim(0.0, x_max)
        for ax in axes_flat[count:]:
            ax.axis("off")
        fig.suptitle(f"resolution={resolution}")
        fig.tight_layout()
        fig_path = out_dir / f"{prefix}_resolution_{format_resolution_tag(resolution)}.png"
        fig.savefig(fig_path, dpi=300)
        plt.close(fig)


def plot_model_w_by_resolution(
    groups: dict[float, dict[tuple[float, float, float], pd.DataFrame]],
    out_dir: Path,
    ncols: int,
    prefix: str,
    lr_map: Optional[dict[tuple[float, float, float, float], tuple[float, float]]] = None,
    x_max_by_resolution: Optional[dict[float, float]] = None,
    raw_groups: Optional[dict[float, dict[tuple[float, float, float], pd.DataFrame]]] = None,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for resolution, res_groups in sorted(groups.items()):
        items = sorted(res_groups.items(), key=lambda item: item[0])
        count = len(items)
        ncols = max(1, ncols)
        nrows = max(1, math.ceil(count / ncols))
        fig, axes = plt.subplots(nrows, ncols, figsize=(4.0 * ncols, 3.2 * nrows), squeeze=False)
        axes_flat = axes.ravel()
        x_max = None
        if x_max_by_resolution is not None:
            x_max = x_max_by_resolution.get(float(resolution))
        for idx, (params, group_df) in enumerate(items):
            ax = axes_flat[idx]
            model_w = group_df["model_W"].astype(float).to_numpy()
            edit_distances = group_df["edit_distance"].astype(float).to_numpy()
            if model_w.size > 0:
                sorted_idx = np.argsort(model_w)
                model_w = model_w[sorted_idx]
                edit_distances = edit_distances[sorted_idx]
            ax.scatter(model_w, edit_distances, s=12, alpha=0.6, color="steelblue")
            if lr_map is not None:
                key = (float(resolution), float(params[0]), float(params[1]), float(params[2]))
                if key in lr_map:
                    l_val, r_val = lr_map[key]
                    raw_group_df = group_df
                    if raw_groups is not None:
                        raw_group_df = raw_groups.get(float(resolution), {}).get(params, group_df)
                    text_len_vals = raw_group_df["text_len"].astype(float)
                    l_mask = (text_len_vals >= (l_val - 50)) & (text_len_vals <= (l_val + 50))
                    r_mask = (text_len_vals >= (r_val - 50)) & (text_len_vals <= (r_val + 50))
                    l_model_w = raw_group_df.loc[l_mask, "model_W"].astype(float).mean()
                    r_model_w = raw_group_df.loc[r_mask, "model_W"].astype(float).mean()
                    if np.isfinite(l_model_w):
                        ax.axvline(l_model_w, color="darkorange", linestyle="--", linewidth=1.2)
                    if np.isfinite(r_model_w):
                        ax.axvline(r_model_w, color="darkorange", linestyle="--", linewidth=1.2)
            ax.set_title(f"fs={params[0]}, ls={params[1]}, cs={params[2]}")
            ax.set_xlabel("model_W")
            ax.set_ylabel("edit_distance")
            ax.set_ylim(0.0, 1.0)
            if x_max is not None and np.isfinite(x_max):
                ax.set_xlim(0.0, x_max)
        for ax in axes_flat[count:]:
            ax.axis("off")
        fig.suptitle(f"resolution={resolution}")
        fig.tight_layout()
        fig_path = out_dir / f"{prefix}_resolution_{format_resolution_tag(resolution)}.png"
        fig.savefig(fig_path, dpi=300)
        plt.close(fig)


def plot_quantile_bins_by_resolution(
    df: pd.DataFrame,
    out_dir: Path,
    bin_width: int = BIN_WIDTH,
    q: float = Q_FOR_THRESHOLD,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for resolution in sorted(df["resolution"].unique()):
        df_r = df[df["resolution"] == resolution].copy()

        fs_list = sorted(df_r["font_size"].unique())
        ls_list = sorted(df_r["line_spacing"].unique())
        cs_list = sorted(df_r["letter_spacing"].unique())

        cs_list = sorted(cs_list)
        if len(cs_list) < 3:
            cs_list = cs_list + [cs_list[-1]] * (3 - len(cs_list))
        if len(cs_list) > 3:
            cs_list = cs_list[:3]

        pairs = []
        i = 0
        while i < len(ls_list):
            ls_a = ls_list[i]
            ls_b = ls_list[i + 1] if (i + 1) < len(ls_list) else None
            pairs.append((ls_a, ls_b))
            i += 2

        ncols = 6
        nrows = max(1, len(fs_list) * len(pairs))
        fig_w = 22
        fig_h = max(6, 3.0 * nrows)
        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(fig_w, fig_h), squeeze=False)
        fig.suptitle(f"resolution={resolution} | q={q:g} | bin_width={bin_width}", fontsize=14)

        x_min = 0.0
        x_max = float(df_r["text_len"].max()) if not df_r.empty else 1.0

        row_idx = 0
        for fs in fs_list:
            for (ls_a, ls_b) in pairs:
                for j, ls_val in enumerate([ls_a, ls_b]):
                    for k, cs in enumerate(cs_list):
                        col_idx = j * 3 + k
                        ax = axes[row_idx, col_idx]

                        if ls_val is None:
                            ax.axis("off")
                            continue

                        gdf = df_r[
                            (df_r["font_size"] == fs)
                            & (df_r["line_spacing"] == ls_val)
                            & (df_r["letter_spacing"] == cs)
                        ]

                        if gdf.empty:
                            ax.set_title(f"fs={fs:g}, ls={ls_val:g}, cs={cs:g}")
                            ax.set_xlim(x_min, x_max)
                            ax.set_ylim(0.0, 1.0)
                            ax.grid(True, alpha=0.2)
                            continue

                        left_edges, q_high, _, q_diff, _ = compute_bin_quantiles(
                            text_len=gdf["text_len"].to_numpy(),
                            edit_distance=gdf["edit_distance"].to_numpy(),
                            bin_width=bin_width,
                            q=q,
                        )

                        if left_edges.size > 0:
                            ax.plot(left_edges, q_high, marker="o", markersize=3, linewidth=1)
                            ax.plot(left_edges, q_diff, color="red", linewidth=1)

                        ax.set_title(f"fs={fs:g}, ls={ls_val:g}, cs={cs:g}")
                        ax.set_xlabel("text_len")
                        ax.set_ylabel(f"q={q:g}")
                        ax.set_ylim(0.0, 1.0)
                        ax.set_xlim(x_min, x_max)
                        ax.grid(True, alpha=0.2)

                row_idx += 1

        plt.tight_layout(rect=[0, 0, 1, 0.97])
        fig_path = out_dir / f"data4analysis_raw_bins_resolution_{format_resolution_tag(resolution)}.png"
        fig.savefig(fig_path, dpi=180)
        plt.close(fig)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv_path", type=Path, default=Path("data_analysis/data4analysis.csv"), help="Path to input CSV file")
    ap.add_argument("--out_csv", type=Path, default=Path("data_analysis/data4analysis_preprocess.csv"), help="Path to save filtered CSV file")
    ap.add_argument("--output_dir", type=Path, default=OUTPUT_DIR, help="Base output directory for preprocess artifacts")
    ap.add_argument("--plot_raw_dir", type=Path, default=None, help="Output directory for raw text_len plots")
    ap.add_argument("--plot_filtered_dir", type=Path, default=None, help="Output directory for filtered text_len plots")
    ap.add_argument("--plot_raw_pd_dir", type=Path, default=None, help="Output directory for raw patch_density plots")
    ap.add_argument("--plot_filtered_pd_dir", type=Path, default=None, help="Output directory for filtered patch_density plots")
    ap.add_argument("--plot_raw_info_dir", type=Path, default=None, help="Output directory for raw information_density plots")
    ap.add_argument("--plot_filtered_info_dir", type=Path, default=None, help="Output directory for filtered information_density plots")
    ap.add_argument("--plot_raw_model_w_dir", type=Path, default=None, help="Output directory for raw model_W plots")
    ap.add_argument("--plot_filtered_model_w_dir", type=Path, default=None, help="Output directory for filtered model_W plots")
    ap.add_argument("--plot_cols", type=int, default=6, help="Number of subplot columns per resolution figure")
    ap.add_argument("--left_iqr_thr", type=float, default=0.1, help="Left-side q_high threshold for L detection")
    ap.add_argument("--right_iqr_thr", type=float, default=0.3, help="Right-side y_iqr threshold for R detection")
    ap.add_argument("--peak_iqr_thr", type=float, default=0.4, help="Peak y_iqr threshold required to search for L/R")
    args = ap.parse_args()

    output_dir = args.output_dir
    raw_bin_plot_dir = output_dir / "data4analysis_raw_bins"
    plot_raw_dir = args.plot_raw_dir or (output_dir / "data4analysis_raw_len_plots")
    plot_filtered_dir = args.plot_filtered_dir or (output_dir / "data4analysis_preprocess_len_plots")
    plot_raw_pd_dir = args.plot_raw_pd_dir or (output_dir / "data4analysis_raw_pd_plots")
    plot_filtered_pd_dir = args.plot_filtered_pd_dir or (output_dir / "data4analysis_preprocess_pd_plots")
    plot_raw_info_dir = args.plot_raw_info_dir or (output_dir / "data4analysis_raw_info_plots")
    plot_filtered_info_dir = args.plot_filtered_info_dir or (output_dir / "data4analysis_preprocess_info_plots")
    plot_raw_model_w_dir = args.plot_raw_model_w_dir or (output_dir / "data4analysis_raw_modelW_plots")
    plot_filtered_model_w_dir = args.plot_filtered_model_w_dir or (output_dir / "data4analysis_preprocess_modelW_plots")
    high_variance_csv = output_dir / "data_preprocess_high_variance.csv"

    df = pd.read_csv(args.csv_path)
    required_cols = ["edit_distance", "text_len", "resolution", "font_size", "line_spacing", "letter_spacing"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    has_patch_density = "patch_density" in df.columns
    has_model_w = "model_W" in df.columns
    if not has_patch_density:
        logging.warning("Missing patch_density; skipping patch density/information density plots.")
    if not has_model_w:
        logging.warning("Missing model_W; skipping model_W plots.")

    if has_patch_density:
        df = add_information_density(df)
    plot_quantile_bins_by_resolution(df, raw_bin_plot_dir, bin_width=BIN_WIDTH, q=Q_FOR_THRESHOLD)

    grouped = list(df.groupby(["resolution", "font_size", "line_spacing", "letter_spacing"], observed=True))
    lr_map: dict[tuple[float, float, float, float], tuple[float, float]] = {}

    for g_key, dfg in grouped:
        resolution, font_size, line_spacing, letter_spacing = [float(val) for val in g_key]
        interval = detect_high_variance_interval(
            dfg,
            token_col="text_len",
            y_col="edit_distance",
            bin_width=BIN_WIDTH,
            left_iqr_thr=args.left_iqr_thr,
            right_iqr_thr=args.right_iqr_thr,
            peak_iqr_thr=args.peak_iqr_thr,
        )
        if np.isfinite(interval.L) and np.isfinite(interval.R):
            lr_map[(resolution, font_size, line_spacing, letter_spacing)] = (interval.L, interval.R)

    lr_rows = []
    for g_key, _ in grouped:
        resolution, font_size, line_spacing, letter_spacing = [float(val) for val in g_key]
        lr_key = (resolution, font_size, line_spacing, letter_spacing)
        L, R = lr_map.get(lr_key, (float("nan"), float("nan")))
        lr_rows.append(
            {
                "font_size": font_size,
                "line_spacing": line_spacing,
                "letter_spacing": letter_spacing,
                "resolution": resolution,
                "L": L,
                "R": R,
            }
        )
    high_variance_csv.parent.mkdir(parents=True, exist_ok=True)
    lr_df = pd.DataFrame(lr_rows, columns=["font_size", "line_spacing", "letter_spacing", "resolution", "L", "R"])
    lr_df.to_csv(high_variance_csv, index=False)
    logging.info("Saved L/R summary to %s", high_variance_csv)

    raw_groups = group_by_resolution(df)
    plot_by_resolution(raw_groups, plot_raw_dir, args.plot_cols, "data4analysis_raw", lr_map=lr_map)
    logging.info("Saved raw plots to %s", plot_raw_dir)
    if has_patch_density:
        max_pd_by_resolution = (
            df.groupby("resolution")["patch_density"].max().dropna().astype(float).to_dict() if not df.empty else {}
        )
        plot_pd_by_resolution(
            raw_groups,
            plot_raw_pd_dir,
            args.plot_cols,
            "data4analysis_raw_pd",
            lr_map=lr_map,
            x_max_by_resolution=max_pd_by_resolution,
            raw_groups=raw_groups,
        )
        logging.info("Saved raw patch density plots to %s", plot_raw_pd_dir)
        max_info_by_resolution = (
            df.groupby("resolution")["information_density"].max().dropna().astype(float).to_dict()
            if not df.empty
            else {}
        )
        plot_info_by_resolution(
            raw_groups,
            plot_raw_info_dir,
            args.plot_cols,
            "data4analysis_raw_info",
            lr_map=lr_map,
            x_max_by_resolution=max_info_by_resolution,
            raw_groups=raw_groups,
        )
        logging.info("Saved raw information density plots to %s", plot_raw_info_dir)
    if has_model_w:
        max_model_w_by_resolution = (
            df.groupby("resolution")["model_W"].max().dropna().astype(float).to_dict() if not df.empty else {}
        )
        plot_model_w_by_resolution(
            raw_groups,
            plot_raw_model_w_dir,
            args.plot_cols,
            "data4analysis_raw_modelW",
            lr_map=lr_map,
            x_max_by_resolution=max_model_w_by_resolution,
            raw_groups=raw_groups,
        )
        logging.info("Saved raw model_W plots to %s", plot_raw_model_w_dir)

    kept_parts = []
    for g_key, dfg in grouped:
        resolution, font_size, line_spacing, letter_spacing = [float(val) for val in g_key]
        lr_key = (resolution, font_size, line_spacing, letter_spacing)
        L, R = lr_map.get(lr_key, (float("nan"), float("nan")))
        scale = FILTER_IQR_SCALE
        if resolution == 1280.0:
            scale = 0.1
        elif resolution == 1024.0:
            scale = 0.2
        filtered_group = filter_by_lr_bins(dfg, L, R, bin_width=BIN_WIDTH, scale=scale)
        drop_count = 0
        if resolution == 512.0:
            drop_count = 5
        elif resolution == 640.0:
            drop_count = 10
        elif resolution == 1024.0:
            drop_count = 15
        elif resolution == 1280.0:
            drop_count = 20
        filtered_group = drop_top_ed_points(filtered_group, drop_count)
        if resolution in (512.0, 640.0):
            filtered_group = filtered_group[filtered_group["edit_distance"] <= 0.4].copy()
        elif resolution in (1024.0, 1280.0):
            filtered_group = filtered_group[filtered_group["edit_distance"] <= 0.2].copy()
        kept_parts.append(filtered_group)

    kept = pd.concat(kept_parts, axis=0).reset_index(drop=True) if kept_parts else pd.DataFrame(columns=df.columns)
    kept.to_csv(args.out_csv, index=False)
    logging.info("Saved filtered data to %s", args.out_csv)

    filtered_groups = group_by_resolution(kept)
    max_text_len_by_resolution = (
        kept.groupby("resolution")["text_len"].max().dropna().astype(float).to_dict() if not kept.empty else {}
    )
    plot_by_resolution(
        filtered_groups,
        plot_filtered_dir,
        args.plot_cols,
        "data4analysis_preprocess",
        lr_map=lr_map,
        x_max_by_resolution=max_text_len_by_resolution,
    )
    logging.info("Saved filtered plots to %s", plot_filtered_dir)

    if has_patch_density:
        max_pd_by_resolution = (
            kept.groupby("resolution")["patch_density"].max().dropna().astype(float).to_dict() if not kept.empty else {}
        )
        plot_pd_by_resolution(
            filtered_groups,
            plot_filtered_pd_dir,
            args.plot_cols,
            "data4analysis_preprocess_pd",
            lr_map=lr_map,
            x_max_by_resolution=max_pd_by_resolution,
            raw_groups=raw_groups,
        )
        logging.info("Saved patch density plots to %s", plot_filtered_pd_dir)
        max_info_by_resolution = (
            kept.groupby("resolution")["information_density"].max().dropna().astype(float).to_dict()
            if not kept.empty
            else {}
        )
        plot_info_by_resolution(
            filtered_groups,
            plot_filtered_info_dir,
            args.plot_cols,
            "data4analysis_preprocess_info",
            lr_map=lr_map,
            x_max_by_resolution=max_info_by_resolution,
            raw_groups=raw_groups,
        )
        logging.info("Saved information density plots to %s", plot_filtered_info_dir)
    if has_model_w:
        max_model_w_by_resolution = (
            kept.groupby("resolution")["model_W"].max().dropna().astype(float).to_dict() if not kept.empty else {}
        )
        plot_model_w_by_resolution(
            filtered_groups,
            plot_filtered_model_w_dir,
            args.plot_cols,
            "data4analysis_preprocess_modelW",
            lr_map=lr_map,
            x_max_by_resolution=max_model_w_by_resolution,
            raw_groups=raw_groups,
        )
        logging.info("Saved model_W plots to %s", plot_filtered_model_w_dir)


if __name__ == "__main__":
    main()
