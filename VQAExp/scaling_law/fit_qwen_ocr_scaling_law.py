import argparse
import json
import logging
import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data_analysis.fit_em_scaling_law_shared_parameters_exceptw import (
    beta_logpdf,
    compute_pi_r_ll_grouped,
    compute_pi_r_ll_simple,
    em_fit_grouped,
    ensure_dir,
    parse_resolutions,
    plot_ed_vs_r,
    plot_hist,
    plot_log_likelihood,
    plot_logv_logg,
    run_constant_baselines,
    summarize_responsibilities,
)


DEFAULT_CSV_PATH = Path("VQAExp/scaling_law/qwen_ocr_scaling_points.csv")
DEFAULT_OUTPUT_DIR = Path("VQAExp/scaling_law/Fitting/ocr")
DEFAULT_RESOLUTIONS = [512, 640, 1024, 1280]
DEFAULT_RESOLUTIONS_ARG = ",".join(str(resolution) for resolution in DEFAULT_RESOLUTIONS)
DEFAULT_HOLDOUT_LAYOUT = "20_0_7"
DEFAULT_MAX_ITER = 200
DEFAULT_TOL = 1e-6
DEFAULT_EPS = 1e-6
DEFAULT_INIT_THRESHOLD = 0.37
DEFAULT_INIT_SOFT = 0.05
DEFAULT_REG_STRENGTH = 0.0
DEFAULT_PLOT_DPI = 200
FONT_SIZES = [20, 28, 36]
LINE_SPACINGS = [0, 6, 24, 42]
LETTER_SPACINGS = [-1, 0, 7]


def iter_layouts() -> list[str]:
    return [f"{font_size}_{line_spacing}_{letter_spacing}" for font_size in FONT_SIZES for line_spacing in LINE_SPACINGS for letter_spacing in LETTER_SPACINGS]


def load_and_prepare_qwen_data(csv_path: Path, resolutions: list[int], eps: float, logger: logging.Logger) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    required_cols = {
        "layout",
        "resolution",
        "text_len",
        "edit_distance",
        "font_size",
        "line_spacing",
        "letter_spacing",
        "qwen_logV",
        "qwen_logG",
        "qwen_V",
        "qwen_G",
        "qwen_vision_token_num",
    }
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    df = df[df["resolution"].isin(resolutions)].copy()
    if df.empty:
        raise ValueError("No rows found for requested resolutions.")
    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=["text_len", "edit_distance", "qwen_logV", "qwen_logG", "qwen_V", "qwen_G"])
    edit_distance = df["edit_distance"].astype(float).to_numpy()
    edit_distance = np.clip(edit_distance, float(eps), 1.0 - float(eps))
    log_v = df["qwen_logV"].astype(float).to_numpy()
    log_g = df["qwen_logG"].astype(float).to_numpy()
    mask = np.isfinite(log_v) & np.isfinite(log_g) & np.isfinite(edit_distance)
    df = df.loc[mask].copy()
    df["edit_distance"] = edit_distance[mask]
    df["logV"] = log_v[mask]
    df["logG"] = log_g[mask]
    df["V"] = df["qwen_V"].astype(float)
    df["G"] = df["qwen_G"].astype(float)
    df["vision_token_num"] = df["qwen_vision_token_num"].astype(float)
    df["resolution"] = df["resolution"].astype(int)
    logger.info("Prepared %d Qwen rows for resolutions=%s", len(df), resolutions)
    return df


def expected_edit_distance(pi: np.ndarray, mean0: float, mean1: float) -> np.ndarray:
    return (1.0 - pi.astype(float)) * float(mean0) + pi.astype(float) * float(mean1)


def plot_grouped_ed_vs_text_len(df: pd.DataFrame, path: Path, dpi: int) -> None:
    layouts = iter_layouts()
    fig, axes = plt.subplots(6, 6, figsize=(24, 18), sharex=True, sharey=True)
    axes_flat = axes.ravel()
    for ax, layout in zip(axes_flat, layouts):
        group = df[df["layout"] == layout].copy()
        if group.empty:
            ax.set_title(f"{layout}\nmissing", fontsize=9)
            ax.set_xlim(0, 25000)
            ax.set_ylim(-0.05, 1.05)
            ax.grid(True, linestyle="--", linewidth=0.4, alpha=0.35)
            continue
        train_group = group[group["split"] == "train"]
        test_group = group[group["split"] == "test"]
        if not train_group.empty:
            ax.scatter(train_group["text_len"], train_group["edit_distance"], s=7, alpha=0.45, edgecolors="none", color="tab:blue")
        if not test_group.empty:
            ax.scatter(test_group["text_len"], test_group["edit_distance"], s=8, alpha=0.55, edgecolors="none", color="tab:orange")
        group_sorted = group.sort_values("text_len")
        ax.plot(group_sorted["text_len"], group_sorted["expected_ed"], color="red", linewidth=1.3)
        mean_ed = float(group["edit_distance"].mean())
        split_label = " test" if (group["split"] == "test").any() else ""
        ax.set_title(f"{layout}{split_label}  mean={mean_ed:.3f}", fontsize=9)
        ax.set_xlim(0, 25000)
        ax.set_ylim(-0.05, 1.05)
        ax.grid(True, linestyle="--", linewidth=0.4, alpha=0.35)
    for ax in axes[-1, :]:
        ax.set_xlabel("Characters")
    for ax in axes[:, 0]:
        ax.set_ylabel("Edit distance")
    resolution = int(df["resolution"].iloc[0]) if not df.empty else -1
    fig.suptitle(f"Qwen OCR scaling-law fit, resolution={resolution}", fontsize=16)
    fig.tight_layout(rect=(0, 0, 1, 0.975))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=int(dpi))
    plt.close(fig)


def compute_single_beta_ll(y: np.ndarray, single_beta: tuple[float, float]) -> float:
    return float(np.sum(beta_logpdf(np.log(y), np.log1p(-y), single_beta[0], single_beta[1])))


def run_fit(args: argparse.Namespace, logger: logging.Logger) -> None:
    resolutions = parse_resolutions(args.resolutions)
    df = load_and_prepare_qwen_data(args.csv_path, resolutions, args.eps, logger)
    resolutions_sorted = sorted(df["resolution"].unique().tolist())
    resolution_to_index = {resolution: idx for idx, resolution in enumerate(resolutions_sorted)}
    group_idx = df["resolution"].map(resolution_to_index).to_numpy()
    num_groups = len(resolutions_sorted)
    log_v = df["logV"].to_numpy()
    log_g = df["logG"].to_numpy()
    y = df["edit_distance"].to_numpy()

    test_mask = (df["layout"] == args.holdout_layout).to_numpy()
    train_mask_all = ~test_mask
    train_idx = np.flatnonzero(train_mask_all)
    test_idx = np.flatnonzero(test_mask)
    if train_idx.size == 0:
        raise ValueError("No training rows remain after applying holdout_layout.")
    if test_idx.size == 0:
        logger.warning("No test rows matched holdout_layout=%s", args.holdout_layout)

    em_result = em_fit_grouped(log_v[train_idx], log_g[train_idx], y[train_idx], group_idx[train_idx], num_groups, args.max_iter, args.tol, args.init_threshold, args.init_soft, args.reg_strength, logger)
    pi_full, r_full, _ = compute_pi_r_ll_grouped(log_v, log_g, y, group_idx, em_result.w0, em_result.a, em_result.alpha, em_result.alpha0, em_result.beta0, em_result.alpha1, em_result.beta1)
    train_ll_overall = compute_pi_r_ll_grouped(log_v[train_idx], log_g[train_idx], y[train_idx], group_idx[train_idx], em_result.w0, em_result.a, em_result.alpha, em_result.alpha0, em_result.beta0, em_result.alpha1, em_result.beta1)[2]
    test_ll_overall = None
    if test_idx.size > 0:
        test_ll_overall = compute_pi_r_ll_grouped(log_v[test_idx], log_g[test_idx], y[test_idx], group_idx[test_idx], em_result.w0, em_result.a, em_result.alpha, em_result.alpha0, em_result.beta0, em_result.alpha1, em_result.beta1)[2]

    mean1 = em_result.alpha1 / (em_result.alpha1 + em_result.beta1)
    mean0 = em_result.alpha0 / (em_result.alpha0 + em_result.beta0)
    var1 = (em_result.alpha1 * em_result.beta1) / (((em_result.alpha1 + em_result.beta1) ** 2) * (em_result.alpha1 + em_result.beta1 + 1.0))
    var0 = (em_result.alpha0 * em_result.beta0) / (((em_result.alpha0 + em_result.beta0) ** 2) * (em_result.alpha0 + em_result.beta0 + 1.0))

    root_plots_dir = args.output_dir / "plots"
    ensure_dir(args.output_dir)
    ensure_dir(root_plots_dir)
    all_points = df.copy()
    all_points["pi"] = pi_full
    all_points["r"] = r_full
    all_points["expected_ed"] = expected_edit_distance(pi_full, float(mean0), float(mean1))
    all_points["split"] = np.where(test_mask, "test", "train")
    all_points.to_csv(args.output_dir / "qwen_em_fit_points_all_resolutions.csv", index=False)

    for resolution in resolutions_sorted:
        group = resolution_to_index[resolution]
        mask = (df["resolution"] == resolution).to_numpy()
        output_dir = args.output_dir / str(resolution)
        plots_dir = output_dir / "plots"
        ensure_dir(output_dir)
        ensure_dir(plots_dir)

        results_df = all_points.loc[mask].copy()
        results_df.to_csv(output_dir / "em_fit_points.csv", index=False)
        plot_log_likelihood(em_result.log_likelihood_history, plots_dir / "em_loglik.png", args.plot_dpi)
        plot_hist(results_df["r"].to_numpy(), plots_dir / "responsibility_hist.png", args.plot_dpi)
        plot_ed_vs_r(results_df["edit_distance"].to_numpy(), results_df["r"].to_numpy(), plots_dir / "edit_distance_vs_r.png", args.plot_dpi)
        plot_logv_logg(results_df["logV"].to_numpy(), results_df["logG"].to_numpy(), results_df["pi"].to_numpy(), plots_dir / "logV_logG_pi.png", args.plot_dpi, "pi")
        plot_logv_logg(results_df["logV"].to_numpy(), results_df["logG"].to_numpy(), results_df["r"].to_numpy(), plots_dir / "logV_logG_r.png", args.plot_dpi, "r")
        plot_grouped_ed_vs_text_len(results_df, plots_dir / "grouped_text_len_ed.png", args.plot_dpi)
        plot_grouped_ed_vs_text_len(results_df, root_plots_dir / f"qwen_scaling_fit_res_{resolution}.png", args.plot_dpi)

        train_mask = mask & train_mask_all
        test_mask_res = mask & test_mask
        train_ll = compute_pi_r_ll_grouped(log_v[train_mask], log_g[train_mask], y[train_mask], group_idx[train_mask], em_result.w0, em_result.a, em_result.alpha, em_result.alpha0, em_result.beta0, em_result.alpha1, em_result.beta1)[2]
        baseline_train_metrics = None
        baseline_test_metrics = None
        const_beta = np.array([0.0])
        const_alpha0 = 1.0
        const_beta0 = 1.0
        const_alpha1 = 1.0
        const_beta1 = 1.0
        single_beta = (1.0, 1.0)
        if np.any(train_mask):
            const_beta, const_alpha0, const_beta0, const_alpha1, const_beta1, single_beta = run_constant_baselines(y[train_mask], args.max_iter, args.tol, args.init_threshold, args.init_soft, args.reg_strength, logger)
            x_const_train = np.ones((int(np.sum(train_mask)), 1), dtype=float)
            baseline_train_metrics = {
                "constant_pi_mixture": compute_pi_r_ll_simple(x_const_train, y[train_mask], const_beta, const_alpha0, const_beta0, const_alpha1, const_beta1),
                "single_beta": compute_single_beta_ll(y[train_mask], single_beta),
            }
            if np.any(test_mask_res):
                x_const_test = np.ones((int(np.sum(test_mask_res)), 1), dtype=float)
                baseline_test_metrics = {
                    "constant_pi_mixture": compute_pi_r_ll_simple(x_const_test, y[test_mask_res], const_beta, const_alpha0, const_beta0, const_alpha1, const_beta1),
                    "single_beta": compute_single_beta_ll(y[test_mask_res], single_beta),
                }
        test_metrics = None
        if np.any(test_mask_res):
            test_ll = compute_pi_r_ll_grouped(log_v[test_mask_res], log_g[test_mask_res], y[test_mask_res], group_idx[test_mask_res], em_result.w0, em_result.a, em_result.alpha, em_result.alpha0, em_result.beta0, em_result.alpha1, em_result.beta1)[2]
            test_metrics = {
                "gated_mixture": float(test_ll),
                "constant_pi_mixture": None if baseline_test_metrics is None else float(baseline_test_metrics["constant_pi_mixture"]),
                "single_beta": None if baseline_test_metrics is None else float(baseline_test_metrics["single_beta"]),
            }
        train_metrics = {
            "gated_mixture": float(train_ll),
            "constant_pi_mixture": None if baseline_train_metrics is None else float(baseline_train_metrics["constant_pi_mixture"]),
            "single_beta": None if baseline_train_metrics is None else float(baseline_train_metrics["single_beta"]),
        }
        summary = {
            "resolution": int(resolution),
            "holdout_layout": args.holdout_layout,
            "n_samples": int(mask.sum()),
            "n_train": int(train_mask.sum()),
            "n_test": int(test_mask_res.sum()),
            "mean_qwen_vision_token_num": float(np.mean(results_df["qwen_vision_token_num"])),
            "gating_params": {"w0": float(em_result.w0[group]), "a": float(em_result.a), "alpha": float(em_result.alpha)},
            "beta_params": {
                "low_ed": {"alpha": float(em_result.alpha0), "beta": float(em_result.beta0), "mean": float(mean0), "var": float(var0)},
                "high_ed": {"alpha": float(em_result.alpha1), "beta": float(em_result.beta1), "mean": float(mean1), "var": float(var1)},
            },
            "responsibility_stats": summarize_responsibilities(results_df["r"].to_numpy()),
            "log_likelihood_history": em_result.log_likelihood_history,
            "converged": bool(em_result.converged),
            "train_log_likelihood": train_metrics,
            "test_log_likelihood": test_metrics,
            "single_beta_params": {"alpha": float(single_beta[0]), "beta": float(single_beta[1])},
            "constant_pi_params": {
                "w0": float(const_beta[0]) if const_beta.size > 0 else 0.0,
                "alpha0": float(const_alpha0),
                "beta0": float(const_beta0),
                "alpha1": float(const_alpha1),
                "beta1": float(const_beta1),
            },
            "init_threshold": float(args.init_threshold),
            "overall_train_log_likelihood": float(train_ll_overall),
            "overall_test_log_likelihood": None if test_ll_overall is None else float(test_ll_overall),
        }
        with (output_dir / "em_fit_summary.json").open("w", encoding="utf-8") as file_handle:
            json.dump(summary, file_handle, ensure_ascii=False, indent=2)
        history_df = pd.DataFrame({"iteration": np.arange(len(em_result.log_likelihood_history)), "log_likelihood": em_result.log_likelihood_history})
        history_df.to_csv(output_dir / "em_loglik_history.csv", index=False)
        logger.info("Saved Qwen fitting results for resolution=%d to %s", resolution, output_dir)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fit Qwen OCR EM mixture scaling law with qwen_logV/qwen_logG and a held-out layout.")
    parser.add_argument("--csv_path", type=Path, default=DEFAULT_CSV_PATH, help="Path to Qwen OCR scaling points CSV.")
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Directory to store Qwen EM fit outputs.")
    parser.add_argument("--resolutions", type=str, default=DEFAULT_RESOLUTIONS_ARG, help="Comma-separated resolutions to fit.")
    parser.add_argument("--holdout_layout", type=str, default=DEFAULT_HOLDOUT_LAYOUT, help="Layout held out from training and used as test set.")
    parser.add_argument("--max_iter", type=int, default=DEFAULT_MAX_ITER, help="Maximum EM iterations.")
    parser.add_argument("--tol", type=float, default=DEFAULT_TOL, help="Convergence tolerance on log-likelihood.")
    parser.add_argument("--eps", type=float, default=DEFAULT_EPS, help="Clipping epsilon for edit_distance.")
    parser.add_argument("--init_threshold", type=float, default=DEFAULT_INIT_THRESHOLD, help="Threshold on edit_distance for initialization split.")
    parser.add_argument("--init_soft", type=float, default=DEFAULT_INIT_SOFT, help="Soft label strength for initialization.")
    parser.add_argument("--reg_strength", type=float, default=DEFAULT_REG_STRENGTH, help="L2 regularization strength.")
    parser.add_argument("--plot_dpi", type=int, default=DEFAULT_PLOT_DPI, help="DPI for saved plots.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")
    logger = logging.getLogger("fit_qwen_ocr_scaling_law")
    run_fit(args, logger)


if __name__ == "__main__":
    main()
