"""Fit EM gating model across resolutions with shared parameters."""

import argparse
import json
import logging
import math
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import digamma, expit, gammaln

DEFAULT_CSV_PATH = Path("data_analysis/data4analysis.csv")
DEFAULT_OUTPUT_DIR = Path("data_analysis/em_fiting_res/shared_parameters")
DEFAULT_RESOLUTIONS = [512, 640, 1024, 1280]
DEFAULT_RESOLUTIONS_ARG = ",".join(str(resolution) for resolution in DEFAULT_RESOLUTIONS)
DEFAULT_MAX_ITER = 200
DEFAULT_TOL = 1e-6
DEFAULT_EPS = 1e-6
DEFAULT_INIT_THRESHOLD = 0.37
DEFAULT_INIT_SOFT = 0.05
DEFAULT_REG_STRENGTH = 0.0
DEFAULT_TEST_SPLIT = 0.0
DEFAULT_RANDOM_SEED = 42
DEFAULT_PLOT_DPI = 200
MIN_ALPHA_BETA = 1e-3
MAX_ALPHA_BETA = 1e4


@dataclass
class EMSharedResult:
    w0: float
    a: float
    alpha: float
    alpha0: float
    beta0: float
    alpha1: float
    beta1: float
    responsibilities: np.ndarray
    pi: np.ndarray
    log_likelihood_history: list[float]
    converged: bool


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def log_sigmoid(z: np.ndarray) -> np.ndarray:
    return -np.logaddexp(0.0, -z)


def log1m_sigmoid(z: np.ndarray) -> np.ndarray:
    return -np.logaddexp(0.0, z)


def beta_logpdf(log_y: np.ndarray, log1m_y: np.ndarray, alpha: float, beta: float) -> np.ndarray:
    norm = gammaln(alpha + beta) - gammaln(alpha) - gammaln(beta)
    return (alpha - 1.0) * log_y + (beta - 1.0) * log1m_y + norm


def weighted_beta_moments(y: np.ndarray, weights: np.ndarray) -> tuple[float, float]:
    weights = np.asarray(weights, dtype=float)
    weights = np.where(np.isfinite(weights) & (weights > 0.0), weights, 0.0)
    wsum = float(np.sum(weights))
    if wsum <= 0.0:
        return 1.0, 1.0
    weights = weights / wsum
    mean = float(np.sum(weights * y))
    var = float(np.sum(weights * (y - mean) ** 2))
    if var <= 1e-12 or mean <= 0.0 or mean >= 1.0:
        return 1.0, 1.0
    scale = mean * (1.0 - mean) / var - 1.0
    if not np.isfinite(scale) or scale <= 0.0:
        return 1.0, 1.0
    alpha = mean * scale
    beta = (1.0 - mean) * scale
    if alpha <= 0.0 or beta <= 0.0:
        return 1.0, 1.0
    return float(alpha), float(beta)


def fit_weighted_beta(
    y: np.ndarray,
    weights: np.ndarray,
    init_params: tuple[float, float] | None = None,
    max_iter: int = 200,
) -> tuple[float, float]:
    weights = np.asarray(weights, dtype=float)
    weights = np.where(np.isfinite(weights) & (weights > 0.0), weights, 0.0)
    wsum = float(np.sum(weights))
    if wsum <= 0.0:
        return 1.0, 1.0
    if init_params is None:
        init_params = weighted_beta_moments(y, weights)
    log_y = np.log(y)
    log1m_y = np.log1p(-y)
    sum_w_log_y = float(np.sum(weights * log_y))
    sum_w_log1m_y = float(np.sum(weights * log1m_y))

    def nll(params: np.ndarray) -> float:
        alpha = float(np.exp(params[0]))
        beta = float(np.exp(params[1]))
        log_norm = gammaln(alpha + beta) - gammaln(alpha) - gammaln(beta)
        ll = (alpha - 1.0) * sum_w_log_y + (beta - 1.0) * sum_w_log1m_y + wsum * log_norm
        return -float(ll)

    def grad(params: np.ndarray) -> np.ndarray:
        alpha = float(np.exp(params[0]))
        beta = float(np.exp(params[1]))
        psi_a = digamma(alpha)
        psi_b = digamma(beta)
        psi_ab = digamma(alpha + beta)
        d_ll_alpha = sum_w_log_y + wsum * (psi_ab - psi_a)
        d_ll_beta = sum_w_log1m_y + wsum * (psi_ab - psi_b)
        g0 = -d_ll_alpha * alpha
        g1 = -d_ll_beta * beta
        return np.array([g0, g1], dtype=float)

    init_alpha, init_beta = init_params
    init_alpha = float(np.clip(init_alpha, MIN_ALPHA_BETA, MAX_ALPHA_BETA))
    init_beta = float(np.clip(init_beta, MIN_ALPHA_BETA, MAX_ALPHA_BETA))
    res = minimize(
        nll,
        np.log([init_alpha, init_beta]),
        method="L-BFGS-B",
        jac=grad,
        bounds=[(math.log(MIN_ALPHA_BETA), math.log(MAX_ALPHA_BETA))] * 2,
        options={"maxiter": int(max_iter)},
    )
    alpha = float(np.exp(res.x[0]))
    beta = float(np.exp(res.x[1]))
    if not res.success or not np.isfinite(alpha) or not np.isfinite(beta):
        return init_alpha, init_beta
    return alpha, beta


def fit_logistic_regression_simple(
    X: np.ndarray,
    y_soft: np.ndarray,
    beta_init: np.ndarray,
    reg_strength: float,
    max_iter: int = 200,
) -> np.ndarray:
    X = np.asarray(X, dtype=float)
    y_soft = np.asarray(y_soft, dtype=float)

    def nll(beta: np.ndarray) -> float:
        z = X @ beta
        log_p = log_sigmoid(z)
        log1m_p = log1m_sigmoid(z)
        ll = np.sum(y_soft * log_p + (1.0 - y_soft) * log1m_p)
        penalty = 0.5 * float(reg_strength) * float(np.sum(beta[1:] ** 2))
        return -float(ll) + penalty

    def grad(beta: np.ndarray) -> np.ndarray:
        z = X @ beta
        pi = expit(z)
        g = X.T @ (pi - y_soft)
        if reg_strength > 0.0:
            g[1:] += reg_strength * beta[1:]
        return g

    res = minimize(
        nll,
        beta_init,
        method="L-BFGS-B",
        jac=grad,
        options={"maxiter": int(max_iter)},
    )
    if not res.success or not np.all(np.isfinite(res.x)):
        return beta_init
    return res.x.astype(float)


def initialize_responsibilities(y: np.ndarray, threshold: float, soft: float) -> np.ndarray:
    r = np.where(y >= float(threshold), 1.0 - soft, soft)
    return r.astype(float)


def em_fit(
    X: np.ndarray,
    y: np.ndarray,
    max_iter: int,
    tol: float,
    init_threshold: float,
    init_soft: float,
    reg_strength: float,
    logger: logging.Logger,
) -> tuple[np.ndarray, float, float, float, float, np.ndarray, np.ndarray, list[float], bool]:
    y = np.asarray(y, dtype=float)
    X = np.asarray(X, dtype=float)
    log_y = np.log(y)
    log1m_y = np.log1p(-y)
    r = initialize_responsibilities(y, init_threshold, init_soft)
    alpha1, beta1 = fit_weighted_beta(y, r)
    alpha0, beta0 = fit_weighted_beta(y, 1.0 - r)
    beta = np.zeros(X.shape[1], dtype=float)
    history: list[float] = []
    converged = False
    for idx in range(int(max_iter)):
        z = X @ beta
        log_pi = log_sigmoid(z)
        log_p1 = beta_logpdf(log_y, log1m_y, alpha1, beta1)
        log_p0 = beta_logpdf(log_y, log1m_y, alpha0, beta0)
        log_mix = np.logaddexp(log_pi + log_p1, log1m_sigmoid(z) + log_p0)
        ll = float(np.sum(log_mix))
        history.append(ll)
        if idx > 0 and abs(history[-1] - history[-2]) < float(tol):
            converged = True
            break
        r = np.exp(log_pi + log_p1 - log_mix)
        beta = fit_logistic_regression_simple(X, r, beta, reg_strength, max_iter=max_iter)
        alpha1, beta1 = fit_weighted_beta(y, r, init_params=(alpha1, beta1))
        alpha0, beta0 = fit_weighted_beta(y, 1.0 - r, init_params=(alpha0, beta0))
        if idx % 10 == 0:
            logger.info("Iter %d log-likelihood %.6f", idx, ll)
    z = X @ beta
    log_pi = log_sigmoid(z)
    log_p1 = beta_logpdf(log_y, log1m_y, alpha1, beta1)
    log_p0 = beta_logpdf(log_y, log1m_y, alpha0, beta0)
    log_mix = np.logaddexp(log_pi + log_p1, log1m_sigmoid(z) + log_p0)
    r = np.exp(log_pi + log_p1 - log_mix)
    pi = expit(z)
    mean1 = alpha1 / (alpha1 + beta1)
    mean0 = alpha0 / (alpha0 + beta0)
    if mean1 < mean0:
        alpha0, alpha1 = alpha1, alpha0
        beta0, beta1 = beta1, beta0
        beta = -beta
        z = X @ beta
        log_pi = log_sigmoid(z)
        log_p1 = beta_logpdf(log_y, log1m_y, alpha1, beta1)
        log_p0 = beta_logpdf(log_y, log1m_y, alpha0, beta0)
        log_mix = np.logaddexp(log_pi + log_p1, log1m_sigmoid(z) + log_p0)
        r = np.exp(log_pi + log_p1 - log_mix)
        pi = expit(z)
    return beta, float(alpha0), float(beta0), float(alpha1), float(beta1), r.astype(float), pi.astype(float), history, converged


def compute_pi_r_ll_simple(
    X: np.ndarray,
    y: np.ndarray,
    beta: np.ndarray,
    alpha0: float,
    beta0: float,
    alpha1: float,
    beta1: float,
) -> float:
    y = np.asarray(y, dtype=float)
    log_y = np.log(y)
    log1m_y = np.log1p(-y)
    z = X @ beta
    log_pi = log_sigmoid(z)
    log_p1 = beta_logpdf(log_y, log1m_y, alpha1, beta1)
    log_p0 = beta_logpdf(log_y, log1m_y, alpha0, beta0)
    log_mix = np.logaddexp(log_pi + log_p1, log1m_sigmoid(z) + log_p0)
    return float(np.sum(log_mix))


def compute_pi_r_ll_simple_full(
    X: np.ndarray,
    y: np.ndarray,
    beta: np.ndarray,
    alpha0: float,
    beta0: float,
    alpha1: float,
    beta1: float,
) -> tuple[np.ndarray, np.ndarray, float]:
    y = np.asarray(y, dtype=float)
    log_y = np.log(y)
    log1m_y = np.log1p(-y)
    z = X @ beta
    log_pi = log_sigmoid(z)
    log_p1 = beta_logpdf(log_y, log1m_y, alpha1, beta1)
    log_p0 = beta_logpdf(log_y, log1m_y, alpha0, beta0)
    log_mix = np.logaddexp(log_pi + log_p1, log1m_sigmoid(z) + log_p0)
    r = np.exp(log_pi + log_p1 - log_mix)
    pi = expit(z)
    return pi.astype(float), r.astype(float), float(np.sum(log_mix))


def fit_single_beta(y: np.ndarray) -> tuple[float, float]:
    weights = np.ones_like(y, dtype=float)
    return fit_weighted_beta(y, weights)


def plot_log_likelihood(history: list[float], path: Path, dpi: int) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(np.arange(len(history)), history, color="tab:blue")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Log-Likelihood")
    ax.set_title("EM Log-Likelihood")
    fig.tight_layout()
    fig.savefig(path, dpi=int(dpi))
    plt.close(fig)


def plot_hist(values: np.ndarray, path: Path, dpi: int) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(values, bins=30, color="tab:green", alpha=0.8)
    ax.set_xlabel("Responsibility r")
    ax.set_ylabel("Count")
    ax.set_title("Responsibility Distribution")
    fig.tight_layout()
    fig.savefig(path, dpi=int(dpi))
    plt.close(fig)


def plot_ed_vs_r(y: np.ndarray, r: np.ndarray, path: Path, dpi: int) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(y, r, s=12, alpha=0.6, color="tab:orange")
    ax.set_xlabel("Edit Distance (clipped)")
    ax.set_ylabel("Responsibility r")
    ax.set_title("Edit Distance vs Responsibility")
    fig.tight_layout()
    fig.savefig(path, dpi=int(dpi))
    plt.close(fig)


def plot_logv_logg(
    log_v: np.ndarray,
    log_g: np.ndarray,
    values: np.ndarray,
    path: Path,
    dpi: int,
    label: str,
) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    scatter = ax.scatter(log_v, log_g, c=values, cmap="viridis", s=14, alpha=0.7)
    fig.colorbar(scatter, ax=ax, label=label)
    ax.set_xlabel("log V")
    ax.set_ylabel("log G")
    ax.set_title("Gating Surface")
    fig.tight_layout()
    fig.savefig(path, dpi=int(dpi))
    plt.close(fig)


def plot_grouped_ed_vs_text_len(
    df: pd.DataFrame,
    mean0: float,
    mean1: float,
    path: Path,
    dpi: int,
) -> None:
    required_cols = {"font_size", "line_spacing", "letter_spacing", "text_len", "edit_distance", "pi"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns for grouped plot: {sorted(missing)}")
    df = df.copy()
    df["expected_ed"] = (1.0 - df["pi"].astype(float)) * float(mean0) + df["pi"].astype(float) * float(mean1)
    groups = list(df.groupby(["font_size", "line_spacing", "letter_spacing"], sort=True))
    n_groups = len(groups)
    ncols = 6
    nrows = max(1, math.ceil(n_groups / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 4, nrows * 3), squeeze=False)
    for ax in axes.flat:
        ax.set_visible(False)
    for idx, ((font_size, line_spacing, letter_spacing), group) in enumerate(groups):
        ax = axes[idx // ncols][idx % ncols]
        ax.set_visible(True)
        ax.scatter(
            group["text_len"],
            group["edit_distance"],
            s=8,
            alpha=0.6,
            color="tab:blue",
        )
        group_sorted = group.sort_values("text_len")
        ax.plot(
            group_sorted["text_len"],
            group_sorted["expected_ed"],
            color="red",
            linewidth=1.5,
        )
        ax.set_title(
            f"font={font_size}, line={line_spacing}, letter={letter_spacing}",
            fontsize=8,
        )
        ax.set_xlabel("text_len")
        ax.set_ylabel("edit_distance")
    fig.tight_layout()
    fig.savefig(path, dpi=int(dpi))
    plt.close(fig)


def summarize_responsibilities(r: np.ndarray) -> dict[str, float]:
    r = np.asarray(r, dtype=float)
    quantiles = np.quantile(r, [0.1, 0.5, 0.9])
    return {
        "mean": float(np.mean(r)),
        "std": float(np.std(r)),
        "q10": float(quantiles[0]),
        "q50": float(quantiles[1]),
        "q90": float(quantiles[2]),
    }


def parse_resolutions(resolutions_arg: str) -> list[int]:
    values: list[int] = []
    for token in resolutions_arg.split(","):
        token = token.strip()
        if not token:
            continue
        try:
            values.append(int(token))
        except ValueError as exc:
            raise ValueError(f"Invalid resolution value: {token}") from exc
    if not values:
        raise ValueError("No valid resolutions provided.")
    return values


def load_and_prepare_data(csv_path: Path, resolutions: list[int], eps: float, logger: logging.Logger) -> pd.DataFrame:
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
    df = df[df["resolution"].isin(resolutions)].copy()
    if df.empty:
        raise ValueError("No rows found for requested resolutions.")
    df = df.replace([np.inf, -np.inf], np.nan).dropna(
        subset=[
            "text_len",
            "edit_distance",
            "character_width",
            "letter_spacing",
            "character_height",
            "line_spacing",
            "font_size",
        ]
    )
    text_len = df["text_len"].astype(float).to_numpy()
    char_width = df["character_width"].astype(float).to_numpy()
    letter_spacing = df["letter_spacing"].astype(float).to_numpy()
    char_height = df["character_height"].astype(float).to_numpy()
    line_spacing = df["line_spacing"].astype(float).to_numpy()
    resolution_vals = df["resolution"].astype(float).to_numpy()
    vision_token_num = (resolution_vals / 64.0) ** 2
    g_vals = text_len / vision_token_num
    numerator = (char_width + letter_spacing) * text_len
    denominator = char_height + line_spacing
    v_vals = np.divide(
        numerator,
        denominator,
        out=np.full_like(text_len, np.nan),
        where=denominator != 0,
    )
    v_vals = np.maximum(v_vals, float(eps))
    g_vals = np.maximum(g_vals, float(eps))
    edit_distance = df["edit_distance"].astype(float).to_numpy()
    edit_distance = np.clip(edit_distance, float(eps), 1.0 - float(eps))
    log_v = np.log(v_vals)
    log_g = np.log(g_vals)
    mask = np.isfinite(log_v) & np.isfinite(log_g) & np.isfinite(edit_distance)
    df = df.loc[mask].copy()
    df["V"] = v_vals[mask]
    df["edit_distance"] = edit_distance[mask]
    df["vision_token_num"] = vision_token_num[mask]
    df["G"] = g_vals[mask]
    df["logV"] = log_v[mask]
    df["logG"] = log_g[mask]
    df["resolution"] = df["resolution"].astype(int)
    logger.info("Prepared %d rows for resolutions=%s", len(df), resolutions)
    return df


def run_constant_baselines(
    y: np.ndarray,
    max_iter: int,
    tol: float,
    init_threshold: float,
    init_soft: float,
    reg_strength: float,
    logger: logging.Logger,
) -> tuple[np.ndarray, float, float, float, float, tuple[float, float]]:
    X_const = np.ones((y.size, 1), dtype=float)
    em_const = em_fit(X_const, y, max_iter, tol, init_threshold, init_soft, reg_strength, logger)
    beta, alpha0, beta0, alpha1, beta1, _, _, _, _ = em_const
    single_beta = fit_single_beta(y)
    return beta, alpha0, beta0, alpha1, beta1, single_beta


def main() -> None:
    parser = argparse.ArgumentParser(description="Fit EM mixture scaling law with shared parameters across resolutions.")
    parser.add_argument("--csv_path", type=Path, default=DEFAULT_CSV_PATH, help="Path to input CSV file.")
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Directory to store EM fit outputs.")
    parser.add_argument("--resolutions", type=str, default=DEFAULT_RESOLUTIONS_ARG, help="Comma-separated resolutions to fit.")
    parser.add_argument("--max_iter", type=int, default=DEFAULT_MAX_ITER, help="Maximum EM iterations.")
    parser.add_argument("--tol", type=float, default=DEFAULT_TOL, help="Convergence tolerance on log-likelihood.")
    parser.add_argument("--eps", type=float, default=DEFAULT_EPS, help="Clipping epsilon for edit_distance.")
    parser.add_argument("--init_threshold", type=float, default=DEFAULT_INIT_THRESHOLD, help="Threshold on edit_distance for initialization split.")
    parser.add_argument("--init_soft", type=float, default=DEFAULT_INIT_SOFT, help="Soft label strength for initialization.")
    parser.add_argument("--reg_strength", type=float, default=DEFAULT_REG_STRENGTH, help="L2 regularization strength.")
    parser.add_argument("--test_split", type=float, default=DEFAULT_TEST_SPLIT, help="Holdout fraction for log-likelihood evaluation.")
    parser.add_argument("--seed", type=int, default=DEFAULT_RANDOM_SEED, help="Random seed for train/test split.")
    parser.add_argument("--plot_dpi", type=int, default=DEFAULT_PLOT_DPI, help="DPI for saved plots.")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")
    logger = logging.getLogger("fit_em_scaling_law_shared_parameters")

    resolutions = parse_resolutions(args.resolutions)
    df = load_and_prepare_data(args.csv_path, resolutions, args.eps, logger)

    resolutions_sorted = sorted(df["resolution"].unique().tolist())

    log_v = df["logV"].to_numpy()
    log_g = df["logG"].to_numpy()
    y = df["edit_distance"].to_numpy()
    X = np.column_stack([np.ones(len(df), dtype=float), log_v, log_g])

    exclude_mask = (
        (df["font_size"] == 28)
        & (df["line_spacing"] == 6)
        & (df["letter_spacing"] == 0)
    ).to_numpy()
    indices = np.arange(len(df))
    eligible_indices = indices[~exclude_mask]
    train_idx = eligible_indices
    test_idx = np.array([], dtype=int)
    if args.test_split and 0.0 < float(args.test_split) < 1.0:
        rng = np.random.default_rng(int(args.seed))
        rng.shuffle(eligible_indices)
        split = int(len(eligible_indices) * (1.0 - float(args.test_split)))
        train_idx = eligible_indices[:split]
        test_idx = eligible_indices[split:]

    em_result = em_fit(
        X[train_idx],
        y[train_idx],
        args.max_iter,
        args.tol,
        args.init_threshold,
        args.init_soft,
        args.reg_strength,
        logger,
    )
    beta, alpha0, beta0, alpha1, beta1, _, _, history, converged = em_result

    pi_full, r_full, _ = compute_pi_r_ll_simple_full(X, y, beta, alpha0, beta0, alpha1, beta1)

    train_ll_overall = compute_pi_r_ll_simple(X[train_idx], y[train_idx], beta, alpha0, beta0, alpha1, beta1)
    test_ll_overall = None
    if test_idx.size > 0:
        test_ll_overall = compute_pi_r_ll_simple(X[test_idx], y[test_idx], beta, alpha0, beta0, alpha1, beta1)

    is_train = np.zeros(len(df), dtype=bool)
    is_train[train_idx] = True
    is_test = np.zeros(len(df), dtype=bool)
    is_test[test_idx] = True
    is_excluded = exclude_mask.copy()

    mean1 = alpha1 / (alpha1 + beta1)
    mean0 = alpha0 / (alpha0 + beta0)
    var1 = (alpha1 * beta1) / (((alpha1 + beta1) ** 2) * (alpha1 + beta1 + 1.0))
    var0 = (alpha0 * beta0) / (((alpha0 + beta0) ** 2) * (alpha0 + beta0 + 1.0))

    shared_result = EMSharedResult(
        w0=float(beta[0]),
        a=float(beta[1]) if beta.size > 1 else 0.0,
        alpha=float(beta[2]) if beta.size > 2 else 0.0,
        alpha0=float(alpha0),
        beta0=float(beta0),
        alpha1=float(alpha1),
        beta1=float(beta1),
        responsibilities=r_full,
        pi=pi_full,
        log_likelihood_history=history,
        converged=bool(converged),
    )

    for resolution in resolutions_sorted:
        mask = df["resolution"] == resolution
        if not np.any(mask):
            continue

        output_dir = args.output_dir / str(resolution)
        plots_dir = output_dir / "plots"
        ensure_dir(output_dir)
        ensure_dir(plots_dir)

        results_df = df.loc[mask].copy()
        results_df["pi"] = shared_result.pi[mask]
        results_df["r"] = shared_result.responsibilities[mask]
        if test_idx.size > 0:
            split_flag = np.array(["train"] * len(results_df), dtype=object)
            split_flag[is_test[mask]] = "test"
            if np.any(is_excluded[mask]):
                split_flag[is_excluded[mask]] = "excluded"
            results_df["split"] = split_flag

        results_df.to_csv(output_dir / "em_fit_points.csv", index=False)

        plot_log_likelihood(shared_result.log_likelihood_history, plots_dir / "em_loglik.png", args.plot_dpi)
        plot_hist(results_df["r"].to_numpy(), plots_dir / "responsibility_hist.png", args.plot_dpi)
        plot_ed_vs_r(results_df["edit_distance"].to_numpy(), results_df["r"].to_numpy(), plots_dir / "edit_distance_vs_r.png", args.plot_dpi)
        plot_logv_logg(results_df["logV"].to_numpy(), results_df["logG"].to_numpy(), results_df["pi"].to_numpy(), plots_dir / "logV_logG_pi.png", args.plot_dpi, "pi")
        plot_logv_logg(results_df["logV"].to_numpy(), results_df["logG"].to_numpy(), results_df["r"].to_numpy(), plots_dir / "logV_logG_r.png", args.plot_dpi, "r")

        train_mask = mask & is_train
        test_mask = mask & is_test
        train_ll = compute_pi_r_ll_simple(X[train_mask], y[train_mask], beta, alpha0, beta0, alpha1, beta1)

        baseline_test_metrics = None
        baseline_train_metrics = None
        if np.any(train_mask):
            const_beta, const_alpha0, const_beta0, const_alpha1, const_beta1, single_beta = run_constant_baselines(
                y[train_mask],
                args.max_iter,
                args.tol,
                args.init_threshold,
                args.init_soft,
                args.reg_strength,
                logger,
            )
            X_const_train = np.ones((int(np.sum(train_mask)), 1), dtype=float)
            const_train_ll = compute_pi_r_ll_simple(
                X_const_train,
                y[train_mask],
                const_beta,
                const_alpha0,
                const_beta0,
                const_alpha1,
                const_beta1,
            )
            single_train_ll = float(
                np.sum(
                    beta_logpdf(
                        np.log(y[train_mask]),
                        np.log1p(-y[train_mask]),
                        single_beta[0],
                        single_beta[1],
                    )
                )
            )
            baseline_train_metrics = {
                "constant_pi_mixture": float(const_train_ll),
                "single_beta": float(single_train_ll),
            }
            if np.any(test_mask):
                X_const_test = np.ones((int(np.sum(test_mask)), 1), dtype=float)
                const_test_ll = compute_pi_r_ll_simple(
                    X_const_test,
                    y[test_mask],
                    const_beta,
                    const_alpha0,
                    const_beta0,
                    const_alpha1,
                    const_beta1,
                )
                single_test_ll = float(
                    np.sum(
                        beta_logpdf(
                            np.log(y[test_mask]),
                            np.log1p(-y[test_mask]),
                            single_beta[0],
                            single_beta[1],
                        )
                    )
                )
                baseline_test_metrics = {
                    "constant_pi_mixture": float(const_test_ll),
                    "single_beta": float(single_test_ll),
                }
        else:
            const_beta = np.array([0.0])
            const_alpha0 = 1.0
            const_beta0 = 1.0
            const_alpha1 = 1.0
            const_beta1 = 1.0
            single_beta = (1.0, 1.0)

        test_metrics = None
        if np.any(test_mask):
            test_ll = compute_pi_r_ll_simple(X[test_mask], y[test_mask], beta, alpha0, beta0, alpha1, beta1)
            test_metrics = {
                "gated_mixture": float(test_ll),
                "constant_pi_mixture": None if baseline_test_metrics is None else baseline_test_metrics["constant_pi_mixture"],
                "single_beta": None if baseline_test_metrics is None else baseline_test_metrics["single_beta"],
            }

        train_metrics = {
            "gated_mixture": float(train_ll),
            "constant_pi_mixture": None if baseline_train_metrics is None else baseline_train_metrics["constant_pi_mixture"],
            "single_beta": None if baseline_train_metrics is None else baseline_train_metrics["single_beta"],
        }

        summary = {
            "resolution": int(resolution),
            "n_samples": int(mask.sum()),
            "vision_token_num": float(np.mean(results_df["vision_token_num"])),
            "gating_params": {
                "w0": shared_result.w0,
                "a": shared_result.a,
                "alpha": shared_result.alpha,
            },
            "beta_params": {
                "low_ed": {"alpha": shared_result.alpha0, "beta": shared_result.beta0, "mean": float(mean0), "var": float(var0)},
                "high_ed": {"alpha": shared_result.alpha1, "beta": shared_result.beta1, "mean": float(mean1), "var": float(var1)},
            },
            "responsibility_stats": summarize_responsibilities(results_df["r"].to_numpy()),
            "log_likelihood_history": shared_result.log_likelihood_history,
            "converged": shared_result.converged,
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
        with (output_dir / "em_fit_summary.json").open("w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)

        plot_grouped_ed_vs_text_len(
            results_df,
            float(mean0),
            float(mean1),
            plots_dir / "grouped_text_len_ed.png",
            args.plot_dpi,
        )

        history_df = pd.DataFrame(
            {
                "iteration": np.arange(len(shared_result.log_likelihood_history)),
                "log_likelihood": shared_result.log_likelihood_history,
            }
        )
        history_df.to_csv(output_dir / "em_loglik_history.csv", index=False)

        logger.info("Saved results for resolution=%d to %s", resolution, output_dir)


if __name__ == "__main__":
    main()
