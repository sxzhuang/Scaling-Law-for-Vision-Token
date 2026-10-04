"""Fit the lambda-corrected OCR scaling law with a two-Beta EM model.

The implementation-space gating equation is

    Z = b_resolution + a * log(V_code) + alpha * log(G) + c * log(Q),

where V_code = N * (w_f + c_s) / (h_f + l_s),
G = N / vision_token_num, and Q = (h_f + l_s) / h_f.

The paper uses V_paper proportional to sqrt(V_code). Therefore the equivalent
paper-space coefficient is omega_V = 2a and lambda_paper = c / (2a).
"""

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
DEFAULT_OUTPUT_DIR = Path("aaai_resub/lambda_scaling_model/outputs")
DEFAULT_RESOLUTIONS = "512,640,1024,1280"
MIN_BETA_SHAPE = 1e-3
MAX_BETA_SHAPE = 1e4


@dataclass
class FitResult:
    biases: np.ndarray
    a_code: float
    alpha_g: float
    c_log_q: float
    beta0_alpha: float
    beta0_beta: float
    beta1_alpha: float
    beta1_beta: float
    responsibilities: np.ndarray
    pi: np.ndarray
    log_likelihood_history: list[float]
    converged: bool


def log_sigmoid(values: np.ndarray) -> np.ndarray:
    return -np.logaddexp(0.0, -values)


def log1m_sigmoid(values: np.ndarray) -> np.ndarray:
    return -np.logaddexp(0.0, values)


def beta_logpdf(log_y: np.ndarray, log1m_y: np.ndarray, alpha: float, beta: float) -> np.ndarray:
    normalizer = gammaln(alpha + beta) - gammaln(alpha) - gammaln(beta)
    return (alpha - 1.0) * log_y + (beta - 1.0) * log1m_y + normalizer


def weighted_beta_moments(y: np.ndarray, weights: np.ndarray) -> tuple[float, float]:
    weights = np.where(np.isfinite(weights) & (weights > 0.0), weights, 0.0)
    weight_sum = float(weights.sum())
    if weight_sum <= 0.0:
        return 1.0, 1.0
    normalized = weights / weight_sum
    mean = float(np.sum(normalized * y))
    variance = float(np.sum(normalized * (y - mean) ** 2))
    if variance <= 1e-12 or not 0.0 < mean < 1.0:
        return 1.0, 1.0
    scale = mean * (1.0 - mean) / variance - 1.0
    if not np.isfinite(scale) or scale <= 0.0:
        return 1.0, 1.0
    return float(mean * scale), float((1.0 - mean) * scale)


def fit_weighted_beta(y: np.ndarray, weights: np.ndarray, initial: tuple[float, float] | None = None) -> tuple[float, float]:
    weights = np.where(np.isfinite(weights) & (weights > 0.0), weights, 0.0)
    weight_sum = float(weights.sum())
    if weight_sum <= 0.0:
        return 1.0, 1.0
    initial = weighted_beta_moments(y, weights) if initial is None else initial
    sum_log_y = float(np.sum(weights * np.log(y)))
    sum_log1m_y = float(np.sum(weights * np.log1p(-y)))

    def objective(log_shapes: np.ndarray) -> float:
        alpha, beta = np.exp(log_shapes)
        normalizer = gammaln(alpha + beta) - gammaln(alpha) - gammaln(beta)
        value = (alpha - 1.0) * sum_log_y + (beta - 1.0) * sum_log1m_y + weight_sum * normalizer
        return -float(value)

    def gradient(log_shapes: np.ndarray) -> np.ndarray:
        alpha, beta = np.exp(log_shapes)
        common = digamma(alpha + beta)
        grad_alpha = sum_log_y + weight_sum * (common - digamma(alpha))
        grad_beta = sum_log1m_y + weight_sum * (common - digamma(beta))
        return np.array([-grad_alpha * alpha, -grad_beta * beta], dtype=float)

    initial_array = np.clip(np.asarray(initial, dtype=float), MIN_BETA_SHAPE, MAX_BETA_SHAPE)
    result = minimize(
        objective,
        np.log(initial_array),
        jac=gradient,
        method="L-BFGS-B",
        bounds=[(math.log(MIN_BETA_SHAPE), math.log(MAX_BETA_SHAPE))] * 2,
        options={"maxiter": 200},
    )
    if not result.success or not np.all(np.isfinite(result.x)):
        return float(initial_array[0]), float(initial_array[1])
    shapes = np.exp(result.x)
    return float(shapes[0]), float(shapes[1])


def fit_grouped_logistic(features: np.ndarray, group_idx: np.ndarray, targets: np.ndarray, initial: np.ndarray, num_groups: int, reg_strength: float, observation_weights: np.ndarray | None = None) -> np.ndarray:
    if observation_weights is None:
        observation_weights = np.ones(targets.size, dtype=float)

    def objective(parameters: np.ndarray) -> float:
        z = parameters[:num_groups][group_idx] + features @ parameters[num_groups:]
        likelihood = np.sum(observation_weights * (targets * log_sigmoid(z) + (1.0 - targets) * log1m_sigmoid(z)))
        penalty = 0.5 * reg_strength * float(np.sum(parameters[num_groups:] ** 2))
        return -float(likelihood) + penalty

    def gradient(parameters: np.ndarray) -> np.ndarray:
        z = parameters[:num_groups][group_idx] + features @ parameters[num_groups:]
        difference = observation_weights * (expit(z) - targets)
        bias_gradient = np.bincount(group_idx, weights=difference, minlength=num_groups)
        feature_gradient = features.T @ difference + reg_strength * parameters[num_groups:]
        return np.concatenate([bias_gradient, feature_gradient])

    result = minimize(objective, initial, jac=gradient, method="L-BFGS-B", options={"maxiter": 200})
    if not result.success or not np.all(np.isfinite(result.x)):
        return initial
    return result.x.astype(float)


def mixture_outputs(features: np.ndarray, group_idx: np.ndarray, y: np.ndarray, parameters: np.ndarray, num_groups: int, shapes: tuple[float, float, float, float], observation_weights: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray, float]:
    alpha0, beta0, alpha1, beta1 = shapes
    z = parameters[:num_groups][group_idx] + features @ parameters[num_groups:]
    log_pi = log_sigmoid(z)
    log_component1 = beta_logpdf(np.log(y), np.log1p(-y), alpha1, beta1)
    log_component0 = beta_logpdf(np.log(y), np.log1p(-y), alpha0, beta0)
    log_mixture = np.logaddexp(log_pi + log_component1, log1m_sigmoid(z) + log_component0)
    responsibilities = np.exp(log_pi + log_component1 - log_mixture)
    if observation_weights is None:
        observation_weights = np.ones(y.size, dtype=float)
    return expit(z), responsibilities, float(np.sum(observation_weights * log_mixture))


def fit_em(features: np.ndarray, group_idx: np.ndarray, y: np.ndarray, num_groups: int, max_iter: int, tolerance: float, init_threshold: float, init_soft: float, reg_strength: float, logger: logging.Logger, observation_weights: np.ndarray | None = None) -> FitResult:
    if observation_weights is None:
        observation_weights = np.ones(y.size, dtype=float)
    observation_weights = np.asarray(observation_weights, dtype=float)
    if observation_weights.shape != y.shape or np.any(observation_weights < 0.0) or not np.any(observation_weights > 0.0):
        raise ValueError("observation_weights must match y, be nonnegative, and contain a positive value.")
    responsibilities = np.where(y >= init_threshold, 1.0 - init_soft, init_soft)
    alpha1, beta1 = fit_weighted_beta(y, observation_weights * responsibilities)
    alpha0, beta0 = fit_weighted_beta(y, observation_weights * (1.0 - responsibilities))
    parameters = np.zeros(num_groups + features.shape[1], dtype=float)
    for group in range(num_groups):
        mask = group_idx == group
        group_weights = observation_weights[mask]
        mean = float(np.average(responsibilities[mask], weights=group_weights)) if np.any(group_weights > 0.0) else 0.5
        mean = float(np.clip(mean, 1e-6, 1.0 - 1e-6))
        parameters[group] = math.log(mean / (1.0 - mean))

    history: list[float] = []
    converged = False
    for iteration in range(max_iter):
        shapes = (alpha0, beta0, alpha1, beta1)
        _, responsibilities, log_likelihood = mixture_outputs(features, group_idx, y, parameters, num_groups, shapes, observation_weights)
        history.append(log_likelihood)
        if iteration > 0 and abs(history[-1] - history[-2]) < tolerance:
            converged = True
            break
        parameters = fit_grouped_logistic(features, group_idx, responsibilities, parameters, num_groups, reg_strength, observation_weights)
        alpha1, beta1 = fit_weighted_beta(y, observation_weights * responsibilities, (alpha1, beta1))
        alpha0, beta0 = fit_weighted_beta(y, observation_weights * (1.0 - responsibilities), (alpha0, beta0))
        if iteration % 10 == 0:
            logger.info("EM iteration %d: log-likelihood=%.6f", iteration, log_likelihood)

    mean0 = alpha0 / (alpha0 + beta0)
    mean1 = alpha1 / (alpha1 + beta1)
    if mean1 < mean0:
        alpha0, alpha1 = alpha1, alpha0
        beta0, beta1 = beta1, beta0
        parameters = -parameters
    pi, responsibilities, _ = mixture_outputs(features, group_idx, y, parameters, num_groups, (alpha0, beta0, alpha1, beta1), observation_weights)
    return FitResult(
        biases=parameters[:num_groups],
        a_code=float(parameters[num_groups]),
        alpha_g=float(parameters[num_groups + 1]),
        c_log_q=float(parameters[num_groups + 2]),
        beta0_alpha=float(alpha0),
        beta0_beta=float(beta0),
        beta1_alpha=float(alpha1),
        beta1_beta=float(beta1),
        responsibilities=responsibilities,
        pi=pi,
        log_likelihood_history=history,
        converged=converged,
    )


def parse_resolutions(raw: str) -> list[int]:
    resolutions = [int(value.strip()) for value in raw.split(",") if value.strip()]
    if not resolutions:
        raise ValueError("At least one resolution is required.")
    return resolutions


def load_data(csv_path: Path, resolutions: list[int], eps: float, max_rows: int, seed: int) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    required = {"resolution", "text_len", "edit_distance", "font_size", "line_spacing", "letter_spacing", "character_width", "character_height"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing CSV columns: {sorted(missing)}")
    df = df[df["resolution"].isin(resolutions)].copy()
    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=list(required))
    if max_rows > 0 and len(df) > max_rows:
        df = df.sample(n=max_rows, random_state=seed).sort_index().copy()

    text_len = df["text_len"].astype(float)
    character_width = df["character_width"].astype(float)
    character_height = df["character_height"].astype(float)
    letter_spacing = df["letter_spacing"].astype(float)
    line_spacing = df["line_spacing"].astype(float)
    resolution = df["resolution"].astype(float)
    vision_token_num = (resolution / 64.0) ** 2
    width_pitch = character_width + letter_spacing
    line_pitch = character_height + line_spacing

    valid = (text_len > 0.0) & (character_width > 0.0) & (width_pitch > 0.0) & (line_pitch > 0.0) & (vision_token_num > 0.0)
    df = df.loc[valid].copy()
    text_len = text_len.loc[valid]
    character_height = character_height.loc[valid]
    width_pitch = width_pitch.loc[valid]
    line_pitch = line_pitch.loc[valid]
    vision_token_num = vision_token_num.loc[valid]

    df["vision_token_num"] = vision_token_num
    df["G"] = text_len / vision_token_num
    df["V_code"] = text_len * width_pitch / line_pitch
    df["Q"] = line_pitch / character_height
    df["log_G"] = np.log(np.maximum(df["G"].astype(float), eps))
    df["log_V_code"] = np.log(np.maximum(df["V_code"].astype(float), eps))
    df["log_Q"] = np.log(np.maximum(df["Q"].astype(float), eps))
    df["edit_distance_raw"] = df["edit_distance"].astype(float)
    df["edit_distance"] = np.clip(df["edit_distance_raw"].astype(float), eps, 1.0 - eps)
    finite = np.isfinite(df[["log_G", "log_V_code", "log_Q", "edit_distance"]]).all(axis=1)
    df = df.loc[finite].copy()
    if df.empty:
        raise ValueError("No valid rows remain after feature construction.")
    df["resolution"] = df["resolution"].astype(int)
    return df


def save_plots(df: pd.DataFrame, history: list[float], plots_dir: Path, dpi: int) -> None:
    plots_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(np.arange(len(history)), history)
    ax.set_xlabel("EM iteration")
    ax.set_ylabel("Log-likelihood")
    fig.tight_layout()
    fig.savefig(plots_dir / "loglik_history.png", dpi=dpi)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(df["Z"], df["edit_distance_raw"], c=df["pi"], s=4, alpha=0.25, cmap="viridis")
    ax.set_xlabel("Fitted pressure Z")
    ax.set_ylabel("Edit distance")
    fig.tight_layout()
    fig.savefig(plots_dir / "pressure_vs_edit_distance.png", dpi=dpi)
    plt.close(fig)

    layout_summary = df.groupby(["font_size", "line_spacing", "letter_spacing"], as_index=False).agg(mean_pi=("pi", "mean"), mean_ed=("edit_distance_raw", "mean"), mean_q=("Q", "mean"))
    fig, ax = plt.subplots(figsize=(7, 5))
    scatter = ax.scatter(layout_summary["mean_q"], layout_summary["mean_ed"], c=layout_summary["mean_pi"], s=45, cmap="viridis")
    fig.colorbar(scatter, ax=ax, label="Mean fitted instability probability")
    ax.set_xlabel("Q = (character_height + line_spacing) / character_height")
    ax.set_ylabel("Mean edit distance")
    fig.tight_layout()
    fig.savefig(plots_dir / "layout_q_summary.png", dpi=dpi)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Fit the lambda-corrected OCR scaling law.")
    parser.add_argument("--csv_path", type=Path, default=DEFAULT_CSV_PATH, help="Input feature/evaluation CSV.")
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Output directory.")
    parser.add_argument("--resolutions", type=str, default=DEFAULT_RESOLUTIONS, help="Comma-separated resolutions.")
    parser.add_argument("--max_iter", type=int, default=200, help="Maximum EM iterations.")
    parser.add_argument("--tol", type=float, default=1e-6, help="Absolute log-likelihood convergence tolerance.")
    parser.add_argument("--eps", type=float, default=1e-6, help="Numerical clipping epsilon.")
    parser.add_argument("--init_threshold", type=float, default=0.37, help="Edit-distance initialization threshold.")
    parser.add_argument("--init_soft", type=float, default=0.05, help="Initial soft-assignment probability.")
    parser.add_argument("--reg_strength", type=float, default=0.0, help="L2 penalty on shared slopes.")
    parser.add_argument("--max_rows", type=int, default=0, help="Optional random row limit for smoke tests; zero uses all rows.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed used by max_rows sampling.")
    parser.add_argument("--plot_dpi", type=int, default=200, help="Plot resolution.")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")
    logger = logging.getLogger("fit_lambda_scaling_law")
    resolutions = parse_resolutions(args.resolutions)
    df = load_data(args.csv_path, resolutions, args.eps, args.max_rows, args.seed)

    resolution_values = sorted(df["resolution"].unique().tolist())
    resolution_to_index = {value: index for index, value in enumerate(resolution_values)}
    group_idx = df["resolution"].map(resolution_to_index).to_numpy(dtype=int)
    features = df[["log_V_code", "log_G", "log_Q"]].to_numpy(dtype=float)
    y = df["edit_distance"].to_numpy(dtype=float)

    excluded = ((df["font_size"].astype(float) == 28.0) & (df["line_spacing"].astype(float) == 6.0) & (df["letter_spacing"].astype(float) == 0.0)).to_numpy()
    train_indices = np.flatnonzero(~excluded)
    if train_indices.size == 0:
        raise ValueError("All rows were excluded by the 28_6_0 holdout rule.")
    logger.info("Loaded %d rows; fitting %d rows and holding out %d rows from layout 28_6_0", len(df), train_indices.size, int(excluded.sum()))

    result = fit_em(features[train_indices], group_idx[train_indices], y[train_indices], len(resolution_values), args.max_iter, args.tol, args.init_threshold, args.init_soft, args.reg_strength, logger)
    parameters = np.concatenate([result.biases, [result.a_code, result.alpha_g, result.c_log_q]])
    pi, responsibilities, full_log_likelihood = mixture_outputs(features, group_idx, y, parameters, len(resolution_values), (result.beta0_alpha, result.beta0_beta, result.beta1_alpha, result.beta1_beta))
    z = result.biases[group_idx] + features @ np.array([result.a_code, result.alpha_g, result.c_log_q])

    denominator_code = result.a_code
    denominator_paper = 2.0 * result.a_code
    lambda_code = result.c_log_q / denominator_code if abs(denominator_code) > 1e-12 else None
    lambda_paper = result.c_log_q / denominator_paper if abs(denominator_paper) > 1e-12 else None
    mean0 = result.beta0_alpha / (result.beta0_alpha + result.beta0_beta)
    mean1 = result.beta1_alpha / (result.beta1_alpha + result.beta1_beta)

    df["excluded_28_6_0"] = excluded
    df["Z"] = z
    df["pi"] = pi
    df["responsibility"] = responsibilities
    df["expected_edit_distance"] = (1.0 - pi) * mean0 + pi * mean1

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "model": {
            "implementation_equation": "Z = b_resolution + a_code*log(V_code) + alpha_g*log(G) + c_log_q*log(Q)",
            "V_code": "text_len * (character_width + letter_spacing) / (character_height + line_spacing)",
            "G": "text_len / (resolution / 64)^2",
            "Q": "(character_height + line_spacing) / character_height",
            "paper_equation": "Z = b_resolution + omega_V*log(V_paper * Q^lambda_paper) + omega_G*log(G)",
            "V_relation": "V_paper is proportional to sqrt(V_code); resolution/aspect constants are absorbed by b_resolution",
            "coefficient_conversion": "omega_V = 2*a_code; lambda_paper = c_log_q/(2*a_code)",
        },
        "csv_variables": {
            "text_len": "N_i, character count",
            "character_width": "w_f, measured glyph width for the font size",
            "character_height": "h_f, measured glyph height for the font size",
            "letter_spacing": "c_s, additional horizontal character spacing",
            "line_spacing": "l_s, additional vertical line spacing",
            "resolution": "input image/model resolution; vision_token_num=(resolution/64)^2",
            "edit_distance": "normalized OCR edit distance used as the response",
        },
        "data": {
            "csv_path": str(args.csv_path),
            "num_rows": int(len(df)),
            "num_fit_rows": int(train_indices.size),
            "num_excluded_28_6_0": int(excluded.sum()),
            "resolutions": resolution_values,
        },
        "fit": {
            "converged": result.converged,
            "iterations": len(result.log_likelihood_history),
            "full_data_log_likelihood": full_log_likelihood,
            "resolution_biases": {str(resolution): float(result.biases[index]) for index, resolution in enumerate(resolution_values)},
            "a_code": result.a_code,
            "alpha_g": result.alpha_g,
            "c_log_q": result.c_log_q,
            "omega_v_paper": 2.0 * result.a_code,
            "lambda_code": lambda_code,
            "lambda_paper": lambda_paper,
            "beta_components": {
                "stable": {"alpha": result.beta0_alpha, "beta": result.beta0_beta, "mean_edit_distance": mean0},
                "unstable": {"alpha": result.beta1_alpha, "beta": result.beta1_beta, "mean_edit_distance": mean1},
            },
        },
    }
    with (args.output_dir / "fit_summary.json").open("w", encoding="utf-8") as file:
        json.dump(summary, file, ensure_ascii=False, indent=2)
    df.to_csv(args.output_dir / "fit_points.csv", index=False)
    pd.DataFrame({"iteration": np.arange(len(result.log_likelihood_history)), "log_likelihood": result.log_likelihood_history}).to_csv(args.output_dir / "loglik_history.csv", index=False)
    save_plots(df, result.log_likelihood_history, args.output_dir / "plots", args.plot_dpi)
    logger.info("lambda_code=%s, lambda_paper=%s", lambda_code, lambda_paper)
    logger.info("Saved outputs to %s", args.output_dir)


if __name__ == "__main__":
    main()
