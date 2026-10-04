"""Estimate lambda by decomposing gating intercepts into resolution and layout parts."""

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
from scipy.special import expit

from fit_lambda_scaling_law import beta_logpdf, fit_em, fit_weighted_beta, load_data, log1m_sigmoid, log_sigmoid, parse_resolutions


DEFAULT_CSV_PATH = Path("data_analysis/data4analysis.csv")
DEFAULT_OUTPUT_DIR = Path("aaai_resub/lambda_scaling_model/outputs/layout_intercept_method")


@dataclass
class InterceptFitResult:
    resolution_biases: np.ndarray
    layout_offsets: np.ndarray
    beta0_alpha: float
    beta0_beta: float
    beta1_alpha: float
    beta1_beta: float
    pi: np.ndarray
    responsibilities: np.ndarray
    history: list[float]
    converged: bool


def fit_intercept_logistic(offset: np.ndarray, group_idx: np.ndarray, layout_idx: np.ndarray, targets: np.ndarray, initial: np.ndarray, num_groups: int, num_layouts: int) -> np.ndarray:
    num_layout_parameters = num_layouts - 1

    def unpack(parameters: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        biases = parameters[:num_groups]
        layout_offsets = np.zeros(num_layouts, dtype=float)
        layout_offsets[1:] = parameters[num_groups:num_groups + num_layout_parameters]
        return biases, layout_offsets

    def objective(parameters: np.ndarray) -> float:
        biases, layout_offsets = unpack(parameters)
        z = offset + biases[group_idx] + layout_offsets[layout_idx]
        likelihood = np.sum(targets * log_sigmoid(z) + (1.0 - targets) * log1m_sigmoid(z))
        return -float(likelihood)

    def gradient(parameters: np.ndarray) -> np.ndarray:
        biases, layout_offsets = unpack(parameters)
        z = offset + biases[group_idx] + layout_offsets[layout_idx]
        difference = expit(z) - targets
        bias_gradient = np.bincount(group_idx, weights=difference, minlength=num_groups)
        layout_gradient = np.bincount(layout_idx, weights=difference, minlength=num_layouts)[1:]
        return np.concatenate([bias_gradient, layout_gradient])

    result = minimize(objective, initial, jac=gradient, method="L-BFGS-B", options={"maxiter": 300})
    if not result.success or not np.all(np.isfinite(result.x)):
        return initial
    return result.x.astype(float)


def intercept_mixture_outputs(offset: np.ndarray, group_idx: np.ndarray, layout_idx: np.ndarray, resolution_biases: np.ndarray, layout_offsets: np.ndarray, y: np.ndarray, shapes: tuple[float, float, float, float]) -> tuple[np.ndarray, np.ndarray, float]:
    alpha0, beta0, alpha1, beta1 = shapes
    z = offset + resolution_biases[group_idx] + layout_offsets[layout_idx]
    log_pi = log_sigmoid(z)
    log_component1 = beta_logpdf(np.log(y), np.log1p(-y), alpha1, beta1)
    log_component0 = beta_logpdf(np.log(y), np.log1p(-y), alpha0, beta0)
    log_mixture = np.logaddexp(log_pi + log_component1, log1m_sigmoid(z) + log_component0)
    responsibilities = np.exp(log_pi + log_component1 - log_mixture)
    return expit(z), responsibilities, float(np.sum(log_mixture))


def fit_layout_intercepts(offset: np.ndarray, group_idx: np.ndarray, layout_idx: np.ndarray, y: np.ndarray, num_groups: int, num_layouts: int, max_iter: int, tolerance: float, init_threshold: float, init_soft: float, logger: logging.Logger) -> InterceptFitResult:
    responsibilities = np.where(y >= init_threshold, 1.0 - init_soft, init_soft)
    alpha1, beta1 = fit_weighted_beta(y, responsibilities)
    alpha0, beta0 = fit_weighted_beta(y, 1.0 - responsibilities)
    parameters = np.zeros(num_groups + num_layouts - 1, dtype=float)
    for group in range(num_groups):
        mask = group_idx == group
        probability = float(np.clip(np.mean(responsibilities[mask]), 1e-6, 1.0 - 1e-6))
        parameters[group] = math.log(probability / (1.0 - probability)) - float(np.mean(offset[mask]))

    history: list[float] = []
    converged = False
    for iteration in range(max_iter):
        resolution_biases = parameters[:num_groups]
        layout_offsets = np.zeros(num_layouts, dtype=float)
        layout_offsets[1:] = parameters[num_groups:]
        shapes = (alpha0, beta0, alpha1, beta1)
        _, responsibilities, log_likelihood = intercept_mixture_outputs(offset, group_idx, layout_idx, resolution_biases, layout_offsets, y, shapes)
        history.append(log_likelihood)
        if iteration > 0 and abs(history[-1] - history[-2]) < tolerance:
            converged = True
            break
        parameters = fit_intercept_logistic(offset, group_idx, layout_idx, responsibilities, parameters, num_groups, num_layouts)
        alpha1, beta1 = fit_weighted_beta(y, responsibilities, (alpha1, beta1))
        alpha0, beta0 = fit_weighted_beta(y, 1.0 - responsibilities, (alpha0, beta0))
        if iteration % 10 == 0:
            logger.info("Layout-intercept EM iteration %d: log-likelihood=%.6f", iteration, log_likelihood)

    if alpha1 / (alpha1 + beta1) <= alpha0 / (alpha0 + beta0):
        raise RuntimeError("The unstable Beta component was not preserved during intercept fitting.")
    resolution_biases = parameters[:num_groups]
    layout_offsets = np.zeros(num_layouts, dtype=float)
    layout_offsets[1:] = parameters[num_groups:]
    mean_layout_offset = float(np.mean(layout_offsets))
    layout_offsets = layout_offsets - mean_layout_offset
    resolution_biases = resolution_biases + mean_layout_offset
    pi, responsibilities, _ = intercept_mixture_outputs(offset, group_idx, layout_idx, resolution_biases, layout_offsets, y, (alpha0, beta0, alpha1, beta1))
    return InterceptFitResult(resolution_biases, layout_offsets, alpha0, beta0, alpha1, beta1, pi, responsibilities, history, converged)


def categorical_dummies(values: pd.Series) -> tuple[np.ndarray, list[str]]:
    levels = sorted(values.unique().tolist())
    if len(levels) <= 1:
        return np.empty((len(values), 0), dtype=float), []
    columns = [(values.to_numpy() == level).astype(float) for level in levels[1:]]
    names = [str(level) for level in levels[1:]]
    return np.column_stack(columns), names


def regress_layout_offsets(layout_df: pd.DataFrame, controls: list[str], a_code: float) -> dict[str, object]:
    columns = [np.ones(len(layout_df), dtype=float), layout_df["log_Q"].to_numpy(dtype=float)]
    coefficient_names = ["intercept", "c_log_q"]
    for control in controls:
        dummies, levels = categorical_dummies(layout_df[control])
        if dummies.shape[1] > 0:
            columns.append(dummies)
            coefficient_names.extend([f"{control}[{level}]" for level in levels])
    design = np.column_stack(columns)
    target = layout_df["layout_offset"].to_numpy(dtype=float)
    coefficients, _, rank, _ = np.linalg.lstsq(design, target, rcond=None)
    fitted = design @ coefficients
    residuals = target - fitted
    residual_sum_squares = float(np.sum(residuals**2))
    total_sum_squares = float(np.sum((target - np.mean(target)) ** 2))
    degrees_of_freedom = len(target) - int(rank)
    residual_variance = residual_sum_squares / degrees_of_freedom if degrees_of_freedom > 0 else float("nan")
    covariance = residual_variance * np.linalg.pinv(design.T @ design)
    standard_errors = np.sqrt(np.maximum(np.diag(covariance), 0.0))
    c_log_q = float(coefficients[1])
    c_standard_error = float(standard_errors[1])
    lambda_paper = c_log_q / (2.0 * a_code)
    lambda_standard_error_conditional = c_standard_error / abs(2.0 * a_code)
    return {
        "controls": controls,
        "num_layouts": int(len(layout_df)),
        "rank": int(rank),
        "r_squared": 1.0 - residual_sum_squares / total_sum_squares if total_sum_squares > 0.0 else float("nan"),
        "residual_std": math.sqrt(residual_variance) if residual_variance >= 0.0 else float("nan"),
        "coefficients": {name: float(value) for name, value in zip(coefficient_names, coefficients)},
        "standard_errors": {name: float(value) for name, value in zip(coefficient_names, standard_errors)},
        "c_log_q": c_log_q,
        "c_log_q_standard_error": c_standard_error,
        "lambda_paper": lambda_paper,
        "lambda_paper_standard_error_conditional_on_a": lambda_standard_error_conditional,
        "lambda_paper_normal_95_interval_conditional_on_a": [lambda_paper - 1.96 * lambda_standard_error_conditional, lambda_paper + 1.96 * lambda_standard_error_conditional],
        "fitted_layout_offsets": fitted.tolist(),
        "residuals": residuals.tolist(),
    }


def save_plot(layout_df: pd.DataFrame, regression: dict[str, object], path: Path, dpi: int) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = {-1: "tab:blue", 0: "tab:orange", 7: "tab:green"}
    coefficients = regression["coefficients"]
    x_values = np.linspace(float(layout_df["log_Q"].min()), float(layout_df["log_Q"].max()), 100)
    for letter_spacing, group in layout_df.groupby("letter_spacing", sort=True):
        color = colors.get(int(letter_spacing), None)
        ax.scatter(group["log_Q"], group["layout_offset"], color=color, label=f"letter spacing={letter_spacing}", s=42)
        adjustment = 0.0 if int(letter_spacing) == -1 else float(coefficients.get(f"letter_spacing[{letter_spacing}]", 0.0))
        fitted = float(coefficients["intercept"]) + adjustment + float(coefficients["c_log_q"]) * x_values
        ax.plot(x_values, fitted, color=color, linewidth=1.5)
    ax.set_xlabel("log Q")
    ax.set_ylabel("Fitted layout intercept offset")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=dpi)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Estimate lambda from fitted layout-specific intercept shifts.")
    parser.add_argument("--csv_path", type=Path, default=DEFAULT_CSV_PATH, help="Input feature/evaluation CSV.")
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Output directory.")
    parser.add_argument("--resolutions", type=str, default="512,640,1024,1280", help="Comma-separated resolutions.")
    parser.add_argument("--max_iter", type=int, default=200, help="Maximum EM iterations per stage.")
    parser.add_argument("--tol", type=float, default=1e-6, help="EM convergence tolerance.")
    parser.add_argument("--eps", type=float, default=1e-6, help="Numerical clipping epsilon.")
    parser.add_argument("--init_threshold", type=float, default=0.37, help="Edit-distance initialization threshold.")
    parser.add_argument("--init_soft", type=float, default=0.05, help="Initial soft assignment.")
    parser.add_argument("--reg_strength", type=float, default=0.0, help="L2 penalty in the original two-feature fit.")
    parser.add_argument("--max_rows", type=int, default=0, help="Optional row limit for smoke testing.")
    parser.add_argument("--seed", type=int, default=42, help="Sampling seed for max_rows.")
    parser.add_argument("--plot_dpi", type=int, default=200, help="Plot DPI.")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")
    logger = logging.getLogger("fit_lambda_via_layout_intercepts")
    df = load_data(args.csv_path, parse_resolutions(args.resolutions), args.eps, args.max_rows, args.seed)
    resolutions = sorted(df["resolution"].unique().tolist())
    resolution_to_index = {resolution: index for index, resolution in enumerate(resolutions)}
    group_idx = df["resolution"].map(resolution_to_index).to_numpy(dtype=int)
    y = df["edit_distance"].to_numpy(dtype=float)

    layout_columns = ["font_size", "line_spacing", "letter_spacing"]
    layout_index = pd.MultiIndex.from_frame(df[layout_columns].astype(int))
    layout_idx, layout_levels = pd.factorize(layout_index, sort=True)
    if len(layout_levels) != 36 and args.max_rows == 0:
        raise ValueError(f"Expected 36 layouts, found {len(layout_levels)}.")
    logger.info("Stage 1: fitting original log(V_code) + log(G) model on %d rows and %d layouts", len(df), len(layout_levels))

    original_features = np.column_stack([df["log_V_code"].to_numpy(dtype=float), df["log_G"].to_numpy(dtype=float), np.zeros(len(df), dtype=float)])
    original_result = fit_em(original_features, group_idx, y, len(resolutions), args.max_iter, args.tol, args.init_threshold, args.init_soft, args.reg_strength, logger)
    fixed_offset = original_result.a_code * df["log_V_code"].to_numpy(dtype=float) + original_result.alpha_g * df["log_G"].to_numpy(dtype=float)
    logger.info("Stage 1 slopes: a_code=%.6f, alpha_g=%.6f", original_result.a_code, original_result.alpha_g)

    logger.info("Stage 2: fixing slopes and fitting shared resolution biases plus layout offsets")
    intercept_result = fit_layout_intercepts(fixed_offset, group_idx, layout_idx, y, len(resolutions), len(layout_levels), args.max_iter, args.tol, args.init_threshold, args.init_soft, logger)
    layout_df = pd.DataFrame(layout_levels.tolist(), columns=layout_columns)
    layout_df["layout_id"] = np.arange(len(layout_df))
    layout_df["layout_offset"] = intercept_result.layout_offsets
    character_heights = df.groupby(layout_columns, as_index=False)["character_height"].first()
    layout_df = layout_df.merge(character_heights, on=layout_columns, how="left", validate="one_to_one")
    layout_df["Q"] = (layout_df["character_height"] + layout_df["line_spacing"]) / layout_df["character_height"]
    layout_df["log_Q"] = np.log(layout_df["Q"])

    regressions = {
        "log_q_only": regress_layout_offsets(layout_df, [], original_result.a_code),
        "primary_log_q_plus_letter_spacing": regress_layout_offsets(layout_df, ["letter_spacing"], original_result.a_code),
        "log_q_plus_letter_spacing_and_font_size": regress_layout_offsets(layout_df, ["letter_spacing", "font_size"], original_result.a_code),
    }
    primary = regressions["primary_log_q_plus_letter_spacing"]
    layout_df["fitted_offset_primary"] = primary["fitted_layout_offsets"]
    layout_df["residual_primary"] = primary["residuals"]

    mean0 = intercept_result.beta0_alpha / (intercept_result.beta0_alpha + intercept_result.beta0_beta)
    mean1 = intercept_result.beta1_alpha / (intercept_result.beta1_alpha + intercept_result.beta1_beta)
    points = df.copy()
    points["layout_id"] = layout_idx
    points["layout_offset"] = intercept_result.layout_offsets[layout_idx]
    points["pi"] = intercept_result.pi
    points["responsibility"] = intercept_result.responsibilities
    points["expected_edit_distance"] = (1.0 - intercept_result.pi) * mean0 + intercept_result.pi * mean1

    args.output_dir.mkdir(parents=True, exist_ok=True)
    layout_df.to_csv(args.output_dir / "layout_intercepts.csv", index=False)
    points.to_csv(args.output_dir / "fit_points.csv", index=False)
    pd.DataFrame({"iteration": np.arange(len(original_result.log_likelihood_history)), "log_likelihood": original_result.log_likelihood_history}).to_csv(args.output_dir / "stage1_loglik_history.csv", index=False)
    pd.DataFrame({"iteration": np.arange(len(intercept_result.history)), "log_likelihood": intercept_result.history}).to_csv(args.output_dir / "stage2_loglik_history.csv", index=False)
    save_plot(layout_df, primary, args.output_dir / "layout_offset_vs_log_q.png", args.plot_dpi)

    summary = {
        "method": {
            "stage1": "Fit the original gating model Z=b_resolution+a_code*log(V_code)+alpha_g*log(G), without log(Q), using all 36 layouts.",
            "stage2": "Fix a_code and alpha_g; fit Z=b_resolution+d_layout+a_code*log(V_code)+alpha_g*log(G).",
            "stage3": "Regress the 36 fitted d_layout values on log(Q), with optional categorical controls.",
            "lambda_conversion": "lambda_paper=c_log_q/(2*a_code) because V_paper is proportional to sqrt(V_code).",
        },
        "data": {"rows": int(len(df)), "layouts": int(len(layout_levels)), "resolutions": resolutions, "excluded_layouts": []},
        "stage1_original_model": {
            "converged": original_result.converged,
            "iterations": len(original_result.log_likelihood_history),
            "a_code": original_result.a_code,
            "alpha_g": original_result.alpha_g,
            "resolution_biases": {str(resolution): float(original_result.biases[index]) for index, resolution in enumerate(resolutions)},
        },
        "stage2_layout_intercepts": {
            "converged": intercept_result.converged,
            "iterations": len(intercept_result.history),
            "resolution_biases": {str(resolution): float(intercept_result.resolution_biases[index]) for index, resolution in enumerate(resolutions)},
            "layout_offset_mean": float(np.mean(intercept_result.layout_offsets)),
            "layout_offset_std": float(np.std(intercept_result.layout_offsets)),
            "stable_mean_edit_distance": mean0,
            "unstable_mean_edit_distance": mean1,
        },
        "stage3_layout_offset_regressions": regressions,
    }
    with (args.output_dir / "fit_summary.json").open("w", encoding="utf-8") as file:
        json.dump(summary, file, ensure_ascii=False, indent=2)
    logger.info("Primary intercept-regression lambda_paper=%.6f, R^2=%.6f", float(primary["lambda_paper"]), float(primary["r_squared"]))
    logger.info("Saved layout-intercept results to %s", args.output_dir)


if __name__ == "__main__":
    main()
