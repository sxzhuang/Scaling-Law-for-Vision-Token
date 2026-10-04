"""Estimate lambda from layout-resolution intercepts with original parameters fixed."""

import argparse
import json
import logging
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar

from fit_lambda_scaling_law import beta_logpdf, load_data, log1m_sigmoid, log_sigmoid, parse_resolutions


DEFAULT_CSV_PATH = Path("data_analysis/data4analysis.csv")
DEFAULT_ORIGINAL_RESULTS_DIR = Path("data_analysis/em_fiting_res/shared_exceptw")
DEFAULT_OUTPUT_DIR = Path("aaai_resub/lambda_scaling_model/outputs/fixed_original_intercepts")


def load_original_parameters(results_dir: Path, resolutions: list[int], tolerance: float = 1e-10) -> dict[str, object]:
    records: dict[int, dict[str, object]] = {}
    for resolution in resolutions:
        path = results_dir / str(resolution) / "em_fit_summary.json"
        if not path.exists():
            raise FileNotFoundError(f"Missing original fit summary: {path}")
        with path.open("r", encoding="utf-8") as file:
            records[resolution] = json.load(file)

    reference = records[resolutions[0]]
    a_code = float(reference["gating_params"]["a"])
    alpha_g = float(reference["gating_params"]["alpha"])
    beta_shapes = {
        "beta0_alpha": float(reference["beta_params"]["low_ed"]["alpha"]),
        "beta0_beta": float(reference["beta_params"]["low_ed"]["beta"]),
        "beta1_alpha": float(reference["beta_params"]["high_ed"]["alpha"]),
        "beta1_beta": float(reference["beta_params"]["high_ed"]["beta"]),
    }
    original_biases: dict[int, float] = {}
    for resolution, record in records.items():
        current_a = float(record["gating_params"]["a"])
        current_alpha = float(record["gating_params"]["alpha"])
        current_shapes = {
            "beta0_alpha": float(record["beta_params"]["low_ed"]["alpha"]),
            "beta0_beta": float(record["beta_params"]["low_ed"]["beta"]),
            "beta1_alpha": float(record["beta_params"]["high_ed"]["alpha"]),
            "beta1_beta": float(record["beta_params"]["high_ed"]["beta"]),
        }
        if abs(current_a - a_code) > tolerance or abs(current_alpha - alpha_g) > tolerance:
            raise ValueError("Original shared slopes differ across resolution summaries.")
        if any(abs(current_shapes[name] - beta_shapes[name]) > tolerance for name in beta_shapes):
            raise ValueError("Original shared Beta parameters differ across resolution summaries.")
        original_biases[resolution] = float(record["gating_params"]["w0"])
    return {"a_code": a_code, "alpha_g": alpha_g, "beta_shapes": beta_shapes, "original_resolution_biases": original_biases}


def intercept_log_likelihood(intercept: float, fixed_offset: np.ndarray, log_component0: np.ndarray, log_component1: np.ndarray) -> float:
    z = fixed_offset + float(intercept)
    log_mixture = np.logaddexp(log_sigmoid(z) + log_component1, log1m_sigmoid(z) + log_component0)
    return float(np.sum(log_mixture))


def fit_one_intercept(fixed_offset: np.ndarray, y: np.ndarray, shapes: tuple[float, float, float, float], initial: float) -> tuple[float, float, float, bool]:
    alpha0, beta0, alpha1, beta1 = shapes
    log_component0 = beta_logpdf(np.log(y), np.log1p(-y), alpha0, beta0)
    log_component1 = beta_logpdf(np.log(y), np.log1p(-y), alpha1, beta1)

    def objective(intercept: float) -> float:
        return -intercept_log_likelihood(intercept, fixed_offset, log_component0, log_component1)

    lower = float(initial) - 30.0
    upper = float(initial) + 30.0
    result = minimize_scalar(objective, bounds=(lower, upper), method="bounded", options={"xatol": 1e-8, "maxiter": 500})
    estimate = float(result.x)
    log_likelihood = -float(result.fun)
    step = 1e-3 * max(1.0, abs(estimate))
    ll_left = intercept_log_likelihood(estimate - step, fixed_offset, log_component0, log_component1)
    ll_right = intercept_log_likelihood(estimate + step, fixed_offset, log_component0, log_component1)
    second_derivative = (ll_left - 2.0 * log_likelihood + ll_right) / (step**2)
    standard_error = math.sqrt(-1.0 / second_derivative) if second_derivative < -1e-12 else float("nan")
    return estimate, standard_error, log_likelihood, bool(result.success)


def fit_all_intercepts(df: pd.DataFrame, fixed_offset: np.ndarray, shapes: tuple[float, float, float, float], original_biases: dict[int, float], logger: logging.Logger) -> pd.DataFrame:
    rows: list[dict[str, float | int | bool]] = []
    group_columns = ["resolution", "font_size", "line_spacing", "letter_spacing"]
    for values, group in df.groupby(group_columns, sort=True):
        resolution, font_size, line_spacing, letter_spacing = values
        indices = group.index.to_numpy(dtype=int)
        estimate, standard_error, log_likelihood, success = fit_one_intercept(fixed_offset[indices], df.loc[indices, "edit_distance"].to_numpy(dtype=float), shapes, original_biases[int(resolution)])
        character_height = float(group["character_height"].iloc[0])
        q_value = (character_height + float(line_spacing)) / character_height
        rows.append({
            "resolution": int(resolution),
            "font_size": int(font_size),
            "line_spacing": int(line_spacing),
            "letter_spacing": int(letter_spacing),
            "num_samples": int(len(group)),
            "Q": q_value,
            "log_Q": math.log(q_value),
            "combined_intercept": estimate,
            "intercept_standard_error": standard_error,
            "log_likelihood": log_likelihood,
            "optimizer_success": success,
        })
    result_df = pd.DataFrame(rows)
    logger.info("Fitted %d layout-resolution intercepts; optimizer successes=%d", len(result_df), int(result_df["optimizer_success"].sum()))
    return result_df


def categorical_dummies(values: pd.Series) -> tuple[np.ndarray, list[str]]:
    levels = sorted(values.unique().tolist())
    if len(levels) <= 1:
        return np.empty((len(values), 0), dtype=float), []
    return np.column_stack([(values.to_numpy() == level).astype(float) for level in levels[1:]]), [str(level) for level in levels[1:]]


def regress_intercepts(intercepts: pd.DataFrame, controls: list[str], a_code: float, precision_weighted: bool) -> dict[str, object]:
    columns = [np.ones(len(intercepts), dtype=float)]
    coefficient_names = ["intercept"]
    resolution_dummies, resolution_levels = categorical_dummies(intercepts["resolution"])
    columns.append(resolution_dummies)
    coefficient_names.extend([f"resolution[{level}]" for level in resolution_levels])
    columns.append(intercepts["log_Q"].to_numpy(dtype=float))
    coefficient_names.append("c_log_q")
    for control in controls:
        dummies, levels = categorical_dummies(intercepts[control])
        if dummies.shape[1] > 0:
            columns.append(dummies)
            coefficient_names.extend([f"{control}[{level}]" for level in levels])
    design = np.column_stack(columns)
    target = intercepts["combined_intercept"].to_numpy(dtype=float)
    if precision_weighted:
        standard_errors = intercepts["intercept_standard_error"].to_numpy(dtype=float)
        valid_standard_errors = standard_errors[np.isfinite(standard_errors) & (standard_errors > 0.0)]
        fallback = float(np.median(valid_standard_errors)) if valid_standard_errors.size > 0 else 1.0
        standard_errors = np.where(np.isfinite(standard_errors) & (standard_errors > 0.0), standard_errors, fallback)
        weights = 1.0 / np.maximum(standard_errors**2, 1e-12)
        weights = weights / float(np.mean(weights))
    else:
        weights = np.ones(len(intercepts), dtype=float)
    sqrt_weights = np.sqrt(weights)
    weighted_design = design * sqrt_weights[:, None]
    weighted_target = target * sqrt_weights
    coefficients, _, rank, _ = np.linalg.lstsq(weighted_design, weighted_target, rcond=None)
    fitted = design @ coefficients
    residuals = target - fitted
    weighted_residual_sum_squares = float(np.sum(weights * residuals**2))
    weighted_mean = float(np.average(target, weights=weights))
    weighted_total_sum_squares = float(np.sum(weights * (target - weighted_mean) ** 2))
    degrees_of_freedom = len(target) - int(rank)
    residual_variance = weighted_residual_sum_squares / degrees_of_freedom if degrees_of_freedom > 0 else float("nan")
    covariance = residual_variance * np.linalg.pinv(weighted_design.T @ weighted_design)
    standard_errors = np.sqrt(np.maximum(np.diag(covariance), 0.0))
    c_index = coefficient_names.index("c_log_q")
    c_log_q = float(coefficients[c_index])
    c_standard_error = float(standard_errors[c_index])
    lambda_paper = c_log_q / (2.0 * a_code)
    lambda_standard_error = c_standard_error / abs(2.0 * a_code)
    return {
        "controls": controls,
        "precision_weighted": precision_weighted,
        "num_intercepts": int(len(intercepts)),
        "rank": int(rank),
        "r_squared": 1.0 - weighted_residual_sum_squares / weighted_total_sum_squares if weighted_total_sum_squares > 0.0 else float("nan"),
        "residual_std": math.sqrt(residual_variance) if residual_variance >= 0.0 else float("nan"),
        "coefficients": {name: float(value) for name, value in zip(coefficient_names, coefficients)},
        "standard_errors": {name: float(value) for name, value in zip(coefficient_names, standard_errors)},
        "c_log_q": c_log_q,
        "c_log_q_standard_error": c_standard_error,
        "lambda_paper": lambda_paper,
        "lambda_paper_standard_error_conditional_on_fixed_a": lambda_standard_error,
        "lambda_paper_normal_95_interval_conditional_on_fixed_a": [lambda_paper - 1.96 * lambda_standard_error, lambda_paper + 1.96 * lambda_standard_error],
        "fitted_intercepts": fitted.tolist(),
        "residuals": residuals.tolist(),
    }


def add_adjusted_intercepts(intercepts: pd.DataFrame, regression: dict[str, object]) -> pd.DataFrame:
    output = intercepts.copy()
    coefficients = regression["coefficients"]
    resolution_reference = int(output["resolution"].min())
    adjusted = output["combined_intercept"].to_numpy(dtype=float) - float(coefficients["intercept"])
    for index, row in output.iterrows():
        resolution_effect = 0.0 if int(row["resolution"]) == resolution_reference else float(coefficients.get(f"resolution[{int(row['resolution'])}]", 0.0))
        letter_reference = int(output["letter_spacing"].min())
        letter_effect = 0.0 if int(row["letter_spacing"]) == letter_reference else float(coefficients.get(f"letter_spacing[{int(row['letter_spacing'])}]", 0.0))
        adjusted[index] -= resolution_effect + letter_effect
    output["intercept_adjusted_for_resolution_and_letter_spacing"] = adjusted
    output["fitted_intercept_primary"] = regression["fitted_intercepts"]
    output["residual_primary"] = regression["residuals"]
    return output


def save_plot(intercepts: pd.DataFrame, regression: dict[str, object], path: Path, dpi: int) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = {512: "tab:blue", 640: "tab:orange", 1024: "tab:green", 1280: "tab:red"}
    for resolution, group in intercepts.groupby("resolution", sort=True):
        ax.scatter(group["log_Q"], group["intercept_adjusted_for_resolution_and_letter_spacing"], color=colors.get(int(resolution)), label=str(resolution), s=22, alpha=0.75)
    x_values = np.linspace(float(intercepts["log_Q"].min()), float(intercepts["log_Q"].max()), 100)
    ax.plot(x_values, float(regression["c_log_q"]) * x_values, color="black", linewidth=2, label="fitted c log Q")
    ax.set_xlabel("log Q")
    ax.set_ylabel("Intercept adjusted for resolution and letter spacing")
    ax.legend(title="Resolution")
    fig.tight_layout()
    fig.savefig(path, dpi=dpi)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Fit lambda using only intercepts while fixing the original scaling-law slopes and Beta components.")
    parser.add_argument("--csv_path", type=Path, default=DEFAULT_CSV_PATH, help="Input feature/evaluation CSV.")
    parser.add_argument("--original_results_dir", type=Path, default=DEFAULT_ORIGINAL_RESULTS_DIR, help="Original shared-exceptw fit results.")
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Output directory.")
    parser.add_argument("--resolutions", type=str, default="512,640,1024,1280", help="Comma-separated resolutions.")
    parser.add_argument("--eps", type=float, default=1e-6, help="Numerical clipping epsilon.")
    parser.add_argument("--max_rows", type=int, default=0, help="Optional row limit for smoke testing.")
    parser.add_argument("--seed", type=int, default=42, help="Sampling seed for max_rows.")
    parser.add_argument("--plot_dpi", type=int, default=200, help="Plot DPI.")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")
    logger = logging.getLogger("fit_lambda_from_fixed_original_intercepts")
    resolutions = parse_resolutions(args.resolutions)
    original = load_original_parameters(args.original_results_dir, resolutions)
    logger.info("Fixed original parameters: a_code=%.9f, alpha_g=%.9f", float(original["a_code"]), float(original["alpha_g"]))
    df = load_data(args.csv_path, resolutions, args.eps, args.max_rows, args.seed).reset_index(drop=True)
    if args.max_rows == 0 and len(df.groupby(["resolution", "font_size", "line_spacing", "letter_spacing"])) != 144:
        raise ValueError("Expected 144 resolution-layout groups from 36 layouts and 4 resolutions.")
    fixed_offset = float(original["a_code"]) * df["log_V_code"].to_numpy(dtype=float) + float(original["alpha_g"]) * df["log_G"].to_numpy(dtype=float)
    beta_shapes = original["beta_shapes"]
    shapes = (float(beta_shapes["beta0_alpha"]), float(beta_shapes["beta0_beta"]), float(beta_shapes["beta1_alpha"]), float(beta_shapes["beta1_beta"]))
    intercepts = fit_all_intercepts(df, fixed_offset, shapes, original["original_resolution_biases"], logger)

    regressions = {
        "log_q_only_equal_weight": regress_intercepts(intercepts, [], float(original["a_code"]), False),
        "primary_log_q_plus_letter_spacing_equal_weight": regress_intercepts(intercepts, ["letter_spacing"], float(original["a_code"]), False),
        "log_q_plus_letter_spacing_and_font_size_equal_weight": regress_intercepts(intercepts, ["letter_spacing", "font_size"], float(original["a_code"]), False),
        "log_q_plus_letter_spacing_precision_weighted": regress_intercepts(intercepts, ["letter_spacing"], float(original["a_code"]), True),
    }
    primary = regressions["primary_log_q_plus_letter_spacing_equal_weight"]
    intercepts = add_adjusted_intercepts(intercepts, primary)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    intercepts.to_csv(args.output_dir / "layout_resolution_intercepts.csv", index=False)
    save_plot(intercepts, primary, args.output_dir / "adjusted_intercept_vs_log_q.png", args.plot_dpi)

    summary = {
        "method": {
            "fixed_parameters": "a_code, alpha_g, and both Beta components are loaded from the original shared_exceptw model and never re-estimated.",
            "intercepts": "Fit one combined intercept b_(resolution,layout) for each of 36 layouts x 4 resolutions.",
            "decomposition": "b_(resolution,layout)=b_resolution+c_log_q*log(Q_layout)+letter-spacing controls+residual.",
            "lambda_conversion": "lambda_paper=c_log_q/(2*a_code), because the original code uses V_code while the paper V is proportional to sqrt(V_code).",
        },
        "data": {"rows": int(len(df)), "num_layout_resolution_intercepts": int(len(intercepts)), "num_layouts": 36, "resolutions": resolutions, "excluded_layouts_in_intercept_fit": []},
        "original_fixed_parameters": original,
        "intercept_optimizer_successes": int(intercepts["optimizer_success"].sum()),
        "regressions": regressions,
    }
    with (args.output_dir / "fit_summary.json").open("w", encoding="utf-8") as file:
        json.dump(summary, file, ensure_ascii=False, indent=2)
    logger.info("Primary fixed-original-parameter lambda_paper=%.6f, R^2=%.6f", float(primary["lambda_paper"]), float(primary["r_squared"]))
    logger.info("Saved results to %s", args.output_dir)


if __name__ == "__main__":
    main()
