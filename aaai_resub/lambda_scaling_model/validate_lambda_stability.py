"""Validate lambda stability with configuration-level resampling and holdouts."""

import argparse
import json
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from fit_lambda_scaling_law import FitResult, fit_em, load_data, mixture_outputs, parse_resolutions


DEFAULT_CSV_PATH = Path("data_analysis/data4analysis.csv")
DEFAULT_OUTPUT_DIR = Path("aaai_resub/lambda_scaling_model/outputs/validation")


def make_arrays(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[int]]:
    resolutions = sorted(df["resolution"].unique().tolist())
    resolution_to_index = {resolution: index for index, resolution in enumerate(resolutions)}
    features = df[["log_V_code", "log_G", "log_Q"]].to_numpy(dtype=float)
    group_idx = df["resolution"].map(resolution_to_index).to_numpy(dtype=int)
    y = df["edit_distance"].to_numpy(dtype=float)
    return features, group_idx, y, resolutions


def lambda_values(result: FitResult) -> tuple[float, float]:
    if abs(result.a_code) <= 1e-12:
        return float("nan"), float("nan")
    return result.c_log_q / result.a_code, result.c_log_q / (2.0 * result.a_code)


def fit_with_weights(features: np.ndarray, group_idx: np.ndarray, y: np.ndarray, weights: np.ndarray, num_groups: int, args: argparse.Namespace, fit_logger: logging.Logger) -> FitResult:
    indices = np.flatnonzero(weights > 0.0)
    if indices.size == 0:
        raise ValueError("No positively weighted rows are available for fitting.")
    return fit_em(features[indices], group_idx[indices], y[indices], num_groups, args.max_iter, args.tol, args.init_threshold, args.init_soft, args.reg_strength, fit_logger, weights[indices])


def result_parameters(result: FitResult) -> np.ndarray:
    return np.concatenate([result.biases, [result.a_code, result.alpha_g, result.c_log_q]])


def summarize_fit(name: str, result: FitResult) -> dict[str, float | int | bool | str]:
    lambda_code, lambda_paper = lambda_values(result)
    return {
        "name": name,
        "converged": result.converged,
        "iterations": len(result.log_likelihood_history),
        "a_code": result.a_code,
        "alpha_g": result.alpha_g,
        "c_log_q": result.c_log_q,
        "lambda_code": lambda_code,
        "lambda_paper": lambda_paper,
    }


def heldout_metrics(result: FitResult, features: np.ndarray, group_idx: np.ndarray, y: np.ndarray, raw_y: np.ndarray, heldout: np.ndarray, num_groups: int, training_weights: np.ndarray) -> dict[str, float | int]:
    indices = np.flatnonzero(heldout)
    if indices.size == 0:
        return {"heldout_rows": 0}
    shapes = (result.beta0_alpha, result.beta0_beta, result.beta1_alpha, result.beta1_beta)
    pi, _, log_likelihood = mixture_outputs(features[indices], group_idx[indices], y[indices], result_parameters(result), num_groups, shapes)
    mean0 = result.beta0_alpha / (result.beta0_alpha + result.beta0_beta)
    mean1 = result.beta1_alpha / (result.beta1_alpha + result.beta1_beta)
    expected = (1.0 - pi) * mean0 + pi * mean1
    train_mean = float(np.average(raw_y, weights=training_weights))
    return {
        "heldout_rows": int(indices.size),
        "heldout_mean_edit_distance": float(np.mean(raw_y[indices])),
        "predicted_mean_edit_distance": float(np.mean(expected)),
        "heldout_mae": float(np.mean(np.abs(raw_y[indices] - expected))),
        "naive_train_mean_mae": float(np.mean(np.abs(raw_y[indices] - train_mean))),
        "heldout_mean_log_likelihood": float(log_likelihood / indices.size),
    }


def equal_group_weights(group_ids: np.ndarray, eligible: np.ndarray) -> np.ndarray:
    weights = np.zeros(group_ids.size, dtype=float)
    valid_groups = np.unique(group_ids[eligible])
    target_total = float(np.sum(eligible)) / len(valid_groups)
    for group in valid_groups:
        mask = eligible & (group_ids == group)
        weights[mask] = target_total / float(np.sum(mask))
    return weights


def run_bootstrap(df: pd.DataFrame, features: np.ndarray, group_idx: np.ndarray, y: np.ndarray, eligible: np.ndarray, vertical_ids: np.ndarray, num_groups: int, args: argparse.Namespace, fit_logger: logging.Logger, logger: logging.Logger) -> pd.DataFrame:
    rng = np.random.default_rng(args.seed)
    clusters = np.unique(vertical_ids)
    rows: list[dict[str, float | int | bool]] = []
    for replicate in range(args.bootstrap_replicates):
        sampled = rng.choice(clusters, size=len(clusters), replace=True)
        counts = np.bincount(sampled, minlength=int(clusters.max()) + 1)
        weights = counts[vertical_ids].astype(float) * eligible.astype(float)
        try:
            result = fit_with_weights(features, group_idx, y, weights, num_groups, args, fit_logger)
            lambda_code, lambda_paper = lambda_values(result)
            rows.append({"replicate": replicate, "converged": result.converged, "iterations": len(result.log_likelihood_history), "lambda_code": lambda_code, "lambda_paper": lambda_paper, "a_code": result.a_code, "c_log_q": result.c_log_q, "unique_clusters": int(np.count_nonzero(counts))})
        except (ValueError, FloatingPointError) as error:
            logger.warning("Bootstrap replicate %d failed: %s", replicate, error)
            rows.append({"replicate": replicate, "converged": False, "iterations": 0, "lambda_code": float("nan"), "lambda_paper": float("nan"), "a_code": float("nan"), "c_log_q": float("nan"), "unique_clusters": int(np.count_nonzero(counts))})
        if (replicate + 1) % 10 == 0:
            logger.info("Completed %d/%d cluster bootstrap fits", replicate + 1, args.bootstrap_replicates)
    return pd.DataFrame(rows)


def run_holdouts(df: pd.DataFrame, features: np.ndarray, group_idx: np.ndarray, y: np.ndarray, raw_y: np.ndarray, eligible: np.ndarray, holdout_columns: list[str], num_groups: int, args: argparse.Namespace, fit_logger: logging.Logger) -> pd.DataFrame:
    rows: list[dict[str, float | int | bool | str]] = []
    unique_groups = df[holdout_columns].drop_duplicates().sort_values(holdout_columns)
    for values in unique_groups.itertuples(index=False, name=None):
        heldout = np.ones(len(df), dtype=bool)
        labels: list[str] = []
        for column, value in zip(holdout_columns, values):
            heldout &= df[column].to_numpy() == value
            labels.append(f"{column}={value}")
        weights = (eligible & ~heldout).astype(float)
        result = fit_with_weights(features, group_idx, y, weights, num_groups, args, fit_logger)
        row = summarize_fit(",".join(labels), result)
        row.update(heldout_metrics(result, features, group_idx, y, raw_y, heldout, num_groups, weights))
        rows.append(row)
    return pd.DataFrame(rows)


def finite_summary(values: pd.Series) -> dict[str, float | int]:
    array = values.to_numpy(dtype=float)
    array = array[np.isfinite(array)]
    if array.size == 0:
        return {"count": 0}
    return {
        "count": int(array.size),
        "mean": float(np.mean(array)),
        "std": float(np.std(array, ddof=1)) if array.size > 1 else 0.0,
        "q025": float(np.quantile(array, 0.025)),
        "median": float(np.quantile(array, 0.5)),
        "q975": float(np.quantile(array, 0.975)),
        "fraction_below_0": float(np.mean(array < 0.0)),
        "fraction_above_0_5": float(np.mean(array > 0.5)),
        "min": float(np.min(array)),
        "max": float(np.max(array)),
    }


def save_plots(bootstrap: pd.DataFrame, holdout_tables: dict[str, pd.DataFrame], output_dir: Path, dpi: int) -> None:
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    values = bootstrap["lambda_paper"].replace([np.inf, -np.inf], np.nan).dropna()
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(values, bins=20, color="tab:blue", alpha=0.8)
    ax.axvline(0.0, color="black", linestyle="--", linewidth=1)
    ax.axvline(0.5, color="red", linestyle="--", linewidth=1)
    ax.set_xlabel("Paper-space lambda")
    ax.set_ylabel("Cluster bootstrap count")
    fig.tight_layout()
    fig.savefig(plots_dir / "bootstrap_lambda_paper.png", dpi=dpi)
    plt.close(fig)

    fig, axes = plt.subplots(len(holdout_tables), 1, figsize=(9, 3 * len(holdout_tables)), squeeze=False)
    for ax, (name, table) in zip(axes.flat, holdout_tables.items()):
        ax.plot(np.arange(len(table)), table["lambda_paper"], marker="o", linestyle="none")
        ax.axhline(0.0, color="black", linestyle="--", linewidth=1)
        ax.axhline(0.5, color="red", linestyle="--", linewidth=1)
        ax.set_xticks(np.arange(len(table)), table["name"], rotation=45, ha="right", fontsize=7)
        ax.set_ylabel("lambda_paper")
        ax.set_title(name)
    fig.tight_layout()
    fig.savefig(plots_dir / "leave_out_lambda_paper.png", dpi=dpi)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Configuration-level stability validation for lambda.")
    parser.add_argument("--csv_path", type=Path, default=DEFAULT_CSV_PATH, help="Input feature/evaluation CSV.")
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Validation output directory.")
    parser.add_argument("--resolutions", type=str, default="512,640,1024,1280", help="Comma-separated resolutions.")
    parser.add_argument("--bootstrap_replicates", type=int, default=100, help="Number of 12-cluster bootstrap fits.")
    parser.add_argument("--max_iter", type=int, default=200, help="Maximum EM iterations per fit.")
    parser.add_argument("--tol", type=float, default=1e-6, help="EM convergence tolerance.")
    parser.add_argument("--eps", type=float, default=1e-6, help="Numerical clipping epsilon.")
    parser.add_argument("--init_threshold", type=float, default=0.37, help="Edit-distance initialization threshold.")
    parser.add_argument("--init_soft", type=float, default=0.05, help="Initial soft assignment.")
    parser.add_argument("--reg_strength", type=float, default=0.0, help="L2 penalty on shared slopes.")
    parser.add_argument("--max_rows", type=int, default=0, help="Optional row limit for smoke testing only.")
    parser.add_argument("--seed", type=int, default=42, help="Bootstrap and sampling seed.")
    parser.add_argument("--plot_dpi", type=int, default=200, help="Plot DPI.")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")
    logger = logging.getLogger("validate_lambda_stability")
    fit_logger = logging.getLogger("validation_em_fit")
    fit_logger.setLevel(logging.WARNING)
    df = load_data(args.csv_path, parse_resolutions(args.resolutions), args.eps, args.max_rows, args.seed)
    features, group_idx, y, resolutions = make_arrays(df)
    raw_y = df["edit_distance_raw"].to_numpy(dtype=float)
    num_groups = len(resolutions)
    excluded = ((df["font_size"].astype(float) == 28.0) & (df["line_spacing"].astype(float) == 6.0) & (df["letter_spacing"].astype(float) == 0.0)).to_numpy()
    eligible = ~excluded

    vertical_keys = pd.MultiIndex.from_frame(df[["font_size", "line_spacing"]].astype(int))
    vertical_ids, vertical_levels = pd.factorize(vertical_keys, sort=True)
    layout_keys = pd.MultiIndex.from_frame(df[["font_size", "line_spacing", "letter_spacing"]].astype(int))
    layout_ids, layout_levels = pd.factorize(layout_keys, sort=True)
    logger.info("Validating %d rows, %d complete layouts, and %d vertical configurations", len(df), len(layout_levels), len(vertical_levels))

    weighting_rows: list[dict[str, float | int | bool | str]] = []
    weighting_schemes = {
        "sample_weighted": eligible.astype(float),
        "complete_layout_equal": equal_group_weights(layout_ids, eligible),
        "vertical_configuration_equal": equal_group_weights(vertical_ids, eligible),
    }
    for name, weights in weighting_schemes.items():
        result = fit_with_weights(features, group_idx, y, weights, num_groups, args, fit_logger)
        row = summarize_fit(name, result)
        row["effective_weight_sum"] = float(np.sum(weights))
        weighting_rows.append(row)
        logger.info("%s: lambda_paper=%.6f", name, float(row["lambda_paper"]))
    weighting = pd.DataFrame(weighting_rows)

    bootstrap = run_bootstrap(df, features, group_idx, y, eligible, vertical_ids, num_groups, args, fit_logger, logger)
    leave_vertical = run_holdouts(df, features, group_idx, y, raw_y, eligible, ["font_size", "line_spacing"], num_groups, args, fit_logger)
    leave_font = run_holdouts(df, features, group_idx, y, raw_y, eligible, ["font_size"], num_groups, args, fit_logger)
    leave_line = run_holdouts(df, features, group_idx, y, raw_y, eligible, ["line_spacing"], num_groups, args, fit_logger)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    weighting.to_csv(args.output_dir / "weighting_comparison.csv", index=False)
    bootstrap.to_csv(args.output_dir / "cluster_bootstrap.csv", index=False)
    leave_vertical.to_csv(args.output_dir / "leave_one_vertical_configuration_out.csv", index=False)
    leave_font.to_csv(args.output_dir / "leave_one_font_size_out.csv", index=False)
    leave_line.to_csv(args.output_dir / "leave_one_line_spacing_out.csv", index=False)
    holdout_tables = {"Leave one vertical configuration out": leave_vertical, "Leave one font size out": leave_font, "Leave one line spacing out": leave_line}
    save_plots(bootstrap, holdout_tables, args.output_dir, args.plot_dpi)

    summary = {
        "method": {
            "independent_cluster": "(font_size, line_spacing); all letter spacings, resolutions, and samples are resampled together",
            "num_vertical_configurations": int(len(vertical_levels)),
            "num_unique_q": int(df["Q"].nunique()),
            "bootstrap_replicates": int(args.bootstrap_replicates),
            "excluded_training_layout": "font_size=28, line_spacing=6, letter_spacing=0",
            "lambda_conversion": "lambda_paper = c_log_q / (2*a_code) because V_paper is proportional to sqrt(V_code)",
        },
        "bootstrap_lambda_paper": finite_summary(bootstrap["lambda_paper"]),
        "leave_one_vertical_lambda_paper": finite_summary(leave_vertical["lambda_paper"]),
        "leave_one_font_lambda_paper": finite_summary(leave_font["lambda_paper"]),
        "leave_one_line_spacing_lambda_paper": finite_summary(leave_line["lambda_paper"]),
        "weighting_comparison": weighting.to_dict(orient="records"),
    }
    with (args.output_dir / "validation_summary.json").open("w", encoding="utf-8") as file:
        json.dump(summary, file, ensure_ascii=False, indent=2)
    logger.info("Saved validation results to %s", args.output_dir)


if __name__ == "__main__":
    main()
