import argparse
import csv
import json
import logging
import math
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize
from scipy.special import expit, gammaln


LOGGER = logging.getLogger(__name__)
FONT_SIZES = [20, 28, 36]
LINE_SPACINGS = [0, 6, 24, 42]
LETTER_SPACINGS = [-1, 0, 7]
RESOLUTIONS = [512, 640, 1024, 1280]
DEFAULT_HOLDOUT_LAYOUT = "20_0_7"
DEFAULT_EPS = 1e-8
DEFAULT_BIN_SIZE = 1000
DEFAULT_TEXT_LEN_SCALE = 1000.0
DEFAULT_MAX_ITER = 200
DEFAULT_TOL = 1e-6
DEFAULT_INIT_THRESHOLD = 0.5
DEFAULT_INIT_SOFT = 0.05
MIN_ALPHA_BETA = 1e-3
MAX_ALPHA_BETA = 1e4


@dataclass
class BetaBinomialEMResult:
    b_task: float
    eta_task: float
    delta_task: float
    alpha0: float
    beta0: float
    alpha1: float
    beta1: float
    responsibilities: np.ndarray
    epsilon_task: np.ndarray
    task_z: np.ndarray
    log_likelihood_history: list[float]
    converged: bool


def iter_layouts() -> list[str]:
    return [f"{font_size}_{line_spacing}_{letter_spacing}" for font_size in FONT_SIZES for line_spacing in LINE_SPACINGS for letter_spacing in LETTER_SPACINGS]


def parse_resolutions(resolutions_arg: str) -> list[int]:
    return [int(part.strip()) for part in resolutions_arg.split(",") if part.strip()]


def log_sigmoid(z: np.ndarray) -> np.ndarray:
    return -np.logaddexp(0.0, -z)


def log1m_sigmoid(z: np.ndarray) -> np.ndarray:
    return -np.logaddexp(0.0, z)


def beta_binomial_logpmf(successes: np.ndarray, totals: np.ndarray, alpha: float, beta: float) -> np.ndarray:
    successes = np.asarray(successes, dtype=float)
    totals = np.asarray(totals, dtype=float)
    failures = totals - successes
    log_choose = gammaln(totals + 1.0) - gammaln(successes + 1.0) - gammaln(failures + 1.0)
    log_beta_num = gammaln(successes + alpha) + gammaln(failures + beta) - gammaln(totals + alpha + beta)
    log_beta_den = gammaln(alpha) + gammaln(beta) - gammaln(alpha + beta)
    return log_choose + log_beta_num - log_beta_den


def weighted_beta_moments(values: np.ndarray, weights: np.ndarray) -> tuple[float, float]:
    weights = np.asarray(weights, dtype=float)
    values = np.asarray(values, dtype=float)
    weights = np.where(np.isfinite(weights) & (weights > 0.0), weights, 0.0)
    wsum = float(np.sum(weights))
    if wsum <= 0.0:
        return 1.0, 1.0
    normalized = weights / wsum
    mean = float(np.sum(normalized * values))
    var = float(np.sum(normalized * (values - mean) ** 2))
    if var <= 1e-10 or mean <= 0.0 or mean >= 1.0:
        strength = 20.0
        mean = min(max(mean, 1e-3), 1.0 - 1e-3)
        return mean * strength, (1.0 - mean) * strength
    scale = mean * (1.0 - mean) / var - 1.0
    if not np.isfinite(scale) or scale <= 0.0:
        return 1.0, 1.0
    alpha = mean * scale
    beta = (1.0 - mean) * scale
    if alpha <= 0.0 or beta <= 0.0:
        return 1.0, 1.0
    return float(alpha), float(beta)


def fit_weighted_beta_binomial(successes: np.ndarray, totals: np.ndarray, weights: np.ndarray, init_params: tuple[float, float] | None, max_iter: int) -> tuple[float, float]:
    successes = np.asarray(successes, dtype=float)
    totals = np.asarray(totals, dtype=float)
    weights = np.asarray(weights, dtype=float)
    weights = np.where(np.isfinite(weights) & (weights > 0.0), weights, 0.0)
    if float(np.sum(weights)) <= 0.0:
        return (1.0, 1.0) if init_params is None else init_params
    if init_params is None:
        smoothed_acc = (successes + 0.5) / (totals + 1.0)
        init_params = weighted_beta_moments(smoothed_acc, weights)
    init_alpha = float(np.clip(init_params[0], MIN_ALPHA_BETA, MAX_ALPHA_BETA))
    init_beta = float(np.clip(init_params[1], MIN_ALPHA_BETA, MAX_ALPHA_BETA))

    def nll(params: np.ndarray) -> float:
        alpha = float(np.exp(params[0]))
        beta = float(np.exp(params[1]))
        log_p = beta_binomial_logpmf(successes, totals, alpha, beta)
        return -float(np.sum(weights * log_p))

    result = minimize(nll, np.log([init_alpha, init_beta]), method="L-BFGS-B", bounds=[(math.log(MIN_ALPHA_BETA), math.log(MAX_ALPHA_BETA))] * 2, options={"maxiter": int(max_iter)})
    alpha = float(np.exp(result.x[0]))
    beta = float(np.exp(result.x[1]))
    if not result.success or not np.isfinite(alpha) or not np.isfinite(beta):
        return init_alpha, init_beta
    return alpha, beta


def fit_logistic_gating(offset: np.ndarray, log_text_len: np.ndarray, log_complexity: np.ndarray, responsibilities: np.ndarray, init_params: np.ndarray, reg_strength: float, max_iter: int) -> np.ndarray:
    offset = np.asarray(offset, dtype=float)
    log_text_len = np.asarray(log_text_len, dtype=float)
    log_complexity = np.asarray(log_complexity, dtype=float)
    responsibilities = np.asarray(responsibilities, dtype=float)

    def nll(params: np.ndarray) -> float:
        b_task, eta_task, delta_task = [float(value) for value in params]
        z = offset + b_task + eta_task * log_text_len + delta_task * log_complexity
        ll = np.sum(responsibilities * log_sigmoid(z) + (1.0 - responsibilities) * log1m_sigmoid(z))
        penalty = 0.5 * float(reg_strength) * float(np.sum(params**2))
        return -float(ll) + penalty

    def grad(params: np.ndarray) -> np.ndarray:
        b_task, eta_task, delta_task = [float(value) for value in params]
        z = offset + b_task + eta_task * log_text_len + delta_task * log_complexity
        diff = expit(z) - responsibilities
        grad_b = np.sum(diff) + reg_strength * b_task
        grad_eta = np.sum(diff * log_text_len) + reg_strength * eta_task
        grad_delta = np.sum(diff * log_complexity) + reg_strength * delta_task
        return np.array([grad_b, grad_eta, grad_delta], dtype=float)

    result = minimize(nll, init_params, method="L-BFGS-B", jac=grad, options={"maxiter": int(max_iter)})
    if not result.success or not np.all(np.isfinite(result.x)):
        return init_params.astype(float)
    return result.x.astype(float)


def load_rows(csv_path: Path, resolutions: list[int], text_len_scale: float) -> list[dict]:
    rows = []
    with csv_path.open("r", newline="", encoding="utf-8") as file_handle:
        reader = csv.DictReader(file_handle)
        required = {"image_id", "layout", "resolution", "text_len", "qwen_logV", "qwen_logG", "log_complexity", "correct"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Scaling CSV is missing required fields: {sorted(missing)}")
        for row in reader:
            resolution = int(row["resolution"])
            if resolution not in resolutions:
                continue
            text_len = int(float(row["text_len"]))
            if text_len <= 0:
                continue
            row["resolution"] = resolution
            row["text_len"] = text_len
            row["qwen_logV"] = float(row["qwen_logV"])
            row["qwen_logG"] = float(row["qwen_logG"])
            row["log_complexity"] = float(row["log_complexity"])
            row["log_text_len"] = math.log(float(text_len) / float(text_len_scale))
            row["correct"] = int(float(row["correct"]))
            if "contains_all" in row:
                row["contains_all"] = float(row["contains_all"])
            rows.append(row)
    if not rows:
        raise ValueError(f"No scaling rows loaded from {csv_path}")
    LOGGER.info("Loaded %d rows from %s", len(rows), csv_path)
    return rows


def load_ocr_gating_params(ocr_fit_dir: Path, resolutions: list[int]) -> dict[int, dict[str, float]]:
    params = {}
    a_values = []
    alpha_values = []
    for resolution in resolutions:
        summary_path = ocr_fit_dir / str(resolution) / "em_fit_summary.json"
        with summary_path.open("r", encoding="utf-8") as file_handle:
            summary = json.load(file_handle)
        gating = summary["gating_params"]
        params[int(resolution)] = {"w0": float(gating["w0"]), "a": float(gating["a"]), "alpha": float(gating["alpha"])}
        a_values.append(float(gating["a"]))
        alpha_values.append(float(gating["alpha"]))
    if max(a_values) - min(a_values) > 1e-6:
        LOGGER.warning("OCR gating parameter a differs across resolution summaries: %s", a_values)
    if max(alpha_values) - min(alpha_values) > 1e-6:
        LOGGER.warning("OCR gating parameter alpha differs across resolution summaries: %s", alpha_values)
    return params


def rows_to_arrays(rows: list[dict], gating_params: dict[int, dict[str, float]]) -> dict[str, np.ndarray]:
    log_text_len = np.array([row["log_text_len"] for row in rows], dtype=float)
    log_complexity = np.array([row["log_complexity"] for row in rows], dtype=float)
    y = np.array([row["correct"] for row in rows], dtype=float)
    resolution = np.array([row["resolution"] for row in rows], dtype=int)
    ocr_z = np.zeros(len(rows), dtype=float)
    for index, row in enumerate(rows):
        params = gating_params[int(row["resolution"])]
        ocr_z[index] = params["w0"] + params["a"] * row["qwen_logV"] + params["alpha"] * row["qwen_logG"]
    ocr_epsilon = expit(ocr_z)
    return {"log_text_len": log_text_len, "log_complexity": log_complexity, "y": y, "resolution": resolution, "ocr_z": ocr_z, "ocr_epsilon": ocr_epsilon}


def build_group_rows(rows: list[dict], arrays: dict[str, np.ndarray], bin_size: int, holdout_layout: str) -> list[dict]:
    grouped: dict[tuple[str, int, int], list[int]] = {}
    for index, row in enumerate(rows):
        bin_start = int(row["text_len"]) // int(bin_size) * int(bin_size)
        grouped.setdefault((row["layout"], int(row["resolution"]), bin_start), []).append(index)
    group_rows = []
    for (layout, resolution, bin_start), indices in sorted(grouped.items()):
        y = arrays["y"][indices]
        text_lens = [int(rows[index]["text_len"]) for index in indices]
        split = "test" if layout == holdout_layout else "train"
        group_rows.append(
            {
                "layout": layout,
                "resolution": resolution,
                "split": split,
                "bin_start": bin_start,
                "bin_end": bin_start + int(bin_size),
                "bin_center": bin_start + int(bin_size) / 2,
                "n": len(indices),
                "s": int(np.sum(y)),
                "mean_text_len": float(np.mean(text_lens)),
                "mean_ocr_z": float(np.mean(arrays["ocr_z"][indices])),
                "mean_ocr_epsilon": float(np.mean(arrays["ocr_epsilon"][indices])),
                "mean_log_text_len": float(np.mean(arrays["log_text_len"][indices])),
                "mean_log_complexity": float(np.mean(arrays["log_complexity"][indices])),
                "observed_acc": float(np.mean(y)),
            }
        )
    return group_rows


def group_rows_to_arrays(group_rows: list[dict]) -> dict[str, np.ndarray]:
    return {
        "n": np.array([row["n"] for row in group_rows], dtype=float),
        "s": np.array([row["s"] for row in group_rows], dtype=float),
        "ocr_z": np.array([row["mean_ocr_z"] for row in group_rows], dtype=float),
        "log_text_len": np.array([row["mean_log_text_len"] for row in group_rows], dtype=float),
        "log_complexity": np.array([row["mean_log_complexity"] for row in group_rows], dtype=float),
        "observed_acc": np.array([row["observed_acc"] for row in group_rows], dtype=float),
    }


def initialize_responsibilities(observed_acc: np.ndarray, threshold: float, soft: float) -> np.ndarray:
    responsibilities = np.where(observed_acc <= float(threshold), 1.0 - float(soft), float(soft))
    return responsibilities.astype(float)


def compute_group_log_likelihood(group_arrays: dict[str, np.ndarray], b_task: float, eta_task: float, delta_task: float, alpha0: float, beta0: float, alpha1: float, beta1: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    task_z = group_arrays["ocr_z"] + float(b_task) + float(eta_task) * group_arrays["log_text_len"] + float(delta_task) * group_arrays["log_complexity"]
    log_epsilon = log_sigmoid(task_z)
    log1m_epsilon = log1m_sigmoid(task_z)
    log_p0 = beta_binomial_logpmf(group_arrays["s"], group_arrays["n"], alpha0, beta0)
    log_p1 = beta_binomial_logpmf(group_arrays["s"], group_arrays["n"], alpha1, beta1)
    log_mix = np.logaddexp(log1m_epsilon + log_p0, log_epsilon + log_p1)
    responsibilities = np.exp(log_epsilon + log_p1 - log_mix)
    epsilon_task = expit(task_z)
    return task_z, epsilon_task, responsibilities, float(np.sum(log_mix))


def em_fit_beta_binomial(group_arrays: dict[str, np.ndarray], max_iter: int, tol: float, init_threshold: float, init_soft: float, reg_strength: float, logger: logging.Logger) -> BetaBinomialEMResult:
    observed_acc = group_arrays["observed_acc"]
    init_thresholds = []
    for value in [init_threshold, float(np.mean(observed_acc)), 0.25, 0.5, 0.75]:
        if 0.0 <= value <= 1.0 and not any(abs(value - existing) < 1e-9 for existing in init_thresholds):
            init_thresholds.append(value)
    best_result = None
    best_ll = -float("inf")
    for threshold in init_thresholds:
        result = em_fit_beta_binomial_once(group_arrays, max_iter, tol, threshold, init_soft, reg_strength, logger)
        final_ll = result.log_likelihood_history[-1] if result.log_likelihood_history else -float("inf")
        logger.info("Init threshold %.4f final train log-likelihood %.6f", threshold, final_ll)
        if final_ll > best_ll:
            best_result = result
            best_ll = final_ll
    if best_result is None:
        raise RuntimeError("Beta-Binomial EM failed to produce a result.")
    return best_result


def em_fit_beta_binomial_once(group_arrays: dict[str, np.ndarray], max_iter: int, tol: float, init_threshold: float, init_soft: float, reg_strength: float, logger: logging.Logger) -> BetaBinomialEMResult:
    responsibilities = initialize_responsibilities(group_arrays["observed_acc"], init_threshold, init_soft)
    alpha1, beta1 = fit_weighted_beta_binomial(group_arrays["s"], group_arrays["n"], responsibilities, None, max_iter)
    alpha0, beta0 = fit_weighted_beta_binomial(group_arrays["s"], group_arrays["n"], 1.0 - responsibilities, None, max_iter)
    gate_params = fit_logistic_gating(group_arrays["ocr_z"], group_arrays["log_text_len"], group_arrays["log_complexity"], responsibilities, np.zeros(3, dtype=float), reg_strength, max_iter)
    history = []
    converged = False
    for iteration in range(int(max_iter)):
        task_z, epsilon_task, responsibilities, ll = compute_group_log_likelihood(group_arrays, gate_params[0], gate_params[1], gate_params[2], alpha0, beta0, alpha1, beta1)
        history.append(ll)
        if iteration > 0 and abs(history[-1] - history[-2]) < float(tol):
            converged = True
            break
        gate_params = fit_logistic_gating(group_arrays["ocr_z"], group_arrays["log_text_len"], group_arrays["log_complexity"], responsibilities, gate_params, reg_strength, max_iter)
        alpha1, beta1 = fit_weighted_beta_binomial(group_arrays["s"], group_arrays["n"], responsibilities, (alpha1, beta1), max_iter)
        alpha0, beta0 = fit_weighted_beta_binomial(group_arrays["s"], group_arrays["n"], 1.0 - responsibilities, (alpha0, beta0), max_iter)
        if iteration % 10 == 0:
            logger.info("EM iter %d log-likelihood %.6f", iteration, ll)
    task_z, epsilon_task, responsibilities, _ = compute_group_log_likelihood(group_arrays, gate_params[0], gate_params[1], gate_params[2], alpha0, beta0, alpha1, beta1)
    return BetaBinomialEMResult(
        b_task=float(gate_params[0]),
        eta_task=float(gate_params[1]),
        delta_task=float(gate_params[2]),
        alpha0=float(alpha0),
        beta0=float(beta0),
        alpha1=float(alpha1),
        beta1=float(beta1),
        responsibilities=responsibilities.astype(float),
        epsilon_task=epsilon_task.astype(float),
        task_z=task_z.astype(float),
        log_likelihood_history=history,
        converged=converged,
    )


def predict_sample_components(rows: list[dict], arrays: dict[str, np.ndarray], result: BetaBinomialEMResult) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    task_z = arrays["ocr_z"] + result.b_task + result.eta_task * arrays["log_text_len"] + result.delta_task * arrays["log_complexity"]
    epsilon_task = expit(task_z)
    mean0 = result.alpha0 / (result.alpha0 + result.beta0)
    mean1 = result.alpha1 / (result.alpha1 + result.beta1)
    predicted_success = (1.0 - epsilon_task) * mean0 + epsilon_task * mean1
    return task_z, epsilon_task, predicted_success


def add_sample_predictions(rows: list[dict], arrays: dict[str, np.ndarray], result: BetaBinomialEMResult, holdout_layout: str) -> list[dict]:
    task_z, epsilon_task, predicted_success = predict_sample_components(rows, arrays, result)
    output_rows = []
    for index, row in enumerate(rows):
        out = dict(row)
        out["ocr_z"] = float(arrays["ocr_z"][index])
        out["ocr_epsilon"] = float(arrays["ocr_epsilon"][index])
        out["task_z"] = float(task_z[index])
        out["epsilon_task"] = float(epsilon_task[index])
        out["predicted_success"] = float(predicted_success[index])
        out["split"] = "test" if row["layout"] == holdout_layout else "train"
        output_rows.append(out)
    return output_rows


def add_group_predictions(group_rows: list[dict], result: BetaBinomialEMResult) -> list[dict]:
    mean0 = result.alpha0 / (result.alpha0 + result.beta0)
    mean1 = result.alpha1 / (result.alpha1 + result.beta1)
    output_rows = []
    for index, row in enumerate(group_rows):
        out = dict(row)
        out["task_z"] = float(result.task_z[index])
        out["epsilon_task"] = float(result.epsilon_task[index])
        out["responsibility"] = float(result.responsibilities[index])
        out["predicted_acc"] = float((1.0 - result.epsilon_task[index]) * mean0 + result.epsilon_task[index] * mean1)
        output_rows.append(out)
    return output_rows


def write_csv(rows: list[dict], output_csv: Path) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with output_csv.open("w", newline="", encoding="utf-8") as file_handle:
        writer = csv.DictWriter(file_handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def plot_resolution(sample_rows: list[dict], group_rows: list[dict], task: str, resolution: int, output_file: Path, dpi: int) -> None:
    layouts = iter_layouts()
    fig, axes = plt.subplots(6, 6, figsize=(24, 18), sharex=True, sharey=True)
    axes_flat = axes.ravel()
    for ax, layout in zip(axes_flat, layouts):
        group = [row for row in sample_rows if row["layout"] == layout and int(row["resolution"]) == int(resolution)]
        bin_group = [row for row in group_rows if row["layout"] == layout and int(row["resolution"]) == int(resolution)]
        if not group:
            ax.set_title(f"{layout}\nmissing", fontsize=9)
            ax.set_xlim(0, 25000)
            ax.set_ylim(-0.05, 1.05)
            ax.grid(True, linestyle="--", linewidth=0.4, alpha=0.35)
            continue
        train_x = [int(row["text_len"]) for row in group if row["split"] == "train"]
        train_y = [float(row["correct"]) for row in group if row["split"] == "train"]
        test_x = [int(row["text_len"]) for row in group if row["split"] == "test"]
        test_y = [float(row["correct"]) for row in group if row["split"] == "test"]
        if train_x:
            ax.scatter(train_x, train_y, s=7, alpha=0.35, edgecolors="none", color="tab:blue")
        if test_x:
            ax.scatter(test_x, test_y, s=8, alpha=0.55, edgecolors="none", color="tab:orange")
        sorted_bin_group = sorted(bin_group, key=lambda item: int(item["bin_start"]))
        if sorted_bin_group:
            bin_x = [float(row["bin_center"]) for row in sorted_bin_group]
            ax.plot(bin_x, [float(row["observed_acc"]) for row in sorted_bin_group], color="tab:green", marker="o", markersize=2.5, linewidth=1.2)
            ax.plot(bin_x, [float(row["predicted_acc"]) for row in sorted_bin_group], color="tab:red", marker="o", markersize=2.5, linewidth=1.2)
        mean_acc = float(np.mean([float(row["correct"]) for row in group]))
        split_label = " test" if any(row["split"] == "test" for row in group) else ""
        ax.set_title(f"{layout}{split_label}  mean={mean_acc:.3f}", fontsize=9)
        ax.set_xlim(0, 25000)
        ax.set_ylim(-0.05, 1.05)
        ax.grid(True, linestyle="--", linewidth=0.4, alpha=0.35)
    for ax in axes[-1, :]:
        ax.set_xlabel("Characters")
    for ax in axes[:, 0]:
        ax.set_ylabel("Exact success")
    handles = [
        plt.Line2D([0], [0], color="tab:green", marker="o", markersize=4, linewidth=1.4, label="Observed bin acc"),
        plt.Line2D([0], [0], color="tab:red", marker="o", markersize=4, linewidth=1.4, label="Predicted bin acc"),
    ]
    fig.legend(handles=handles, loc="upper right", bbox_to_anchor=(0.995, 0.995), fontsize=10)
    fig.suptitle(f"Qwen {task} scheme3 Beta-Binomial EM fit, resolution={resolution}", fontsize=16)
    fig.tight_layout(rect=(0, 0, 1, 0.975))
    output_file.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_file, dpi=int(dpi))
    plt.close(fig)


def compute_nll_for_group_rows(group_rows: list[dict], result: BetaBinomialEMResult) -> float | None:
    if not group_rows:
        return None
    group_arrays = group_rows_to_arrays(group_rows)
    _, _, _, ll = compute_group_log_likelihood(group_arrays, result.b_task, result.eta_task, result.delta_task, result.alpha0, result.beta0, result.alpha1, result.beta1)
    return -float(ll)


def summarize(rows: list[dict], sample_arrays: dict[str, np.ndarray], group_rows: list[dict], result: BetaBinomialEMResult, args: argparse.Namespace) -> dict:
    train_groups = [row for row in group_rows if row["split"] == "train"]
    test_groups = [row for row in group_rows if row["split"] == "test"]
    mean0 = result.alpha0 / (result.alpha0 + result.beta0)
    mean1 = result.alpha1 / (result.alpha1 + result.beta1)
    return {
        "scheme": "scheme3_betabinomial_em",
        "task": args.task,
        "csv_path": str(args.csv_path),
        "ocr_fit_dir": str(args.ocr_fit_dir),
        "holdout_layout": args.holdout_layout,
        "n_samples": len(rows),
        "n_groups": len(group_rows),
        "n_train_samples": sum(1 for row in rows if row["layout"] != args.holdout_layout),
        "n_test_samples": sum(1 for row in rows if row["layout"] == args.holdout_layout),
        "n_train_groups": len(train_groups),
        "n_test_groups": len(test_groups),
        "train_accuracy_mean": float(np.mean([row["correct"] for row in rows if row["layout"] != args.holdout_layout])),
        "test_accuracy_mean": float(np.mean([row["correct"] for row in rows if row["layout"] == args.holdout_layout])),
        "gating_params": {"b_task": float(result.b_task), "eta_task": float(result.eta_task), "delta_task": float(result.delta_task)},
        "beta_binomial_params": {
            "state0": {"alpha": float(result.alpha0), "beta": float(result.beta0), "mean": float(mean0)},
            "state1": {"alpha": float(result.alpha1), "beta": float(result.beta1), "mean": float(mean1)},
        },
        "train_nll": compute_nll_for_group_rows(train_groups, result),
        "test_nll": compute_nll_for_group_rows(test_groups, result),
        "all_nll": compute_nll_for_group_rows(group_rows, result),
        "mean_ocr_epsilon": float(np.mean(sample_arrays["ocr_epsilon"])),
        "mean_group_epsilon_task": float(np.mean(result.epsilon_task)),
        "mean_group_responsibility": float(np.mean(result.responsibilities)),
        "text_len_scale": float(args.text_len_scale),
        "log_text_len_definition": "log(text_len / text_len_scale)",
        "bin_size": int(args.bin_size),
        "max_iter": int(args.max_iter),
        "tol": float(args.tol),
        "init_threshold": float(args.init_threshold),
        "init_soft": float(args.init_soft),
        "reg_strength": float(args.reg_strength),
        "log_likelihood_history": result.log_likelihood_history,
        "converged": bool(result.converged),
    }


def run_fit(args: argparse.Namespace) -> None:
    resolutions = parse_resolutions(args.resolutions)
    rows = load_rows(args.csv_path, resolutions, args.text_len_scale)
    gating_params = load_ocr_gating_params(args.ocr_fit_dir, resolutions)
    sample_arrays = rows_to_arrays(rows, gating_params)
    group_rows = build_group_rows(rows, sample_arrays, args.bin_size, args.holdout_layout)
    train_group_rows = [row for row in group_rows if row["split"] == "train"]
    if not train_group_rows:
        raise ValueError("No training groups remain after applying holdout_layout.")
    train_group_arrays = group_rows_to_arrays(train_group_rows)
    result_train = em_fit_beta_binomial(train_group_arrays, args.max_iter, args.tol, args.init_threshold, args.init_soft, args.reg_strength, LOGGER)
    all_group_arrays = group_rows_to_arrays(group_rows)
    task_z, epsilon_task, responsibilities, _ = compute_group_log_likelihood(all_group_arrays, result_train.b_task, result_train.eta_task, result_train.delta_task, result_train.alpha0, result_train.beta0, result_train.alpha1, result_train.beta1)
    result = BetaBinomialEMResult(
        b_task=result_train.b_task,
        eta_task=result_train.eta_task,
        delta_task=result_train.delta_task,
        alpha0=result_train.alpha0,
        beta0=result_train.beta0,
        alpha1=result_train.alpha1,
        beta1=result_train.beta1,
        responsibilities=responsibilities,
        epsilon_task=epsilon_task,
        task_z=task_z,
        log_likelihood_history=result_train.log_likelihood_history,
        converged=result_train.converged,
    )
    sample_output_rows = add_sample_predictions(rows, sample_arrays, result, args.holdout_layout)
    group_output_rows = add_group_predictions(group_rows, result)
    output_dir = args.output_dir / args.task
    plots_dir = output_dir / "plots"
    output_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)
    write_csv(sample_output_rows, output_dir / "fit_points.csv")
    write_csv(group_output_rows, output_dir / "fit_points_binned.csv")
    summary = summarize(rows, sample_arrays, group_output_rows, result, args)
    with (output_dir / "fit_summary.json").open("w", encoding="utf-8") as file_handle:
        json.dump(summary, file_handle, ensure_ascii=False, indent=2)
    for resolution in resolutions:
        plot_resolution(sample_output_rows, group_output_rows, args.task, resolution, plots_dir / f"qwen_{args.task}_scheme3_betabinomial_em_scaling_fit_res_{resolution}.png", args.plot_dpi)
    LOGGER.info("Saved %s scheme3 Beta-Binomial EM fit to %s", args.task, output_dir)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fit Qwen Retrieval/Reasoning scheme3 unified-Z scaling law with group-level Beta-Binomial EM.")
    parser.add_argument("--csv_path", type=Path, required=True, help="Task scaling CSV path.")
    parser.add_argument("--task", type=str, choices=["retrieval", "reasoning"], required=True, help="Downstream task name.")
    parser.add_argument("--ocr_fit_dir", type=Path, default=Path("VQAExp/scaling_law/Fitting/ocr"), help="OCR fit directory containing per-resolution em_fit_summary.json files.")
    parser.add_argument("--output_dir", type=Path, default=Path("VQAExp/scaling_law/Fitting/scheme3_betabinomial_em"), help="Root output directory.")
    parser.add_argument("--resolutions", type=str, default="512,640,1024,1280", help="Comma-separated resolutions to include.")
    parser.add_argument("--holdout_layout", type=str, default=DEFAULT_HOLDOUT_LAYOUT, help="Layout held out from fitting.")
    parser.add_argument("--max_iter", type=int, default=DEFAULT_MAX_ITER, help="Maximum EM iterations.")
    parser.add_argument("--tol", type=float, default=DEFAULT_TOL, help="EM convergence tolerance on log-likelihood difference.")
    parser.add_argument("--init_threshold", type=float, default=DEFAULT_INIT_THRESHOLD, help="Initial hard-state threshold on observed bin accuracy.")
    parser.add_argument("--init_soft", type=float, default=DEFAULT_INIT_SOFT, help="Soft initialization mass for the non-selected state.")
    parser.add_argument("--eps", type=float, default=DEFAULT_EPS, help="Reserved probability epsilon for compatibility.")
    parser.add_argument("--reg_strength", type=float, default=0.0, help="L2 regularization strength for task gating parameters.")
    parser.add_argument("--bin_size", type=int, default=DEFAULT_BIN_SIZE, help="Character bin size for group-level Beta-Binomial observations.")
    parser.add_argument("--text_len_scale", type=float, default=DEFAULT_TEXT_LEN_SCALE, help="Scale used in log_text_len=log(text_len/text_len_scale).")
    parser.add_argument("--plot_dpi", type=int, default=200, help="DPI for saved plots.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    run_fit(args)


if __name__ == "__main__":
    main()
