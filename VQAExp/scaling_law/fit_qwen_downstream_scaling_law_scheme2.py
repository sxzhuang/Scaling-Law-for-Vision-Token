import argparse
import csv
import json
import logging
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize
from scipy.special import expit


LOGGER = logging.getLogger(__name__)
FONT_SIZES = [20, 28, 36]
LINE_SPACINGS = [0, 6, 24, 42]
LETTER_SPACINGS = [-1, 0, 7]
RESOLUTIONS = [512, 640, 1024, 1280]
DEFAULT_HOLDOUT_LAYOUT = "20_0_7"
DEFAULT_EPS = 1e-8
DEFAULT_BIN_SIZE = 1000
DEFAULT_TEXT_LEN_SCALE = 1000.0


def iter_layouts() -> list[str]:
    return [f"{font_size}_{line_spacing}_{letter_spacing}" for font_size in FONT_SIZES for line_spacing in LINE_SPACINGS for letter_spacing in LETTER_SPACINGS]


def parse_resolutions(resolutions_arg: str) -> list[int]:
    return [int(part.strip()) for part in resolutions_arg.split(",") if part.strip()]


def logit(value: float) -> float:
    value = min(max(float(value), 1e-6), 1.0 - 1e-6)
    return math.log(value / (1.0 - value))


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
            row["log_text_len"] = math.log(float(text_len) / float(text_len_scale))
            row["log_complexity"] = float(row["log_complexity"])
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
    resolution_values = np.array(sorted(gating_params.keys()), dtype=int)
    resolution_to_index = {int(value): index for index, value in enumerate(resolution_values)}
    resolution_index = np.array([resolution_to_index[int(row["resolution"])] for row in rows], dtype=int)
    ocr_z = np.zeros(len(rows), dtype=float)
    for index, row in enumerate(rows):
        params = gating_params[int(row["resolution"])]
        ocr_z[index] = params["w0"] + params["a"] * row["qwen_logV"] + params["alpha"] * row["qwen_logG"]
    ocr_epsilon = expit(ocr_z)
    return {"log_text_len": log_text_len, "log_complexity": log_complexity, "y": y, "resolution": resolution, "resolution_values": resolution_values, "resolution_index": resolution_index, "ocr_z": ocr_z, "ocr_epsilon": ocr_epsilon}


def unpack_params(params: np.ndarray, n_resolutions: int) -> dict[str, np.ndarray | float]:
    beta0_by_resolution = params[:n_resolutions].astype(float)
    lambda0 = float(params[n_resolutions])
    d0 = float(params[n_resolutions + 1])
    beta1_start = n_resolutions + 2
    beta1_by_resolution = params[beta1_start:beta1_start + n_resolutions].astype(float)
    lambda1 = float(params[beta1_start + n_resolutions])
    d1 = float(params[beta1_start + n_resolutions + 1])
    return {"beta0_by_resolution": beta0_by_resolution, "lambda0": lambda0, "d0": d0, "beta1_by_resolution": beta1_by_resolution, "lambda1": lambda1, "d1": d1}


def predict_components(params: np.ndarray, ocr_epsilon: np.ndarray, log_text_len: np.ndarray, log_complexity: np.ndarray, resolution_index: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n_resolutions = int(np.max(resolution_index)) + 1
    unpacked = unpack_params(params, n_resolutions)
    beta0 = unpacked["beta0_by_resolution"][resolution_index]
    beta1 = unpacked["beta1_by_resolution"][resolution_index]
    lambda0 = float(unpacked["lambda0"])
    d0 = float(unpacked["d0"])
    lambda1 = float(unpacked["lambda1"])
    d1 = float(unpacked["d1"])
    mu0 = expit(beta0 - lambda0 * log_text_len - d0 * log_complexity)
    mu1 = expit(beta1 - lambda1 * log_text_len - d1 * log_complexity)
    predicted_success = (1.0 - ocr_epsilon) * mu0 + ocr_epsilon * mu1
    return mu0, mu1, predicted_success


def predict_success(params: np.ndarray, ocr_epsilon: np.ndarray, log_text_len: np.ndarray, log_complexity: np.ndarray, resolution_index: np.ndarray) -> np.ndarray:
    _, _, predicted_success = predict_components(params, ocr_epsilon, log_text_len, log_complexity, resolution_index)
    return predicted_success


def bernoulli_nll(params: np.ndarray, ocr_epsilon: np.ndarray, log_text_len: np.ndarray, log_complexity: np.ndarray, resolution_index: np.ndarray, y: np.ndarray, eps: float, reg_strength: float) -> float:
    p = np.clip(predict_success(params, ocr_epsilon, log_text_len, log_complexity, resolution_index), eps, 1.0 - eps)
    nll = -float(np.sum(y * np.log(p) + (1.0 - y) * np.log1p(-p)))
    penalty = 0.5 * float(reg_strength) * float(np.sum(params**2))
    return nll + penalty


def build_beta_order_constraints(n_resolutions: int) -> list[dict]:
    constraints = []
    beta1_start = n_resolutions + 2
    for resolution_index in range(n_resolutions):
        constraints.append({"type": "ineq", "fun": lambda params, index=resolution_index: params[index] - params[beta1_start + index]})
    return constraints


def fit_downstream(arrays: dict[str, np.ndarray], train_mask: np.ndarray, eps: float, reg_strength: float, max_iter: int, constrain_sensitivity_nonnegative: bool) -> tuple[np.ndarray, float]:
    y_train = arrays["y"][train_mask]
    ocr_epsilon_train = arrays["ocr_epsilon"][train_mask]
    log_text_len_train = arrays["log_text_len"][train_mask]
    log_complexity_train = arrays["log_complexity"][train_mask]
    resolution_index_train = arrays["resolution_index"][train_mask]
    n_resolutions = len(arrays["resolution_values"])
    mean_y = float(np.mean(y_train))
    beta0_high = logit(min(mean_y + 0.2, 0.95))
    beta1_low = logit(max(mean_y - 0.2, 0.05))
    starts = [
        np.concatenate([np.full(n_resolutions, logit(mean_y)), np.array([0.0, 0.0]), np.full(n_resolutions, logit(mean_y)), np.array([0.0, 0.0])]),
        np.concatenate([np.full(n_resolutions, beta0_high), np.array([0.5, 0.0]), np.full(n_resolutions, beta1_low), np.array([0.5, 0.0])]),
        np.concatenate([np.full(n_resolutions, 2.0), np.array([0.5, 0.5]), np.full(n_resolutions, -2.0), np.array([0.5, 0.5])]),
        np.concatenate([np.full(n_resolutions, 4.0), np.array([1.0, 1.0]), np.full(n_resolutions, 0.0), np.array([1.0, 1.0])]),
        np.zeros(2 * n_resolutions + 4, dtype=float),
    ]
    bounds = None
    if constrain_sensitivity_nonnegative:
        bounds = [(None, None)] * (2 * n_resolutions + 4)
        bounds[n_resolutions] = (0.0, None)
        bounds[n_resolutions + 1] = (0.0, None)
        bounds[2 * n_resolutions + 2] = (0.0, None)
        bounds[2 * n_resolutions + 3] = (0.0, None)
    best_params = starts[0]
    best_nll = float("inf")
    constraints = build_beta_order_constraints(n_resolutions)
    for start in starts:
        result = minimize(bernoulli_nll, start, args=(ocr_epsilon_train, log_text_len_train, log_complexity_train, resolution_index_train, y_train, eps, reg_strength), method="SLSQP", bounds=bounds, constraints=constraints, options={"maxiter": int(max_iter), "ftol": 1e-8})
        candidate_params = result.x if result.success and np.all(np.isfinite(result.x)) else start
        candidate_nll = bernoulli_nll(candidate_params, ocr_epsilon_train, log_text_len_train, log_complexity_train, resolution_index_train, y_train, eps, reg_strength)
        if candidate_nll < best_nll:
            best_params = candidate_params.astype(float)
            best_nll = float(candidate_nll)
    return best_params, best_nll


def compute_nll_for_mask(params: np.ndarray, arrays: dict[str, np.ndarray], mask: np.ndarray, eps: float) -> float | None:
    if not np.any(mask):
        return None
    return float(bernoulli_nll(params, arrays["ocr_epsilon"][mask], arrays["log_text_len"][mask], arrays["log_complexity"][mask], arrays["resolution_index"][mask], arrays["y"][mask], eps, 0.0))


def add_predictions(rows: list[dict], arrays: dict[str, np.ndarray], params: np.ndarray, holdout_layout: str) -> list[dict]:
    mu0, mu1, predicted_success = predict_components(params, arrays["ocr_epsilon"], arrays["log_text_len"], arrays["log_complexity"], arrays["resolution_index"])
    output_rows = []
    for index, row in enumerate(rows):
        out = dict(row)
        out["ocr_z"] = float(arrays["ocr_z"][index])
        out["ocr_epsilon"] = float(arrays["ocr_epsilon"][index])
        out["mu0"] = float(mu0[index])
        out["mu1"] = float(mu1[index])
        out["predicted_success"] = float(predicted_success[index])
        out["split"] = "test" if row["layout"] == holdout_layout else "train"
        output_rows.append(out)
    return output_rows


def write_csv(rows: list[dict], output_csv: Path) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with output_csv.open("w", newline="", encoding="utf-8") as file_handle:
        writer = csv.DictWriter(file_handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_binned_rows(rows: list[dict], bin_size: int) -> list[dict]:
    grouped: dict[tuple[str, int, int], list[dict]] = {}
    for row in rows:
        bin_start = int(row["text_len"]) // int(bin_size) * int(bin_size)
        grouped.setdefault((row["layout"], int(row["resolution"]), bin_start), []).append(row)
    binned_rows = []
    for (layout, resolution, bin_start), group in sorted(grouped.items()):
        correct_values = [float(row["correct"]) for row in group]
        predicted_values = [float(row["predicted_success"]) for row in group]
        mu0_values = [float(row["mu0"]) for row in group]
        mu1_values = [float(row["mu1"]) for row in group]
        epsilon_values = [float(row["ocr_epsilon"]) for row in group]
        text_lens = [int(row["text_len"]) for row in group]
        split_values = sorted({str(row["split"]) for row in group})
        binned_rows.append(
            {
                "layout": layout,
                "resolution": resolution,
                "split": split_values[0] if len(split_values) == 1 else "mixed",
                "bin_start": bin_start,
                "bin_end": bin_start + int(bin_size),
                "bin_center": bin_start + int(bin_size) / 2,
                "n": len(group),
                "mean_text_len": float(np.mean(text_lens)),
                "observed_acc": float(np.mean(correct_values)),
                "predicted_acc": float(np.mean(predicted_values)),
                "mean_mu0": float(np.mean(mu0_values)),
                "mean_mu1": float(np.mean(mu1_values)),
                "mean_ocr_epsilon": float(np.mean(epsilon_values)),
            }
        )
    return binned_rows


def compute_binned_mae(binned_rows: list[dict], split: str | None = None) -> float | None:
    selected = [row for row in binned_rows if split is None or row["split"] == split]
    if not selected:
        return None
    return float(np.mean([abs(float(row["observed_acc"]) - float(row["predicted_acc"])) for row in selected]))


def plot_resolution(rows: list[dict], binned_rows: list[dict], task: str, resolution: int, output_file: Path, dpi: int) -> None:
    layouts = iter_layouts()
    fig, axes = plt.subplots(6, 6, figsize=(24, 18), sharex=True, sharey=True)
    axes_flat = axes.ravel()
    for ax, layout in zip(axes_flat, layouts):
        group = [row for row in rows if row["layout"] == layout and int(row["resolution"]) == int(resolution)]
        bin_group = [row for row in binned_rows if row["layout"] == layout and int(row["resolution"]) == int(resolution)]
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
    fig.suptitle(f"Qwen {task} scheme2 downstream scaling-law fit, resolution={resolution}", fontsize=16)
    fig.tight_layout(rect=(0, 0, 1, 0.975))
    output_file.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_file, dpi=int(dpi))
    plt.close(fig)


def summarize(rows: list[dict], arrays: dict[str, np.ndarray], params: np.ndarray, train_mask: np.ndarray, test_mask: np.ndarray, train_nll: float, binned_rows: list[dict], eps: float, args: argparse.Namespace) -> dict:
    unpacked = unpack_params(params, len(arrays["resolution_values"]))
    beta0_by_resolution = {str(int(resolution)): float(value) for resolution, value in zip(arrays["resolution_values"], unpacked["beta0_by_resolution"])}
    beta1_by_resolution = {str(int(resolution)): float(value) for resolution, value in zip(arrays["resolution_values"], unpacked["beta1_by_resolution"])}
    return {
        "scheme": "scheme2_fixed_ocr_resolution_specific_downstream_baseline_beta0_ge_beta1",
        "task": args.task,
        "csv_path": str(args.csv_path),
        "ocr_fit_dir": str(args.ocr_fit_dir),
        "holdout_layout": args.holdout_layout,
        "n_samples": len(rows),
        "n_train": int(np.sum(train_mask)),
        "n_test": int(np.sum(test_mask)),
        "train_accuracy_mean": float(np.mean(arrays["y"][train_mask])) if np.any(train_mask) else None,
        "test_accuracy_mean": float(np.mean(arrays["y"][test_mask])) if np.any(test_mask) else None,
        "params": {"beta0_by_resolution": beta0_by_resolution, "lambda0": float(unpacked["lambda0"]), "d0": float(unpacked["d0"]), "beta1_by_resolution": beta1_by_resolution, "lambda1": float(unpacked["lambda1"]), "d1": float(unpacked["d1"])},
        "train_nll": float(train_nll),
        "test_nll": compute_nll_for_mask(params, arrays, test_mask, eps),
        "all_nll": compute_nll_for_mask(params, arrays, np.ones(len(rows), dtype=bool), eps),
        "mean_ocr_epsilon": float(np.mean(arrays["ocr_epsilon"])),
        "binned_mae_all": compute_binned_mae(binned_rows),
        "binned_mae_train": compute_binned_mae(binned_rows, "train"),
        "binned_mae_test": compute_binned_mae(binned_rows, "test"),
        "eps": float(args.eps),
        "reg_strength": float(args.reg_strength),
        "bin_size": int(args.bin_size),
        "text_len_scale": float(args.text_len_scale),
        "log_text_len_definition": "log(text_len / text_len_scale)",
        "downstream_baseline_definition": "beta0 and beta1 are fitted separately for each resolution; lambda and d are shared across resolutions.",
        "beta_order_constraint": "beta0_by_resolution[resolution] >= beta1_by_resolution[resolution]",
        "constrain_sensitivity_nonnegative": bool(args.constrain_sensitivity_nonnegative),
    }


def run_fit(args: argparse.Namespace) -> None:
    resolutions = parse_resolutions(args.resolutions)
    rows = load_rows(args.csv_path, resolutions, args.text_len_scale)
    gating_params = load_ocr_gating_params(args.ocr_fit_dir, resolutions)
    arrays = rows_to_arrays(rows, gating_params)
    train_mask = np.array([row["layout"] != args.holdout_layout for row in rows], dtype=bool)
    test_mask = ~train_mask
    if not np.any(train_mask):
        raise ValueError("No training rows remain after applying holdout_layout.")
    params, train_nll = fit_downstream(arrays, train_mask, args.eps, args.reg_strength, args.max_iter, args.constrain_sensitivity_nonnegative)
    output_rows = add_predictions(rows, arrays, params, args.holdout_layout)
    binned_rows = build_binned_rows(output_rows, args.bin_size)
    output_dir = args.output_dir / args.task
    plots_dir = output_dir / "plots"
    output_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_rows, output_dir / "fit_points.csv")
    write_csv(binned_rows, output_dir / "fit_points_binned.csv")
    summary = summarize(rows, arrays, params, train_mask, test_mask, train_nll, binned_rows, args.eps, args)
    with (output_dir / "fit_summary.json").open("w", encoding="utf-8") as file_handle:
        json.dump(summary, file_handle, ensure_ascii=False, indent=2)
    for resolution in resolutions:
        plot_resolution(output_rows, binned_rows, args.task, resolution, plots_dir / f"qwen_{args.task}_scheme2_scaling_fit_res_{resolution}.png", args.plot_dpi)
    LOGGER.info("Saved %s scheme2 fit to %s", args.task, output_dir)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fit Qwen Retrieval/Reasoning scheme2 scaling law with fixed OCR gating, resolution-specific downstream baselines, and beta0>=beta1 constraints.")
    parser.add_argument("--csv_path", type=Path, required=True, help="Task scaling CSV path.")
    parser.add_argument("--task", type=str, choices=["retrieval", "reasoning"], required=True, help="Downstream task name.")
    parser.add_argument("--ocr_fit_dir", type=Path, default=Path("VQAExp/scaling_law/Fitting/ocr"), help="OCR fit directory containing per-resolution em_fit_summary.json files.")
    parser.add_argument("--output_dir", type=Path, default=Path("VQAExp/scaling_law/Fitting/scheme2_resolution_beta_constrained"), help="Root output directory.")
    parser.add_argument("--resolutions", type=str, default="512,640,1024,1280", help="Comma-separated resolutions to include.")
    parser.add_argument("--holdout_layout", type=str, default=DEFAULT_HOLDOUT_LAYOUT, help="Layout held out from fitting.")
    parser.add_argument("--max_iter", type=int, default=1000, help="Maximum optimizer iterations.")
    parser.add_argument("--eps", type=float, default=DEFAULT_EPS, help="Probability clipping epsilon for Bernoulli NLL.")
    parser.add_argument("--reg_strength", type=float, default=0.0, help="L2 regularization strength for downstream parameters.")
    parser.add_argument("--bin_size", type=int, default=DEFAULT_BIN_SIZE, help="Character bin size for plotting observed and predicted bin-level accuracy.")
    parser.add_argument("--text_len_scale", type=float, default=DEFAULT_TEXT_LEN_SCALE, help="Scale used in log_text_len=log(text_len/text_len_scale).")
    parser.add_argument("--plot_dpi", type=int, default=200, help="DPI for saved plots.")
    parser.add_argument("--constrain_sensitivity_nonnegative", action="store_true", help="Constrain lambda and d parameters to be nonnegative.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    run_fit(args)


if __name__ == "__main__":
    main()
