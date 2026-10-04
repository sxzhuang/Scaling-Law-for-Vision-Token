import argparse
import csv
import json
import logging
import math
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit


LOGGER = logging.getLogger(__name__)
RESOLUTIONS = [512, 640, 1024, 1280]
DEFAULT_HOLDOUT_LAYOUT = "20_0_7"
DEFAULT_TEXT_LEN_SCALE = 1000.0
DEFAULT_BIN_SIZE = 1000
DEFAULT_EPS = 1e-8


def parse_resolutions(resolutions_arg: str) -> list[int]:
    return [int(part.strip()) for part in resolutions_arg.split(",") if part.strip()]


def logit(value: float) -> float:
    value = min(max(float(value), 1e-6), 1.0 - 1e-6)
    return math.log(value / (1.0 - value))


def load_task_rows(csv_path: Path, resolutions: list[int], text_len_scale: float) -> list[dict]:
    rows = []
    with csv_path.open("r", newline="", encoding="utf-8") as file_handle:
        reader = csv.DictReader(file_handle)
        required = {"image_id", "layout", "resolution", "text_len", "qwen_logV", "qwen_logG", "log_complexity", "correct"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Task CSV is missing required fields: {sorted(missing)}")
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
            rows.append(row)
    if not rows:
        raise ValueError(f"No rows loaded from {csv_path}")
    return rows


def load_ocr_gating_params(ocr_fit_dir: Path, resolutions: list[int]) -> dict[int, dict[str, float]]:
    params = {}
    for resolution in resolutions:
        summary_path = ocr_fit_dir / str(resolution) / "em_fit_summary.json"
        with summary_path.open("r", encoding="utf-8") as file_handle:
            summary = json.load(file_handle)
        gating = summary["gating_params"]
        params[int(resolution)] = {"w0": float(gating["w0"]), "a": float(gating["a"]), "alpha": float(gating["alpha"])}
    return params


def load_ocr_responsibility_rows(ocr_points_csv: Path, resolutions: list[int], holdout_layout: str, text_len_scale: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    log_text_len = []
    responsibility = []
    resolution_values = np.array(sorted(resolutions), dtype=int)
    resolution_to_index = {int(value): index for index, value in enumerate(resolution_values)}
    resolution_index = []
    train_mask = []
    with ocr_points_csv.open("r", newline="", encoding="utf-8") as file_handle:
        reader = csv.DictReader(file_handle)
        required = {"layout", "resolution", "text_len", "r"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"OCR points CSV is missing required fields: {sorted(missing)}")
        for row in reader:
            resolution = int(row["resolution"])
            if resolution not in resolutions:
                continue
            text_len = int(float(row["text_len"]))
            if text_len <= 0:
                continue
            log_text_len.append(math.log(float(text_len) / float(text_len_scale)))
            responsibility.append(float(row["r"]))
            resolution_index.append(resolution_to_index[resolution])
            train_mask.append(row["layout"] != holdout_layout)
    if not log_text_len:
        raise ValueError(f"No OCR responsibility rows loaded from {ocr_points_csv}")
    return np.array(log_text_len, dtype=float), np.array(responsibility, dtype=float), np.array(resolution_index, dtype=int), np.array(train_mask, dtype=bool)


def arrays_from_rows(rows: list[dict], resolutions: list[int], ocr_gating_params: dict[int, dict[str, float]]) -> dict[str, np.ndarray]:
    resolution_values = np.array(sorted(resolutions), dtype=int)
    resolution_to_index = {int(value): index for index, value in enumerate(resolution_values)}
    log_text_len = np.array([row["log_text_len"] for row in rows], dtype=float)
    log_complexity = np.array([row["log_complexity"] for row in rows], dtype=float)
    y = np.array([row["correct"] for row in rows], dtype=float)
    resolution = np.array([row["resolution"] for row in rows], dtype=int)
    resolution_index = np.array([resolution_to_index[int(row["resolution"])] for row in rows], dtype=int)
    ocr_z = np.zeros(len(rows), dtype=float)
    for index, row in enumerate(rows):
        params = ocr_gating_params[int(row["resolution"])]
        ocr_z[index] = params["w0"] + params["a"] * row["qwen_logV"] + params["alpha"] * row["qwen_logG"]
    return {"log_text_len": log_text_len, "log_complexity": log_complexity, "y": y, "resolution": resolution, "resolution_values": resolution_values, "resolution_index": resolution_index, "ocr_z": ocr_z, "full_ocr_epsilon": expit(ocr_z)}


def length_gate_nll(params: np.ndarray, log_text_len: np.ndarray, resolution_index: np.ndarray, responsibility: np.ndarray, variant: str, n_resolutions: int, eps: float, reg_strength: float) -> float:
    if variant == "length_shared_gate":
        z = params[0] + params[1] * log_text_len
    elif variant == "length_resolution_gate":
        intercept = params[:n_resolutions][resolution_index]
        slope = params[n_resolutions:2 * n_resolutions][resolution_index]
        z = intercept + slope * log_text_len
    else:
        raise ValueError(f"Unsupported length gate variant: {variant}")
    p = np.clip(expit(z), eps, 1.0 - eps)
    nll = -float(np.sum(responsibility * np.log(p) + (1.0 - responsibility) * np.log1p(-p)))
    return nll + 0.5 * float(reg_strength) * float(np.sum(params**2))


def fit_length_gate(log_text_len: np.ndarray, responsibility: np.ndarray, resolution_index: np.ndarray, train_mask: np.ndarray, variant: str, n_resolutions: int, eps: float, reg_strength: float, max_iter: int) -> dict:
    x_train = log_text_len[train_mask]
    r_train = responsibility[train_mask]
    idx_train = resolution_index[train_mask]
    starts = []
    mean_r = float(np.mean(r_train))
    if variant == "length_shared_gate":
        starts = [np.array([logit(mean_r), 0.0]), np.array([0.0, 1.0]), np.array([-2.0, 2.0])]
    elif variant == "length_resolution_gate":
        starts = [
            np.concatenate([np.full(n_resolutions, logit(mean_r)), np.zeros(n_resolutions)]),
            np.concatenate([np.zeros(n_resolutions), np.ones(n_resolutions)]),
            np.concatenate([np.full(n_resolutions, -2.0), np.full(n_resolutions, 2.0)]),
        ]
    best_params = starts[0]
    best_nll = float("inf")
    for start in starts:
        result = minimize(length_gate_nll, start, args=(x_train, idx_train, r_train, variant, n_resolutions, eps, reg_strength), method="L-BFGS-B", options={"maxiter": int(max_iter), "ftol": 1e-10})
        candidate = result.x if result.success and np.all(np.isfinite(result.x)) else start
        candidate_nll = length_gate_nll(candidate, x_train, idx_train, r_train, variant, n_resolutions, eps, reg_strength)
        if candidate_nll < best_nll:
            best_params = candidate.astype(float)
            best_nll = float(candidate_nll)
    return {"params": best_params, "train_nll": best_nll}


def apply_length_gate(params: np.ndarray, log_text_len: np.ndarray, resolution_index: np.ndarray, variant: str, n_resolutions: int) -> np.ndarray:
    if variant == "length_shared_gate":
        z = params[0] + params[1] * log_text_len
    elif variant == "length_resolution_gate":
        z = params[:n_resolutions][resolution_index] + params[n_resolutions:2 * n_resolutions][resolution_index] * log_text_len
    else:
        raise ValueError(f"Unsupported length gate variant: {variant}")
    return expit(z)


def unpack_mixture_params(params: np.ndarray, n_resolutions: int) -> dict:
    beta0_by_resolution = params[:n_resolutions].astype(float)
    lambda0 = float(params[n_resolutions])
    d0 = float(params[n_resolutions + 1])
    beta1_start = n_resolutions + 2
    beta1_by_resolution = params[beta1_start:beta1_start + n_resolutions].astype(float)
    lambda1 = float(params[beta1_start + n_resolutions])
    d1 = float(params[beta1_start + n_resolutions + 1])
    return {"beta0_by_resolution": beta0_by_resolution, "lambda0": lambda0, "d0": d0, "beta1_by_resolution": beta1_by_resolution, "lambda1": lambda1, "d1": d1}


def mixture_components(params: np.ndarray, gate_epsilon: np.ndarray, log_text_len: np.ndarray, log_complexity: np.ndarray, resolution_index: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n_resolutions = int(np.max(resolution_index)) + 1
    unpacked = unpack_mixture_params(params, n_resolutions)
    beta0 = unpacked["beta0_by_resolution"][resolution_index]
    beta1 = unpacked["beta1_by_resolution"][resolution_index]
    mu0 = expit(beta0 - float(unpacked["lambda0"]) * log_text_len - float(unpacked["d0"]) * log_complexity)
    mu1 = expit(beta1 - float(unpacked["lambda1"]) * log_text_len - float(unpacked["d1"]) * log_complexity)
    predicted_success = (1.0 - gate_epsilon) * mu0 + gate_epsilon * mu1
    return mu0, mu1, predicted_success


def mixture_nll(params: np.ndarray, gate_epsilon: np.ndarray, log_text_len: np.ndarray, log_complexity: np.ndarray, resolution_index: np.ndarray, y: np.ndarray, eps: float, reg_strength: float) -> float:
    p = np.clip(mixture_components(params, gate_epsilon, log_text_len, log_complexity, resolution_index)[2], eps, 1.0 - eps)
    nll = -float(np.sum(y * np.log(p) + (1.0 - y) * np.log1p(-p)))
    return nll + 0.5 * float(reg_strength) * float(np.sum(params**2))


def beta_order_constraints(n_resolutions: int) -> list[dict]:
    constraints = []
    beta1_start = n_resolutions + 2
    for resolution_index in range(n_resolutions):
        constraints.append({"type": "ineq", "fun": lambda params, index=resolution_index: params[index] - params[beta1_start + index]})
    return constraints


def fit_mixture_downstream(arrays: dict[str, np.ndarray], gate_epsilon: np.ndarray, train_mask: np.ndarray, eps: float, reg_strength: float, max_iter: int) -> tuple[np.ndarray, float]:
    n_resolutions = len(arrays["resolution_values"])
    y_train = arrays["y"][train_mask]
    mean_y = float(np.mean(y_train))
    starts = [
        np.concatenate([np.full(n_resolutions, logit(mean_y)), np.array([0.0, 0.0]), np.full(n_resolutions, logit(mean_y)), np.array([0.0, 0.0])]),
        np.concatenate([np.full(n_resolutions, logit(min(mean_y + 0.2, 0.95))), np.array([0.5, 0.0]), np.full(n_resolutions, logit(max(mean_y - 0.2, 0.05))), np.array([0.5, 0.0])]),
        np.concatenate([np.full(n_resolutions, 4.0), np.array([1.0, 1.0]), np.full(n_resolutions, 0.0), np.array([1.0, 1.0])]),
        np.zeros(2 * n_resolutions + 4, dtype=float),
    ]
    constraints = beta_order_constraints(n_resolutions)
    best_params = starts[0]
    best_nll = float("inf")
    for start in starts:
        result = minimize(mixture_nll, start, args=(gate_epsilon[train_mask], arrays["log_text_len"][train_mask], arrays["log_complexity"][train_mask], arrays["resolution_index"][train_mask], y_train, eps, reg_strength), method="SLSQP", constraints=constraints, options={"maxiter": int(max_iter), "ftol": 1e-8})
        candidate = result.x if result.success and np.all(np.isfinite(result.x)) else start
        candidate_nll = mixture_nll(candidate, gate_epsilon[train_mask], arrays["log_text_len"][train_mask], arrays["log_complexity"][train_mask], arrays["resolution_index"][train_mask], y_train, eps, reg_strength)
        if candidate_nll < best_nll:
            best_params = candidate.astype(float)
            best_nll = float(candidate_nll)
    return best_params, best_nll


def single_logistic_components(params: np.ndarray, log_text_len: np.ndarray, log_complexity: np.ndarray, resolution_index: np.ndarray) -> np.ndarray:
    n_resolutions = int(np.max(resolution_index)) + 1
    beta = params[:n_resolutions][resolution_index]
    lambda_value = float(params[n_resolutions])
    d_value = float(params[n_resolutions + 1])
    return expit(beta - lambda_value * log_text_len - d_value * log_complexity)


def single_logistic_nll(params: np.ndarray, log_text_len: np.ndarray, log_complexity: np.ndarray, resolution_index: np.ndarray, y: np.ndarray, eps: float, reg_strength: float) -> float:
    p = np.clip(single_logistic_components(params, log_text_len, log_complexity, resolution_index), eps, 1.0 - eps)
    nll = -float(np.sum(y * np.log(p) + (1.0 - y) * np.log1p(-p)))
    return nll + 0.5 * float(reg_strength) * float(np.sum(params**2))


def fit_single_logistic_downstream(arrays: dict[str, np.ndarray], train_mask: np.ndarray, eps: float, reg_strength: float, max_iter: int) -> tuple[np.ndarray, float]:
    n_resolutions = len(arrays["resolution_values"])
    mean_y = float(np.mean(arrays["y"][train_mask]))
    starts = [
        np.concatenate([np.full(n_resolutions, logit(mean_y)), np.array([0.0, 0.0])]),
        np.concatenate([np.full(n_resolutions, 2.0), np.array([1.0, 1.0])]),
        np.zeros(n_resolutions + 2, dtype=float),
    ]
    best_params = starts[0]
    best_nll = float("inf")
    for start in starts:
        result = minimize(single_logistic_nll, start, args=(arrays["log_text_len"][train_mask], arrays["log_complexity"][train_mask], arrays["resolution_index"][train_mask], arrays["y"][train_mask], eps, reg_strength), method="L-BFGS-B", options={"maxiter": int(max_iter), "ftol": 1e-10})
        candidate = result.x if result.success and np.all(np.isfinite(result.x)) else start
        candidate_nll = single_logistic_nll(candidate, arrays["log_text_len"][train_mask], arrays["log_complexity"][train_mask], arrays["resolution_index"][train_mask], arrays["y"][train_mask], eps, reg_strength)
        if candidate_nll < best_nll:
            best_params = candidate.astype(float)
            best_nll = float(candidate_nll)
    return best_params, best_nll


def compute_binned_rows(rows: list[dict], bin_size: int) -> list[dict]:
    grouped: dict[tuple[str, int, int], list[dict]] = {}
    for row in rows:
        bin_start = int(row["text_len"]) // int(bin_size) * int(bin_size)
        grouped.setdefault((row["layout"], int(row["resolution"]), bin_start), []).append(row)
    binned_rows = []
    for (layout, resolution, bin_start), group in sorted(grouped.items()):
        correct_values = [float(row["correct"]) for row in group]
        predicted_values = [float(row["predicted_success"]) for row in group]
        text_lens = [int(row["text_len"]) for row in group]
        split_values = sorted({str(row["split"]) for row in group})
        gate_values = [float(row["gate_epsilon"]) for row in group if str(row.get("gate_epsilon", "")) != ""]
        binned_rows.append({"layout": layout, "resolution": resolution, "split": split_values[0] if len(split_values) == 1 else "mixed", "bin_start": bin_start, "bin_end": bin_start + int(bin_size), "bin_center": bin_start + int(bin_size) / 2, "n": len(group), "mean_text_len": float(np.mean(text_lens)), "observed_acc": float(np.mean(correct_values)), "predicted_acc": float(np.mean(predicted_values)), "mean_gate_epsilon": float(np.mean(gate_values)) if gate_values else None})
    return binned_rows


def binned_mae(binned_rows: list[dict], split: str | None = None) -> float | None:
    selected = [row for row in binned_rows if split is None or row["split"] == split]
    if not selected:
        return None
    return float(np.mean([abs(float(row["observed_acc"]) - float(row["predicted_acc"])) for row in selected]))


def write_csv(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file_handle:
        writer = csv.DictWriter(file_handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def compute_nll(predicted_success: np.ndarray, y: np.ndarray, mask: np.ndarray, eps: float) -> float | None:
    if not np.any(mask):
        return None
    p = np.clip(predicted_success[mask], eps, 1.0 - eps)
    yy = y[mask]
    return -float(np.sum(yy * np.log(p) + (1.0 - yy) * np.log1p(-p)))


def evaluate_variant(rows: list[dict], arrays: dict[str, np.ndarray], variant: str, gate_epsilon: np.ndarray | None, train_mask: np.ndarray, test_mask: np.ndarray, args: argparse.Namespace) -> tuple[list[dict], list[dict], dict]:
    if variant == "no_ocr_gate":
        params, train_nll = fit_single_logistic_downstream(arrays, train_mask, args.eps, args.reg_strength, args.max_iter)
        predicted_success = single_logistic_components(params, arrays["log_text_len"], arrays["log_complexity"], arrays["resolution_index"])
        output_rows = []
        for index, row in enumerate(rows):
            out = dict(row)
            out["ablation_variant"] = variant
            out["gate_epsilon"] = ""
            out["mu0"] = ""
            out["mu1"] = ""
            out["predicted_success"] = float(predicted_success[index])
            out["split"] = "test" if row["layout"] == args.holdout_layout else "train"
            output_rows.append(out)
        beta_by_resolution = {str(int(resolution)): float(value) for resolution, value in zip(arrays["resolution_values"], params[:len(arrays["resolution_values"])])}
        param_summary = {"beta_by_resolution": beta_by_resolution, "lambda": float(params[len(arrays["resolution_values"])]), "d": float(params[len(arrays["resolution_values"]) + 1])}
        gate_summary = None
    else:
        if gate_epsilon is None:
            raise ValueError(f"Variant {variant} requires a gate_epsilon array")
        params, train_nll = fit_mixture_downstream(arrays, gate_epsilon, train_mask, args.eps, args.reg_strength, args.max_iter)
        mu0, mu1, predicted_success = mixture_components(params, gate_epsilon, arrays["log_text_len"], arrays["log_complexity"], arrays["resolution_index"])
        output_rows = []
        for index, row in enumerate(rows):
            out = dict(row)
            out["ablation_variant"] = variant
            out["gate_epsilon"] = float(gate_epsilon[index])
            out["mu0"] = float(mu0[index])
            out["mu1"] = float(mu1[index])
            out["predicted_success"] = float(predicted_success[index])
            out["split"] = "test" if row["layout"] == args.holdout_layout else "train"
            output_rows.append(out)
        unpacked = unpack_mixture_params(params, len(arrays["resolution_values"]))
        beta0_by_resolution = {str(int(resolution)): float(value) for resolution, value in zip(arrays["resolution_values"], unpacked["beta0_by_resolution"])}
        beta1_by_resolution = {str(int(resolution)): float(value) for resolution, value in zip(arrays["resolution_values"], unpacked["beta1_by_resolution"])}
        param_summary = {"beta0_by_resolution": beta0_by_resolution, "lambda0": float(unpacked["lambda0"]), "d0": float(unpacked["d0"]), "beta1_by_resolution": beta1_by_resolution, "lambda1": float(unpacked["lambda1"]), "d1": float(unpacked["d1"])}
        gate_summary = {"mean": float(np.mean(gate_epsilon)), "train_mean": float(np.mean(gate_epsilon[train_mask])), "test_mean": float(np.mean(gate_epsilon[test_mask])) if np.any(test_mask) else None}
    binned_rows = compute_binned_rows(output_rows, args.bin_size)
    summary = {"task": args.task, "variant": variant, "holdout_layout": args.holdout_layout, "n_samples": len(rows), "n_train": int(np.sum(train_mask)), "n_test": int(np.sum(test_mask)), "train_accuracy_mean": float(np.mean(arrays["y"][train_mask])), "test_accuracy_mean": float(np.mean(arrays["y"][test_mask])) if np.any(test_mask) else None, "params": param_summary, "gate_summary": gate_summary, "train_nll": float(train_nll), "test_nll": compute_nll(predicted_success, arrays["y"], test_mask, args.eps), "all_nll": compute_nll(predicted_success, arrays["y"], np.ones(len(rows), dtype=bool), args.eps), "binned_mae_all": binned_mae(binned_rows), "binned_mae_train": binned_mae(binned_rows, "train"), "binned_mae_test": binned_mae(binned_rows, "test"), "bin_size": int(args.bin_size), "text_len_scale": float(args.text_len_scale)}
    return output_rows, binned_rows, summary


def save_variant(output_root: Path, variant: str, output_rows: list[dict], binned_rows: list[dict], summary: dict) -> None:
    variant_dir = output_root / variant
    variant_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_rows, variant_dir / "fit_points.csv")
    write_csv(binned_rows, variant_dir / "fit_points_binned.csv")
    with (variant_dir / "fit_summary.json").open("w", encoding="utf-8") as file_handle:
        json.dump(summary, file_handle, ensure_ascii=False, indent=2)


def run_task(args: argparse.Namespace) -> list[dict]:
    resolutions = parse_resolutions(args.resolutions)
    rows = load_task_rows(args.csv_path, resolutions, args.text_len_scale)
    ocr_gating_params = load_ocr_gating_params(args.ocr_fit_dir, resolutions)
    arrays = arrays_from_rows(rows, resolutions, ocr_gating_params)
    train_mask = np.array([row["layout"] != args.holdout_layout for row in rows], dtype=bool)
    test_mask = ~train_mask
    n_resolutions = len(arrays["resolution_values"])

    ocr_log_text_len, ocr_responsibility, ocr_resolution_index, ocr_train_mask = load_ocr_responsibility_rows(args.ocr_points_csv, resolutions, args.holdout_layout, args.text_len_scale)
    length_gate_fits = {
        "length_shared_gate": fit_length_gate(ocr_log_text_len, ocr_responsibility, ocr_resolution_index, ocr_train_mask, "length_shared_gate", n_resolutions, args.eps, args.gate_reg_strength, args.max_iter),
        "length_resolution_gate": fit_length_gate(ocr_log_text_len, ocr_responsibility, ocr_resolution_index, ocr_train_mask, "length_resolution_gate", n_resolutions, args.eps, args.gate_reg_strength, args.max_iter),
    }
    gate_by_variant = {
        "length_shared_gate": apply_length_gate(length_gate_fits["length_shared_gate"]["params"], arrays["log_text_len"], arrays["resolution_index"], "length_shared_gate", n_resolutions),
        "length_resolution_gate": apply_length_gate(length_gate_fits["length_resolution_gate"]["params"], arrays["log_text_len"], arrays["resolution_index"], "length_resolution_gate", n_resolutions),
        "no_ocr_gate": None,
    }

    summary_rows = []
    task_output_root = args.output_dir / args.task
    for variant in ["length_shared_gate", "length_resolution_gate", "no_ocr_gate"]:
        LOGGER.info("Fitting %s %s", args.task, variant)
        output_rows, binned_rows, summary = evaluate_variant(rows, arrays, variant, gate_by_variant[variant], train_mask, test_mask, args)
        if variant in length_gate_fits:
            fit = length_gate_fits[variant]
            params = fit["params"]
            if variant == "length_shared_gate":
                summary["frozen_gate_fit"] = {"source": str(args.ocr_points_csv), "target": "OCR EM high-ED responsibility r", "params": {"c0": float(params[0]), "c1": float(params[1])}, "train_nll": float(fit["train_nll"])}
            else:
                c0_by_resolution = {str(int(resolution)): float(value) for resolution, value in zip(arrays["resolution_values"], params[:n_resolutions])}
                c1_by_resolution = {str(int(resolution)): float(value) for resolution, value in zip(arrays["resolution_values"], params[n_resolutions:2 * n_resolutions])}
                summary["frozen_gate_fit"] = {"source": str(args.ocr_points_csv), "target": "OCR EM high-ED responsibility r", "params": {"c0_by_resolution": c0_by_resolution, "c1_by_resolution": c1_by_resolution}, "train_nll": float(fit["train_nll"])}
        save_variant(task_output_root, variant, output_rows, binned_rows, summary)
        summary_rows.append({"task": args.task, "variant": variant, "binned_mae_all": summary["binned_mae_all"], "binned_mae_train": summary["binned_mae_train"], "binned_mae_test": summary["binned_mae_test"], "train_nll": summary["train_nll"], "test_nll": summary["test_nll"], "all_nll": summary["all_nll"], "train_accuracy_mean": summary["train_accuracy_mean"], "test_accuracy_mean": summary["test_accuracy_mean"]})
    return summary_rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Qwen scheme2 downstream ablations: length-only shared gate, length-only resolution-specific gate, and no OCR gate.")
    parser.add_argument("--csv_path", type=Path, required=True, help="Task scaling CSV path.")
    parser.add_argument("--task", type=str, choices=["retrieval", "reasoning"], required=True, help="Downstream task name.")
    parser.add_argument("--ocr_fit_dir", type=Path, default=Path("VQAExp/scaling_law/Fitting/ocr"), help="OCR fit directory containing per-resolution em_fit_summary.json files.")
    parser.add_argument("--ocr_points_csv", type=Path, default=Path("VQAExp/scaling_law/Fitting/ocr/qwen_em_fit_points_all_resolutions.csv"), help="OCR fit-points CSV containing EM responsibility r.")
    parser.add_argument("--output_dir", type=Path, default=Path("VQAExp/scaling_law/Fitting/scheme2_ablation"), help="Root output directory.")
    parser.add_argument("--resolutions", type=str, default="512,640,1024,1280", help="Comma-separated resolutions to include.")
    parser.add_argument("--holdout_layout", type=str, default=DEFAULT_HOLDOUT_LAYOUT, help="Layout held out from fitting.")
    parser.add_argument("--max_iter", type=int, default=1000, help="Maximum optimizer iterations.")
    parser.add_argument("--eps", type=float, default=DEFAULT_EPS, help="Probability clipping epsilon.")
    parser.add_argument("--reg_strength", type=float, default=0.0, help="L2 regularization strength for downstream parameters.")
    parser.add_argument("--gate_reg_strength", type=float, default=0.0, help="L2 regularization strength for frozen length-gate fitting.")
    parser.add_argument("--bin_size", type=int, default=DEFAULT_BIN_SIZE, help="Character bin size for binned MAE.")
    parser.add_argument("--text_len_scale", type=float, default=DEFAULT_TEXT_LEN_SCALE, help="Scale used in log_text_len=log(text_len/text_len_scale).")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    summary_rows = run_task(args)
    summary_path = args.output_dir / args.task / "ablation_summary.csv"
    write_csv(summary_rows, summary_path)
    LOGGER.info("Saved ablation summary to %s", summary_path)


if __name__ == "__main__":
    main()
