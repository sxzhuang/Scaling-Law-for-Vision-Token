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


def iter_layouts() -> list[str]:
    return [f"{font_size}_{line_spacing}_{letter_spacing}" for font_size in FONT_SIZES for line_spacing in LINE_SPACINGS for letter_spacing in LETTER_SPACINGS]


def parse_resolutions(resolutions_arg: str) -> list[int]:
    return [int(part.strip()) for part in resolutions_arg.split(",") if part.strip()]


def logit(value: float) -> float:
    value = min(max(float(value), 1e-6), 1.0 - 1e-6)
    return math.log(value / (1.0 - value))


def load_rows(csv_path: Path, resolutions: list[int]) -> list[dict]:
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
            row["resolution"] = resolution
            row["text_len"] = int(float(row["text_len"]))
            row["qwen_logV"] = float(row["qwen_logV"])
            row["qwen_logG"] = float(row["qwen_logG"])
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
    log_v = np.array([row["qwen_logV"] for row in rows], dtype=float)
    log_g = np.array([row["qwen_logG"] for row in rows], dtype=float)
    log_complexity = np.array([row["log_complexity"] for row in rows], dtype=float)
    y = np.array([row["correct"] for row in rows], dtype=float)
    resolution = np.array([row["resolution"] for row in rows], dtype=int)
    z = np.zeros(len(rows), dtype=float)
    for index, row in enumerate(rows):
        params = gating_params[int(row["resolution"])]
        z[index] = params["w0"] + params["a"] * row["qwen_logV"] + params["alpha"] * row["qwen_logG"]
    epsilon = expit(z)
    return {"log_v": log_v, "log_g": log_g, "log_complexity": log_complexity, "y": y, "resolution": resolution, "z": z, "epsilon": epsilon}


def predict_success(params: np.ndarray, epsilon: np.ndarray, log_complexity: np.ndarray) -> np.ndarray:
    beta0, gamma0, beta1, gamma1 = [float(value) for value in params]
    mu0 = expit(beta0 - gamma0 * log_complexity)
    mu1 = expit(beta1 - gamma1 * log_complexity)
    return (1.0 - epsilon) * mu0 + epsilon * mu1


def bernoulli_nll(params: np.ndarray, epsilon: np.ndarray, log_complexity: np.ndarray, y: np.ndarray, eps: float, reg_strength: float) -> float:
    p = np.clip(predict_success(params, epsilon, log_complexity), eps, 1.0 - eps)
    nll = -float(np.sum(y * np.log(p) + (1.0 - y) * np.log1p(-p)))
    penalty = 0.5 * float(reg_strength) * float(np.sum(params**2))
    return nll + penalty


def fit_downstream(arrays: dict[str, np.ndarray], train_mask: np.ndarray, eps: float, reg_strength: float, max_iter: int) -> tuple[np.ndarray, float]:
    y_train = arrays["y"][train_mask]
    epsilon_train = arrays["epsilon"][train_mask]
    log_complexity_train = arrays["log_complexity"][train_mask]
    mean_y = float(np.mean(y_train))
    starts = [
        np.array([logit(mean_y), 0.0, logit(mean_y), 0.0], dtype=float),
        np.array([logit(min(mean_y + 0.2, 0.95)), 0.0, logit(max(mean_y - 0.2, 0.05)), 0.0], dtype=float),
        np.array([2.0, 0.0, -2.0, 0.0], dtype=float),
        np.array([2.0, 1.0, -2.0, 1.0], dtype=float),
        np.array([0.0, 0.0, 0.0, 0.0], dtype=float),
    ]
    best_params = starts[0]
    best_nll = float("inf")
    for start in starts:
        result = minimize(bernoulli_nll, start, args=(epsilon_train, log_complexity_train, y_train, eps, reg_strength), method="L-BFGS-B", options={"maxiter": int(max_iter)})
        candidate_params = result.x if result.success and np.all(np.isfinite(result.x)) else start
        candidate_nll = bernoulli_nll(candidate_params, epsilon_train, log_complexity_train, y_train, eps, reg_strength)
        if candidate_nll < best_nll:
            best_params = candidate_params.astype(float)
            best_nll = float(candidate_nll)
    return best_params, best_nll


def compute_nll_for_mask(params: np.ndarray, arrays: dict[str, np.ndarray], mask: np.ndarray, eps: float) -> float | None:
    if not np.any(mask):
        return None
    return float(bernoulli_nll(params, arrays["epsilon"][mask], arrays["log_complexity"][mask], arrays["y"][mask], eps, 0.0))


def add_predictions(rows: list[dict], arrays: dict[str, np.ndarray], params: np.ndarray, holdout_layout: str) -> list[dict]:
    predicted_p = predict_success(params, arrays["epsilon"], arrays["log_complexity"])
    output_rows = []
    for index, row in enumerate(rows):
        out = dict(row)
        out["epsilon"] = float(arrays["epsilon"][index])
        out["gating_z"] = float(arrays["z"][index])
        out["predicted_success"] = float(predicted_p[index])
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
            }
        )
    return binned_rows


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
    fig.suptitle(f"Qwen {task} downstream scaling-law fit, resolution={resolution}", fontsize=16)
    fig.tight_layout(rect=(0, 0, 1, 0.975))
    output_file.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_file, dpi=int(dpi))
    plt.close(fig)


def summarize(rows: list[dict], arrays: dict[str, np.ndarray], params: np.ndarray, train_mask: np.ndarray, test_mask: np.ndarray, train_nll: float, eps: float, args: argparse.Namespace) -> dict:
    beta0, gamma0, beta1, gamma1 = [float(value) for value in params]
    return {
        "task": args.task,
        "csv_path": str(args.csv_path),
        "ocr_fit_dir": str(args.ocr_fit_dir),
        "holdout_layout": args.holdout_layout,
        "n_samples": len(rows),
        "n_train": int(np.sum(train_mask)),
        "n_test": int(np.sum(test_mask)),
        "train_accuracy_mean": float(np.mean(arrays["y"][train_mask])) if np.any(train_mask) else None,
        "test_accuracy_mean": float(np.mean(arrays["y"][test_mask])) if np.any(test_mask) else None,
        "params": {"beta0": beta0, "gamma0": gamma0, "beta1": beta1, "gamma1": gamma1},
        "train_nll": float(train_nll),
        "test_nll": compute_nll_for_mask(params, arrays, test_mask, eps),
        "all_nll": compute_nll_for_mask(params, arrays, np.ones(len(rows), dtype=bool), eps),
        "mean_epsilon": float(np.mean(arrays["epsilon"])),
        "eps": float(args.eps),
        "reg_strength": float(args.reg_strength),
        "bin_size": int(args.bin_size),
    }


def run_fit(args: argparse.Namespace) -> None:
    resolutions = parse_resolutions(args.resolutions)
    rows = load_rows(args.csv_path, resolutions)
    gating_params = load_ocr_gating_params(args.ocr_fit_dir, resolutions)
    arrays = rows_to_arrays(rows, gating_params)
    train_mask = np.array([row["layout"] != args.holdout_layout for row in rows], dtype=bool)
    test_mask = ~train_mask
    if not np.any(train_mask):
        raise ValueError("No training rows remain after applying holdout_layout.")
    params, train_nll = fit_downstream(arrays, train_mask, args.eps, args.reg_strength, args.max_iter)
    output_rows = add_predictions(rows, arrays, params, args.holdout_layout)
    output_dir = args.output_dir / args.task
    plots_dir = output_dir / "plots"
    output_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_rows, output_dir / "fit_points.csv")
    binned_rows = build_binned_rows(output_rows, args.bin_size)
    write_csv(binned_rows, output_dir / "fit_points_binned.csv")
    summary = summarize(rows, arrays, params, train_mask, test_mask, train_nll, args.eps, args)
    with (output_dir / "fit_summary.json").open("w", encoding="utf-8") as file_handle:
        json.dump(summary, file_handle, ensure_ascii=False, indent=2)
    for resolution in resolutions:
        plot_resolution(output_rows, binned_rows, args.task, resolution, plots_dir / f"qwen_{args.task}_scaling_fit_res_{resolution}.png", args.plot_dpi)
    LOGGER.info("Saved %s downstream fit to %s", args.task, output_dir)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fit Qwen Retrieval/Reasoning downstream Bernoulli scaling law with fixed OCR visual gating.")
    parser.add_argument("--csv_path", type=Path, required=True, help="Task scaling CSV path.")
    parser.add_argument("--task", type=str, choices=["retrieval", "reasoning"], required=True, help="Downstream task name.")
    parser.add_argument("--ocr_fit_dir", type=Path, default=Path("VQAExp/scaling_law/Fitting/ocr"), help="OCR fit directory containing per-resolution em_fit_summary.json files.")
    parser.add_argument("--output_dir", type=Path, default=Path("VQAExp/scaling_law/Fitting"), help="Root output directory.")
    parser.add_argument("--resolutions", type=str, default="512,640,1024,1280", help="Comma-separated resolutions to include.")
    parser.add_argument("--holdout_layout", type=str, default=DEFAULT_HOLDOUT_LAYOUT, help="Layout held out from fitting.")
    parser.add_argument("--max_iter", type=int, default=1000, help="Maximum optimizer iterations.")
    parser.add_argument("--eps", type=float, default=DEFAULT_EPS, help="Probability clipping epsilon for Bernoulli NLL.")
    parser.add_argument("--reg_strength", type=float, default=0.0, help="L2 regularization strength for downstream parameters.")
    parser.add_argument("--bin_size", type=int, default=DEFAULT_BIN_SIZE, help="Character bin size for plotting observed and predicted bin-level accuracy.")
    parser.add_argument("--plot_dpi", type=int, default=200, help="DPI for saved plots.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    run_fit(args)


if __name__ == "__main__":
    main()
