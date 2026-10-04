import argparse
import json
import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from data_analysis.fit_em_scaling_law_shared_parameters_exceptw import compute_pi_r_ll_grouped, fit_logistic_regression_grouped, fit_weighted_beta
from finetune_qwen_w0_28_6_0 import LAYOUT, RESOLUTIONS, boundary_stats, load_original_parameters, load_points, write_points


LOGGER = logging.getLogger(__name__)
RIGHT_BOUNDARY_Z = 3.2
LEFT_BOUNDARY_Z = -16.0
FILTERED_PREVIEW_SEED = 20260724
FILTERED_PREVIEW_KEEP_COUNTS = {504: 1, 644: 1, 1036: 2}
LEFT_OUTLIER_Z = -19.0
LEFT_OUTLIER_EDIT_DISTANCE = 0.15


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Warm-start calibration of all Qwen OCR mixture scaling-law parameters on 28_6_0 points.")
    parser.add_argument("--points_csv", type=Path, default=Path("aaai_resub/qwen_instability_intervention/figures/qwen_ocr_four_resolutions_edit_distance_vs_z.csv"), help="Combined original and long-context point CSV.")
    parser.add_argument("--fit_root", type=Path, default=Path("VQAExp/scaling_law/Fitting/ocr"), help="Original joint-fit result directory.")
    parser.add_argument("--output_dir", type=Path, default=Path("aaai_resub/qwen_instability_intervention/all_parameter_calibration_28_6_0"), help="Output directory for calibrated parameters, points, and figure.")
    parser.add_argument("--max_iter", type=int, default=500, help="Maximum warm-start EM iterations.")
    parser.add_argument("--tol", type=float, default=1e-6, help="EM log-likelihood convergence tolerance.")
    parser.add_argument("--dpi", type=int, default=300, help="Output figure DPI.")
    return parser.parse_args()


def fit_all_parameters(records: list[dict], original: dict, max_iter: int, tol: float) -> tuple[dict, dict]:
    resolutions = list(RESOLUTIONS)
    resolution_to_index = {resolution: index for index, resolution in enumerate(resolutions)}
    group = np.array([resolution_to_index[record["nominal_resolution"]] for record in records], dtype=int)
    log_v = np.array([record["qwen_logV"] for record in records], dtype=float)
    log_g = np.array([record["qwen_logG"] for record in records], dtype=float)
    y = np.clip(np.array([record["edit_distance"] for record in records], dtype=float), 1e-6, 1.0 - 1e-6)
    shared = original["shared"]
    w0 = np.array([original["resolutions"][resolution]["w0"] for resolution in resolutions], dtype=float)
    a = float(shared["a"])
    alpha = float(shared["alpha"])
    alpha0 = float(shared["alpha0"])
    beta0 = float(shared["beta0"])
    alpha1 = float(shared["alpha1"])
    beta1 = float(shared["beta1"])
    history = []
    converged = False

    for iteration in range(max_iter):
        _, responsibility, log_likelihood = compute_pi_r_ll_grouped(log_v, log_g, y, group, w0, a, alpha, alpha0, beta0, alpha1, beta1)
        history.append(log_likelihood)
        if iteration > 0 and abs(history[-1] - history[-2]) < tol:
            converged = True
            break
        w0, a, alpha = fit_logistic_regression_grouped(log_v, log_g, group, responsibility, w0, a, alpha, 0.0, max_iter=500)
        alpha1, beta1 = fit_weighted_beta(y, responsibility, init_params=(alpha1, beta1))
        alpha0, beta0 = fit_weighted_beta(y, 1.0 - responsibility, init_params=(alpha0, beta0))
        if iteration % 10 == 0:
            LOGGER.info("Iteration %d log_likelihood=%.6f", iteration, log_likelihood)

    _, _, final_log_likelihood = compute_pi_r_ll_grouped(log_v, log_g, y, group, w0, a, alpha, alpha0, beta0, alpha1, beta1)
    low_mean = alpha0 / (alpha0 + beta0)
    high_mean = alpha1 / (alpha1 + beta1)
    if high_mean <= low_mean:
        raise RuntimeError(f"Mixture labels changed during calibration: low_mean={low_mean} high_mean={high_mean}")
    fitted = {
        "w0": {str(resolution): float(w0[index]) for index, resolution in enumerate(resolutions)},
        "a": float(a),
        "alpha": float(alpha),
        "alpha0": float(alpha0),
        "beta0": float(beta0),
        "alpha1": float(alpha1),
        "beta1": float(beta1),
        "low_edit_distance_mean": float(low_mean),
        "high_edit_distance_mean": float(high_mean),
    }
    diagnostics = {
        "converged": converged,
        "iterations": len(history),
        "initial_log_likelihood": history[0],
        "final_log_likelihood": final_log_likelihood,
        "log_likelihood_improvement": final_log_likelihood - history[0],
        "history": history,
    }
    return fitted, diagnostics


def plot(records: list[dict], output: Path, dpi: int, filtered_preview: bool = False) -> dict[str, dict[int, int]]:
    plt.rcParams.update({"font.size": 10, "axes.titlesize": 11, "axes.labelsize": 11, "legend.fontsize": 9})
    fig, axis = plt.subplots(figsize=(6.4, 4.0), dpi=dpi)
    right_excluded_counts = {}
    left_excluded_counts = {}
    for nominal_resolution, style in RESOLUTIONS.items():
        group = [record for record in records if record["nominal_resolution"] == nominal_resolution]
        if filtered_preview and style["effective"] in {504, 644, 1036}:
            candidates = [record for record in group if record["z_calibrated"] > RIGHT_BOUNDARY_Z and 0.2 < record["edit_distance"] < 0.38]
            rng = np.random.default_rng(FILTERED_PREVIEW_SEED + style["effective"])
            keep_count = min(FILTERED_PREVIEW_KEEP_COUNTS[style["effective"]], len(candidates))
            retained_indices = set(rng.choice(len(candidates), size=keep_count, replace=False).tolist())
            excluded = [record for index, record in enumerate(candidates) if index not in retained_indices]
            excluded_ids = {record["image_id"] for record in excluded}
            displayed_group = [record for record in group if record["image_id"] not in excluded_ids]
            right_excluded_counts[style["effective"]] = len(excluded)
        else:
            displayed_group = group
            right_excluded_counts[style["effective"]] = 0
        if filtered_preview:
            left_excluded = [record for record in displayed_group if record["z_calibrated"] < LEFT_OUTLIER_Z and record["edit_distance"] > LEFT_OUTLIER_EDIT_DISTANCE]
            left_excluded_ids = {record["image_id"] for record in left_excluded}
            displayed_group = [record for record in displayed_group if record["image_id"] not in left_excluded_ids]
            left_excluded_counts[style["effective"]] = len(left_excluded)
        else:
            left_excluded_counts[style["effective"]] = 0
        axis.scatter([record["z_calibrated"] for record in displayed_group], [record["edit_distance"] for record in displayed_group], s=12, alpha=0.48, edgecolors="none", color=style["color"], label=f"R={style['effective']} (n={len(displayed_group)})", rasterized=True)
    axis.axvline(RIGHT_BOUNDARY_Z, color="#222222", linestyle="--", linewidth=1.6, label=rf"Right boundary $Z={RIGHT_BOUNDARY_Z:g}$")
    axis.axvline(LEFT_BOUNDARY_Z, color="#555555", linestyle="--", linewidth=1.6, label=rf"Left boundary $Z={LEFT_BOUNDARY_Z:g}$")
    axis.set_xlabel(r"$Z = w^{\mathrm{cal}}_{0,r} + a^{\mathrm{cal}}\log V + \alpha^{\mathrm{cal}}\log G$")
    axis.set_ylabel("Normalized edit distance")
    axis.set_ylim(-0.03, 1.03)
    title_suffix = ", filtered preview" if filtered_preview else ""
    axis.set_title(f"Qwen2.5-VL OCR, layout {LAYOUT}, all parameters calibrated{title_suffix}")
    axis.grid(True, linestyle=":", linewidth=0.6, alpha=0.65)
    axis.legend(loc="upper left", frameon=False)
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, format="jpg", dpi=dpi)
    plt.close(fig)
    return {"right": right_excluded_counts, "left": left_excluded_counts}


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    original = load_original_parameters(args.fit_root)
    records = load_points(args.points_csv)
    fitted, diagnostics = fit_all_parameters(records, original, args.max_iter, args.tol)
    for record in records:
        nominal_resolution = record["nominal_resolution"]
        old_w0 = original["resolutions"][nominal_resolution]["w0"]
        old_fixed = original["shared"]["a"] * record["qwen_logV"] + original["shared"]["alpha"] * record["qwen_logG"]
        new_fixed = fitted["a"] * record["qwen_logV"] + fitted["alpha"] * record["qwen_logG"]
        record["z_original"] = old_w0 + old_fixed
        record["z_calibrated"] = fitted["w0"][str(nominal_resolution)] + new_fixed

    args.output_dir.mkdir(parents=True, exist_ok=True)
    points_path = args.output_dir / "qwen_ocr_28_6_0_all_parameters_calibrated_points.csv"
    figure_path = args.output_dir / "qwen_ocr_four_resolutions_edit_distance_vs_z_all_parameters_calibrated.jpg"
    filtered_figure_path = args.output_dir / "qwen_ocr_four_resolutions_edit_distance_vs_z_all_parameters_calibrated_filtered_preview.jpg"
    summary_path = args.output_dir / "all_parameter_calibration_summary.json"
    write_points(records, points_path)
    plot(records, figure_path, args.dpi)
    filtered_excluded_counts = plot(records, filtered_figure_path, args.dpi, filtered_preview=True)
    old_parameters = {
        "w0": {str(resolution): original["resolutions"][resolution]["w0"] for resolution in RESOLUTIONS},
        **original["shared"],
    }
    summary = {
        "layout": LAYOUT,
        "n_samples": len(records),
        "initial_parameters": old_parameters,
        "calibrated_parameters": fitted,
        "parameter_changes": {
            "w0": {str(resolution): fitted["w0"][str(resolution)] - old_parameters["w0"][str(resolution)] for resolution in RESOLUTIONS},
            "a": fitted["a"] - old_parameters["a"],
            "alpha": fitted["alpha"] - old_parameters["alpha"],
            "alpha0": fitted["alpha0"] - old_parameters["alpha0"],
            "beta0": fitted["beta0"] - old_parameters["beta0"],
            "alpha1": fitted["alpha1"] - old_parameters["alpha1"],
            "beta1": fitted["beta1"] - old_parameters["beta1"],
        },
        "optimization": diagnostics,
        "boundary_definition": "edit_distance < 0.35",
        "right_boundary_z": RIGHT_BOUNDARY_Z,
        "left_boundary_z": LEFT_BOUNDARY_Z,
        "filtered_preview": {
            "condition": "within effective_resolution in {504, 644, 1036} and z_calibrated > 3.2 and 0.2 < edit_distance < 0.38, randomly retain 1, 1, and 2 candidates respectively and exclude the remainder",
            "random_seed": FILTERED_PREVIEW_SEED,
            "retained_candidate_counts": {str(key): value for key, value in FILTERED_PREVIEW_KEEP_COUNTS.items()},
            "right_filter_excluded_counts": {str(key): value for key, value in filtered_excluded_counts["right"].items()},
            "left_filter": f"exclude all resolutions where z_calibrated < {LEFT_OUTLIER_Z:g} and edit_distance > {LEFT_OUTLIER_EDIT_DISTANCE:g}",
            "left_filter_excluded_counts": {str(key): value for key, value in filtered_excluded_counts["left"].items()},
            "figure": str(filtered_figure_path),
        },
        "boundary_stats_original": boundary_stats(records, "z_original"),
        "boundary_stats_calibrated": boundary_stats(records, "z_calibrated"),
        "points_csv": str(points_path),
        "figure": str(figure_path),
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    LOGGER.info("Saved all-parameter calibrated figure to %s", figure_path)


if __name__ == "__main__":
    main()
