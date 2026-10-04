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
from scipy.special import betaln, expit


LOGGER = logging.getLogger(__name__)
LAYOUT = "28_6_0"
LOW_EDIT_DISTANCE = 0.35
RESOLUTIONS = {
    512: {"effective": 504, "color": "#1F77B4"},
    640: {"effective": 644, "color": "#D95F02"},
    1024: {"effective": 1036, "color": "#2CA02C"},
    1280: {"effective": 1288, "color": "#9467BD"},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Calibrate Qwen OCR resolution intercepts on all 28_6_0 points while freezing shared parameters.")
    parser.add_argument("--points_csv", type=Path, default=Path("aaai_resub/qwen_instability_intervention/figures/qwen_ocr_four_resolutions_edit_distance_vs_z.csv"), help="Combined original and long-context point CSV.")
    parser.add_argument("--fit_root", type=Path, default=Path("VQAExp/scaling_law/Fitting/ocr"), help="Original joint-fit result directory.")
    parser.add_argument("--output_dir", type=Path, default=Path("aaai_resub/qwen_instability_intervention/w0_calibration_28_6_0"), help="Output directory for calibrated parameters, points, and figure.")
    parser.add_argument("--dpi", type=int, default=300, help="Output figure DPI.")
    return parser.parse_args()


def load_original_parameters(fit_root: Path) -> dict:
    parameters = {}
    shared = None
    for nominal_resolution in RESOLUTIONS:
        summary_path = fit_root / str(nominal_resolution) / "em_fit_summary.json"
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        gating = summary["gating_params"]
        beta = summary["beta_params"]
        current_shared = {
            "a": float(gating["a"]),
            "alpha": float(gating["alpha"]),
            "alpha0": float(beta["low_ed"]["alpha"]),
            "beta0": float(beta["low_ed"]["beta"]),
            "alpha1": float(beta["high_ed"]["alpha"]),
            "beta1": float(beta["high_ed"]["beta"]),
        }
        if shared is None:
            shared = current_shared
        elif any(not math.isclose(current_shared[key], shared[key], rel_tol=0.0, abs_tol=1e-12) for key in shared):
            raise ValueError("Expected a, alpha, and beta parameters to be shared across resolutions.")
        parameters[nominal_resolution] = {"w0": float(gating["w0"]), "summary": str(summary_path)}
    return {"shared": shared, "resolutions": parameters}


def load_points(path: Path) -> list[dict]:
    records = []
    with path.open("r", newline="", encoding="utf-8") as file_handle:
        for row in csv.DictReader(file_handle):
            nominal_resolution = int(row["nominal_resolution"])
            if nominal_resolution not in RESOLUTIONS:
                continue
            records.append({
                "image_id": str(row["image_id"]),
                "nominal_resolution": nominal_resolution,
                "effective_resolution": int(row["effective_resolution"]),
                "text_len": int(row["text_len"]),
                "edit_distance": float(row["edit_distance"]),
                "qwen_logV": float(row["qwen_logV"]),
                "qwen_logG": float(row["qwen_logG"]),
                "source": str(row["source"]),
            })
    expected_counts = {512: 600, 640: 600, 1024: 780, 1280: 1060}
    actual_counts = {resolution: sum(record["nominal_resolution"] == resolution for record in records) for resolution in RESOLUTIONS}
    if actual_counts != expected_counts:
        raise ValueError(f"Unexpected point counts: expected={expected_counts} actual={actual_counts}")
    return records


def fit_intercepts(records: list[dict], original: dict) -> tuple[np.ndarray, dict]:
    resolutions = list(RESOLUTIONS)
    resolution_to_index = {resolution: index for index, resolution in enumerate(resolutions)}
    group = np.array([resolution_to_index[record["nominal_resolution"]] for record in records], dtype=int)
    log_v = np.array([record["qwen_logV"] for record in records], dtype=float)
    log_g = np.array([record["qwen_logG"] for record in records], dtype=float)
    y = np.clip(np.array([record["edit_distance"] for record in records], dtype=float), 1e-6, 1.0 - 1e-6)
    shared = original["shared"]
    fixed_term = shared["a"] * log_v + shared["alpha"] * log_g
    log_pdf0 = (shared["alpha0"] - 1.0) * np.log(y) + (shared["beta0"] - 1.0) * np.log1p(-y) - betaln(shared["alpha0"], shared["beta0"])
    log_pdf1 = (shared["alpha1"] - 1.0) * np.log(y) + (shared["beta1"] - 1.0) * np.log1p(-y) - betaln(shared["alpha1"], shared["beta1"])
    initial = np.array([original["resolutions"][resolution]["w0"] for resolution in resolutions], dtype=float)

    def objective(w0: np.ndarray) -> tuple[float, np.ndarray]:
        z = w0[group] + fixed_term
        log_pi = -np.logaddexp(0.0, -z)
        log_one_minus_pi = -np.logaddexp(0.0, z)
        log_mix = np.logaddexp(log_one_minus_pi + log_pdf0, log_pi + log_pdf1)
        responsibility = np.exp(log_pi + log_pdf1 - log_mix)
        probability = expit(z)
        gradient = np.zeros_like(w0)
        for index in range(len(w0)):
            mask = group == index
            gradient[index] = np.sum(probability[mask] - responsibility[mask])
        return -float(np.sum(log_mix)), gradient

    initial_nll, _ = objective(initial)
    result = minimize(objective, initial, method="L-BFGS-B", jac=True, options={"maxiter": 1000, "ftol": 1e-12, "gtol": 1e-8})
    if not result.success:
        raise RuntimeError(f"Intercept calibration failed: {result.message}")
    final_nll, _ = objective(result.x)
    diagnostics = {
        "success": bool(result.success),
        "message": str(result.message),
        "iterations": int(result.nit),
        "initial_negative_log_likelihood": initial_nll,
        "final_negative_log_likelihood": final_nll,
        "negative_log_likelihood_improvement": initial_nll - final_nll,
    }
    return result.x, diagnostics


def boundary_stats(records: list[dict], z_field: str) -> dict:
    stats = {}
    for nominal_resolution, style in RESOLUTIONS.items():
        group = [record for record in records if record["nominal_resolution"] == nominal_resolution]
        low_z = sorted(record[z_field] for record in group if record["edit_distance"] < LOW_EDIT_DISTANCE)
        beyond_five = [record for record in group if record[z_field] >= 5.0]
        low_beyond_five = [record for record in beyond_five if record["edit_distance"] < LOW_EDIT_DISTANCE]
        stats[str(nominal_resolution)] = {
            "effective_resolution": style["effective"],
            "n_samples": len(group),
            "low_edit_distance_samples": len(low_z),
            "low_z_p95": float(np.quantile(low_z, 0.95)),
            "low_z_p99": float(np.quantile(low_z, 0.99)),
            "low_z_max": max(low_z),
            "samples_z_ge_5": len(beyond_five),
            "low_samples_z_ge_5": len(low_beyond_five),
            "low_fraction_z_ge_5": len(low_beyond_five) / len(beyond_five) if beyond_five else None,
        }
    return stats


def write_points(records: list[dict], path: Path) -> None:
    fieldnames = ["image_id", "nominal_resolution", "effective_resolution", "text_len", "edit_distance", "qwen_logV", "qwen_logG", "source", "z_original", "z_calibrated"]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file_handle:
        writer = csv.DictWriter(file_handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)


def plot(records: list[dict], output: Path, dpi: int) -> None:
    plt.rcParams.update({"font.size": 10, "axes.titlesize": 11, "axes.labelsize": 11, "legend.fontsize": 9})
    fig, axis = plt.subplots(figsize=(6.4, 4.0), dpi=dpi)
    for nominal_resolution, style in RESOLUTIONS.items():
        group = [record for record in records if record["nominal_resolution"] == nominal_resolution]
        axis.scatter([record["z_calibrated"] for record in group], [record["edit_distance"] for record in group], s=12, alpha=0.48, edgecolors="none", color=style["color"], label=f"R={style['effective']} (n={len(group)})", rasterized=True)
    axis.set_xlabel(r"$Z = w^{\mathrm{cal}}_{0,r} + a\log V + \alpha\log G$")
    axis.set_ylabel("Normalized edit distance")
    axis.set_ylim(-0.03, 1.03)
    axis.set_title(f"Qwen2.5-VL OCR, layout {LAYOUT}, calibrated $w_0$")
    axis.grid(True, linestyle=":", linewidth=0.6, alpha=0.65)
    axis.legend(loc="upper left", frameon=False)
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, format="jpg", dpi=dpi)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    original = load_original_parameters(args.fit_root)
    records = load_points(args.points_csv)
    calibrated_w0, diagnostics = fit_intercepts(records, original)
    for index, nominal_resolution in enumerate(RESOLUTIONS):
        original_w0 = original["resolutions"][nominal_resolution]["w0"]
        calibrated = float(calibrated_w0[index])
        for record in records:
            if record["nominal_resolution"] != nominal_resolution:
                continue
            fixed = original["shared"]["a"] * record["qwen_logV"] + original["shared"]["alpha"] * record["qwen_logG"]
            record["z_original"] = original_w0 + fixed
            record["z_calibrated"] = calibrated + fixed

    args.output_dir.mkdir(parents=True, exist_ok=True)
    points_path = args.output_dir / "qwen_ocr_28_6_0_w0_calibrated_points.csv"
    figure_path = args.output_dir / "qwen_ocr_four_resolutions_edit_distance_vs_z_w0_calibrated.jpg"
    summary_path = args.output_dir / "w0_calibration_summary.json"
    write_points(records, points_path)
    plot(records, figure_path, args.dpi)
    summary = {
        "layout": LAYOUT,
        "n_samples": len(records),
        "frozen_shared_parameters": original["shared"],
        "optimization": diagnostics,
        "resolutions": {},
        "boundary_definition": f"edit_distance < {LOW_EDIT_DISTANCE}",
        "boundary_stats_original": boundary_stats(records, "z_original"),
        "boundary_stats_calibrated": boundary_stats(records, "z_calibrated"),
        "points_csv": str(points_path),
        "figure": str(figure_path),
    }
    for index, nominal_resolution in enumerate(RESOLUTIONS):
        old_w0 = original["resolutions"][nominal_resolution]["w0"]
        new_w0 = float(calibrated_w0[index])
        summary["resolutions"][str(nominal_resolution)] = {"effective_resolution": RESOLUTIONS[nominal_resolution]["effective"], "old_w0": old_w0, "calibrated_w0": new_w0, "delta_w0": new_w0 - old_w0}
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    LOGGER.info("Saved calibrated figure to %s", figure_path)


if __name__ == "__main__":
    main()
