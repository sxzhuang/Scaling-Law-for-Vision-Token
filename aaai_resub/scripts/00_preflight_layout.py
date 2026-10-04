#!/usr/bin/env python3
"""Verify the law-guided layout using the renderer's actual font metrics."""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from utils.generate_square_images import average_character_width, load_font


logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Measure actual renderer metrics and verify the intervention capacity-pressure shift.")
    parser.add_argument("--experiment-config", type=Path, default=REPO_ROOT / "aaai_resub/configs/experiment.json", help="Experiment configuration JSON.")
    parser.add_argument("--output", type=Path, default=REPO_ROOT / "aaai_resub/analysis/layout_preflight.json", help="Preflight result JSON.")
    parser.add_argument("--overwrite", action="store_true", help="Replace an existing preflight result.")
    return parser.parse_args()


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as file:
        value = json.load(file)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def resolve_repo_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else REPO_ROOT / path


def measure_font(font_size: int, font_path: Path | None) -> dict[str, float]:
    font = load_font(font_size, font_path)
    bbox = font.getbbox("Hg")
    return {
        "character_width": float(average_character_width(font)),
        "character_height": float(bbox[3] - bbox[1]),
    }


def pressure_boundary(expected_ed: float, mean_low_ed: float, mean_high_ed: float) -> float:
    collapse_probability = (expected_ed - mean_low_ed) / (mean_high_ed - mean_low_ed)
    return math.log(collapse_probability / (1.0 - collapse_probability))


def main() -> None:
    args = parse_args()
    config = load_json(args.experiment_config)
    baseline = config["baseline_layout"]
    intervention = config["intervention_layout"]
    rendering = config["rendering"]
    font_path_value = rendering.get("font_path")
    font_path = None if font_path_value is None else resolve_repo_path(str(font_path_value))
    if font_path is not None and not font_path.exists():
        raise FileNotFoundError(f"Configured font path does not exist: {font_path}")

    measured_baseline = measure_font(int(baseline["font_size"]), font_path)
    measured_intervention = measure_font(int(intervention["font_size"]), font_path)
    baseline_ratio = (
        measured_baseline["character_width"] + float(baseline["letter_spacing"])
    ) / (
        measured_baseline["character_height"] + float(baseline["line_spacing"])
    )
    intervention_ratio = (
        measured_intervention["character_width"] + float(intervention["letter_spacing"])
    ) / (
        measured_intervention["character_height"] + float(intervention["line_spacing"])
    )
    if baseline_ratio <= 0.0 or intervention_ratio <= 0.0:
        raise ValueError("Measured layout ratios must be positive.")

    summary_root = resolve_repo_path(str(config["model_summary_root"]))
    checks: list[dict] = []
    for resolution in config["resolutions"]:
        summary_path = summary_root / str(resolution) / "em_fit_summary.json"
        summary = load_json(summary_path)
        gating = summary["gating_params"]
        beta_params = summary["beta_params"]
        omega_v = float(gating["a"])
        mean_low_ed = float(beta_params["low_ed"]["mean"])
        mean_high_ed = float(beta_params["high_ed"]["mean"])
        stable_boundary = pressure_boundary(float(config["selection"]["stable_expected_edit_distance"]), mean_low_ed, mean_high_ed)
        collapse_boundary = pressure_boundary(float(config["selection"]["collapse_expected_edit_distance"]), mean_low_ed, mean_high_ed)
        delta_z = omega_v * math.log(intervention_ratio / baseline_ratio)
        mapped_upper_boundary = collapse_boundary + delta_z
        required_delta_z = stable_boundary - collapse_boundary
        checks.append(
            {
                "resolution": int(resolution),
                "omega_V": omega_v,
                "stable_boundary": stable_boundary,
                "collapse_boundary": collapse_boundary,
                "required_delta_Z": required_delta_z,
                "actual_delta_Z": delta_z,
                "mapped_instability_upper_boundary": mapped_upper_boundary,
                "stable_margin": stable_boundary - mapped_upper_boundary,
                "passes": mapped_upper_boundary < stable_boundary,
            }
        )

    result = {
        "renderer_font": "DejaVuSans.ttf" if font_path is None else str(font_path),
        "baseline_layout": baseline,
        "intervention_layout": intervention,
        "measured_baseline_metrics": measured_baseline,
        "measured_intervention_metrics": measured_intervention,
        "configured_intervention_metrics": {
            "character_width": float(intervention["character_width"]),
            "character_height": float(intervention["character_height"]),
        },
        "baseline_layout_ratio": baseline_ratio,
        "intervention_layout_ratio": intervention_ratio,
        "checks": checks,
        "passes_all_resolutions": all(check["passes"] for check in checks),
    }
    if args.output.exists() and not args.overwrite:
        raise FileExistsError(f"Preflight output already exists: {args.output}. Pass --overwrite to replace it.")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("Measured baseline metrics: %s", measured_baseline)
    logger.info("Measured intervention metrics: %s", measured_intervention)
    for check in checks:
        logger.info(
            "R=%d: delta_Z=%.6f, mapped upper=%.6f, stable boundary=%.6f, pass=%s",
            check["resolution"],
            check["actual_delta_Z"],
            check["mapped_instability_upper_boundary"],
            check["stable_boundary"],
            check["passes"],
        )
    if not result["passes_all_resolutions"]:
        raise RuntimeError("The measured layout does not move every Instability upper boundary into Stable.")


if __name__ == "__main__":
    main()
