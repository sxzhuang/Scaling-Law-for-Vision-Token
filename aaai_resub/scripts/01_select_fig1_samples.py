#!/usr/bin/env python3
"""Select Figure 1 samples left of the EM-derived collapse boundary.

The script deliberately uses the implementation variable from
``fit_em_scaling_law_shared_parameters_exceptw.py``:

    V_code = ((character_width + letter_spacing) * text_len)
             / (character_height + line_spacing)

It writes selected metadata in the repository's existing list-of-records
format, so ``build_image_square_font_line_letter.py`` can render it directly.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import shutil
from pathlib import Path


logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_ROOT = REPO_ROOT / "aaai_resub" / "selected_samples"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Select Figure 1 Stable and Instability samples using the fitted capacity pressure.")
    parser.add_argument("--experiment-config", type=Path, default=REPO_ROOT / "aaai_resub/configs/experiment.json", help="Experiment configuration JSON.")
    parser.add_argument("--sources-config", type=Path, default=REPO_ROOT / "aaai_resub/configs/fig1_sources.json", help="Figure 1 source-path JSON.")
    parser.add_argument("--domains", nargs="+", default=None, help="Optional subset of domains to select.")
    parser.add_argument("--resolutions", nargs="+", type=int, default=None, help="Optional subset of resolutions to select.")
    parser.add_argument("--overwrite", action="store_true", help="Replace existing selection directories for the requested domain-resolution pairs.")
    return parser.parse_args()


def load_json(path: Path) -> dict | list:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def resolve_repo_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else REPO_ROOT / path


def load_metric_map(path: Path) -> dict[str, float]:
    records = load_json(path)
    if not isinstance(records, list):
        raise ValueError(f"Metrics JSON must contain a list: {path}")
    metric_map: dict[str, float] = {}
    for record in records:
        if not isinstance(record, dict):
            continue
        sample_id = record.get("id")
        edit_dist = record.get("edit_dist")
        if sample_id is None or edit_dist is None:
            continue
        metric_map[str(sample_id)] = float(edit_dist)
    if not metric_map:
        raise ValueError(f"No per-sample edit distances found in {path}")
    return metric_map


def load_model_summary(path: Path) -> dict[str, float]:
    summary = load_json(path)
    if not isinstance(summary, dict):
        raise ValueError(f"Model summary must be a JSON object: {path}")
    gating = summary.get("gating_params", {})
    beta_params = summary.get("beta_params", {})
    low_ed = beta_params.get("low_ed", {})
    high_ed = beta_params.get("high_ed", {})
    required = {
        "bias": gating.get("w0"),
        "omega_v": gating.get("a"),
        "omega_g": gating.get("alpha"),
        "mean_low_ed": low_ed.get("mean"),
        "mean_high_ed": high_ed.get("mean"),
    }
    missing = [name for name, value in required.items() if value is None]
    if missing:
        raise ValueError(f"Missing {', '.join(missing)} in {path}")
    return {name: float(value) for name, value in required.items()}


def pressure_boundary(expected_ed: float, mean_low_ed: float, mean_high_ed: float) -> float:
    if not mean_low_ed < expected_ed < mean_high_ed:
        raise ValueError("Expected-ED threshold must lie strictly between the low-ED and high-ED component means.")
    collapse_probability = (expected_ed - mean_low_ed) / (mean_high_ed - mean_low_ed)
    return math.log(collapse_probability / (1.0 - collapse_probability))


def phase_for_z(z_value: float, stable_boundary: float, collapse_boundary: float) -> str:
    if z_value < stable_boundary:
        return "stable"
    if z_value < collapse_boundary:
        return "instability"
    return "collapse"


def selected_for_phase(phase: str, selection_config: dict) -> bool:
    if phase == "stable":
        return bool(selection_config["include_stable"])
    if phase == "instability":
        return bool(selection_config["include_instability"])
    return not bool(selection_config["exclude_collapse"])


def write_json(path: Path, value: dict | list) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def replace_output_dir(output_dir: Path, overwrite: bool) -> None:
    if output_dir.exists():
        if not overwrite:
            raise FileExistsError(f"Selection output already exists: {output_dir}. Pass --overwrite to replace it.")
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)


def select_domain_resolution(
    domain: str,
    resolution: int,
    source_config: dict,
    experiment_config: dict,
    overwrite: bool,
) -> dict[str, int]:
    baseline = experiment_config["baseline_layout"]
    selection_config = experiment_config["selection"]
    metadata_path = resolve_repo_path(source_config["metadata"])
    evaluation_root = resolve_repo_path(source_config["evaluation_root"])
    metric_path = evaluation_root / str(resolution) / "dpsk_eval_metric.json"
    summary_root = resolve_repo_path(experiment_config["model_summary_root"])
    summary_path = summary_root / str(resolution) / "em_fit_summary.json"

    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata not found for {domain}: {metadata_path}")
    if not metric_path.exists():
        raise FileNotFoundError(f"Baseline metrics not found for {domain}, R={resolution}: {metric_path}")
    if not summary_path.exists():
        raise FileNotFoundError(f"Fitted model summary not found for R={resolution}: {summary_path}")

    metadata_records = load_json(metadata_path)
    if not isinstance(metadata_records, list):
        raise ValueError(f"Metadata must be a list: {metadata_path}")
    metric_map = load_metric_map(metric_path)
    model = load_model_summary(summary_path)
    stable_boundary = pressure_boundary(
        float(selection_config["stable_expected_edit_distance"]),
        model["mean_low_ed"],
        model["mean_high_ed"],
    )
    collapse_boundary = pressure_boundary(
        float(selection_config["collapse_expected_edit_distance"]),
        model["mean_low_ed"],
        model["mean_high_ed"],
    )

    char_width = float(baseline["character_width"])
    char_height = float(baseline["character_height"])
    letter_spacing = float(baseline["letter_spacing"])
    line_spacing = float(baseline["line_spacing"])
    layout_ratio = (char_width + letter_spacing) / (char_height + line_spacing)
    if layout_ratio <= 0.0:
        raise ValueError("Baseline layout must have a positive character-width to line-height ratio.")

    vision_token_num = (float(resolution) / 64.0) ** 2
    rows: list[dict] = []
    selected_metadata: list[dict] = []
    missing_metrics: list[str] = []

    for record in metadata_records:
        if not isinstance(record, dict):
            continue
        image_name = str(record.get("image", "")).strip()
        text = str(record.get("gt", "")).strip()
        if not image_name or not text:
            continue
        image_id = Path(image_name).stem
        text_len = int(record.get("text_len") or len(text))
        v_code = layout_ratio * text_len
        g_value = text_len / vision_token_num
        z_value = model["bias"] + model["omega_v"] * math.log(v_code) + model["omega_g"] * math.log(g_value)
        phase = phase_for_z(z_value, stable_boundary, collapse_boundary)
        selected = selected_for_phase(phase, selection_config)
        baseline_edit_distance = metric_map.get(image_id)
        if baseline_edit_distance is None:
            missing_metrics.append(image_id)

        row = {
            "domain": domain,
            "resolution": resolution,
            "image_id": image_id,
            "image": image_name,
            "text_len": text_len,
            "baseline_edit_distance": baseline_edit_distance,
            "font_size": int(baseline["font_size"]),
            "line_spacing": int(baseline["line_spacing"]),
            "letter_spacing": int(baseline["letter_spacing"]),
            "character_width": char_width,
            "character_height": char_height,
            "vision_token_num": vision_token_num,
            "V_code": v_code,
            "G": g_value,
            "logV": math.log(v_code),
            "logG": math.log(g_value),
            "bias": model["bias"],
            "omega_V": model["omega_v"],
            "omega_G": model["omega_g"],
            "Z": z_value,
            "stable_boundary": stable_boundary,
            "collapse_boundary": collapse_boundary,
            "phase": phase,
            "selected": selected,
        }
        rows.append(row)
        if selected:
            selected_record = dict(record)
            selected_record.update(
                {
                    "domain": domain,
                    "resolution": resolution,
                    "baseline_edit_distance": baseline_edit_distance,
                    "vision_token_num": vision_token_num,
                    "V_code": v_code,
                    "G": g_value,
                    "Z_baseline": z_value,
                    "phase_baseline": phase,
                    "stable_boundary": stable_boundary,
                    "collapse_boundary": collapse_boundary,
                    "selected": True,
                }
            )
            selected_metadata.append(selected_record)

    if missing_metrics:
        logger.warning("%s, R=%d: %d metadata records have no matching baseline metric.", domain, resolution, len(missing_metrics))

    output_dir = OUTPUT_ROOT / domain / str(resolution)
    replace_output_dir(output_dir, overwrite)
    selected_ids = [str(record["image_id"]) for record in rows if record["selected"]]
    write_json(output_dir / "selected_ids.json", selected_ids)
    write_json(output_dir / "selected_metadata.json", selected_metadata)
    fieldnames = list(rows[0].keys()) if rows else []
    with (output_dir / "all_samples_with_z.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    phase_counts = {phase: sum(row["phase"] == phase for row in rows) for phase in ("stable", "instability", "collapse")}
    summary = {
        "domain": domain,
        "resolution": resolution,
        "source_metadata": str(metadata_path.relative_to(REPO_ROOT)),
        "source_metrics": str(metric_path.relative_to(REPO_ROOT)),
        "model_summary": str(summary_path.relative_to(REPO_ROOT)),
        "total_samples": len(rows),
        "stable_samples": phase_counts["stable"],
        "instability_samples": phase_counts["instability"],
        "collapse_samples": phase_counts["collapse"],
        "selected_samples": len(selected_metadata),
        "records_without_baseline_metric": len(missing_metrics),
        "stable_boundary": stable_boundary,
        "collapse_boundary": collapse_boundary,
        "gating_parameters": {
            "bias": model["bias"],
            "omega_V": model["omega_v"],
            "omega_G": model["omega_g"],
        },
        "beta_component_means": {
            "low_ed": model["mean_low_ed"],
            "high_ed": model["mean_high_ed"],
        },
        "baseline_layout": baseline,
    }
    write_json(output_dir / "selection_summary.json", summary)
    logger.info(
        "%s, R=%d: selected %d/%d samples (stable=%d, instability=%d, collapse=%d).",
        domain,
        resolution,
        len(selected_metadata),
        len(rows),
        phase_counts["stable"],
        phase_counts["instability"],
        phase_counts["collapse"],
    )
    return {"total": len(rows), "selected": len(selected_metadata)}


def main() -> None:
    args = parse_args()
    experiment_config = load_json(args.experiment_config)
    sources_config = load_json(args.sources_config)
    if not isinstance(experiment_config, dict) or not isinstance(sources_config, dict):
        raise ValueError("Experiment and source configuration files must be JSON objects.")
    domains = args.domains or list(sources_config.keys())
    resolutions = args.resolutions or [int(value) for value in experiment_config["resolutions"]]
    unknown_domains = [domain for domain in domains if domain not in sources_config]
    if unknown_domains:
        raise ValueError(f"Unknown domains: {', '.join(unknown_domains)}")

    total_samples = 0
    total_selected = 0
    for domain in domains:
        source_config = sources_config[domain]
        if not isinstance(source_config, dict):
            raise ValueError(f"Source configuration for {domain} must be an object.")
        for resolution in resolutions:
            result = select_domain_resolution(domain, resolution, source_config, experiment_config, args.overwrite)
            total_samples += result["total"]
            total_selected += result["selected"]
    logger.info("Selection complete: %d/%d samples selected across %d domain-resolution groups.", total_selected, total_samples, len(domains) * len(resolutions))


if __name__ == "__main__":
    main()
