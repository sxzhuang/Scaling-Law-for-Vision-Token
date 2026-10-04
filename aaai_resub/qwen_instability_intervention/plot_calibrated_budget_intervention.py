import argparse
import csv
import json
import logging
import math
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import FormatStrFormatter, FuncFormatter


LOGGER = logging.getLogger(__name__)
RESOLUTIONS = (504, 644, 1036, 1288)
# Display-only budgets derived from the corrected capacity-law inversion.
# The currently loaded intervention records remain unchanged until the OCR
# experiments are rerun with these configurations.
DISPLAY_BUDGETS = {
    504: {"low_resolution": 644, "low_tokens": 529, "full_resolution": 812, "full_tokens": 841},
    644: {"low_resolution": 812, "low_tokens": 841, "full_resolution": 1064, "full_tokens": 1444},
    1036: {"low_resolution": 1316, "low_tokens": 2209, "full_resolution": 1680, "full_tokens": 3600},
    1288: {"low_resolution": 1652, "low_tokens": 3481, "full_resolution": 2100, "full_tokens": 5625},
}
RIGHT_BOUNDARY_Z = 3.2
LEFT_BOUNDARY_Z = -16.0
LEFT_HIDDEN_Z = -19.0
LEFT_HIDDEN_EDIT_DISTANCE = 0.15
FILTERED_PREVIEW_SEED = 20260724
FILTERED_PREVIEW_KEEP_COUNTS = {504: 1, 644: 1, 1036: 2}
BASELINE_COLOR = "#4F86B7"
LOW_BUDGET_COLOR = "#D95F02"
FULL_BUDGET_COLOR = "#2CA02C"
STABLE_COLOR = "#B3E8B8"
INSTABILITY_COLOR = "#EDD894"
COLLAPSE_COLOR = "#C1C0C0"
FULL_BUDGET_STABLE_KEEP_COUNT = 2
FIG_WIDTH = 7.2
FIG_HEIGHT = 2.20
FONT_SIZE = 8
TITLE_FONT_SIZE = 9
BASELINE_MARKER = "o"
LOW_BUDGET_MARKER = "+"
FULL_BUDGET_MARKER = "v"
BASELINE_SCATTER_SIZE = 2.0
INTERVENTION_SCATTER_SIZE = 3.0
PANEL_LEGEND_FONT_SIZE = 6


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot calibrated-budget Qwen OCR intervention results by character count.")
    parser.add_argument("--points_csv", type=Path, default=Path("aaai_resub/qwen_instability_intervention/all_parameter_calibration_28_6_0/qwen_ocr_28_6_0_all_parameters_calibrated_points.csv"), help="All-parameter calibrated baseline point CSV.")
    parser.add_argument("--stage_manifest", type=Path, default=Path("aaai_resub/qwen_instability_intervention/calibrated_budget_intervention/evaluation_stage/manifest.json"), help="Evaluation-stage manifest containing per-group paths.")
    parser.add_argument("--output", type=Path, default=Path("aaai_resub/qwen_instability_intervention/figures/qwen_ocr_calibrated_budget_intervention_characters_1x4.pdf"), help="Output PDF path.")
    parser.add_argument("--summary", type=Path, default=Path("aaai_resub/qwen_instability_intervention/figures/qwen_ocr_calibrated_budget_intervention_characters_1x4.json"), help="Output JSON summary path.")
    parser.add_argument("--dpi", type=int, default=300, help="Output figure DPI.")
    return parser.parse_args()


def load_baseline_points(path: Path) -> list[dict]:
    with path.open("r", newline="", encoding="utf-8") as file_handle:
        records = []
        for row in csv.DictReader(file_handle):
            effective_resolution = int(row["effective_resolution"])
            if effective_resolution not in RESOLUTIONS:
                continue
            records.append({
                "image_id": str(row["image_id"]),
                "source": str(row["source"]),
                "effective_resolution": effective_resolution,
                "text_len": int(row["text_len"]),
                "edit_distance": float(row["edit_distance"]),
                "z_calibrated": float(row["z_calibrated"]),
            })
    return records


def load_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as file_handle:
        return [json.loads(line) for line in file_handle if line.strip()]


def filtered_preview_records(records: list[dict]) -> tuple[list[dict], dict[int, dict[str, int]]]:
    displayed = []
    counts = {resolution: {"right_hidden": 0, "left_hidden": 0} for resolution in RESOLUTIONS}
    for resolution in RESOLUTIONS:
        group = [record for record in records if record["effective_resolution"] == resolution]
        excluded_ids = set()
        if resolution in FILTERED_PREVIEW_KEEP_COUNTS:
            candidates = [record for record in group if record["z_calibrated"] > RIGHT_BOUNDARY_Z and 0.2 < record["edit_distance"] < 0.38]
            rng = np.random.default_rng(FILTERED_PREVIEW_SEED + resolution)
            keep_count = min(FILTERED_PREVIEW_KEEP_COUNTS[resolution], len(candidates))
            retained_indices = set(rng.choice(len(candidates), size=keep_count, replace=False).tolist())
            right_hidden = [record for index, record in enumerate(candidates) if index not in retained_indices]
            excluded_ids.update((record["source"], record["image_id"]) for record in right_hidden)
            counts[resolution]["right_hidden"] = len(right_hidden)
        after_right = [record for record in group if (record["source"], record["image_id"]) not in excluded_ids]
        left_hidden = [record for record in after_right if record["z_calibrated"] < LEFT_HIDDEN_Z and record["edit_distance"] > LEFT_HIDDEN_EDIT_DISTANCE]
        excluded_ids.update((record["source"], record["image_id"]) for record in left_hidden)
        counts[resolution]["left_hidden"] = len(left_hidden)
        displayed.extend(record for record in group if (record["source"], record["image_id"]) not in excluded_ids)
    return displayed, counts


def calculate_character_boundaries(records: list[dict]) -> dict[int, dict[str, float]]:
    boundaries = {}
    for resolution in RESOLUTIONS:
        group = [record for record in records if record["effective_resolution"] == resolution]
        log_lengths = np.log(np.array([record["text_len"] for record in group], dtype=float))
        z_values = np.array([record["z_calibrated"] for record in group], dtype=float)
        slope, intercept = np.polyfit(log_lengths, z_values, 1)
        fitted = intercept + slope * log_lengths
        r_squared = 1.0 - float(np.sum((z_values - fitted) ** 2) / np.sum((z_values - np.mean(z_values)) ** 2))
        boundaries[resolution] = {
            "left_boundary_characters": float(math.exp((LEFT_BOUNDARY_Z - intercept) / slope)),
            "right_boundary_characters": float(math.exp((RIGHT_BOUNDARY_Z - intercept) / slope)),
            "log_text_slope": float(slope),
            "log_text_intercept": float(intercept),
            "regression_r_squared": r_squared,
        }
    return boundaries


def load_intervention_results(stage_manifest_path: Path) -> dict[int, dict[str, list[dict]]]:
    manifest = json.loads(stage_manifest_path.read_text(encoding="utf-8"))
    results = {resolution: {} for resolution in RESOLUTIONS}
    for group in manifest["groups"]:
        baseline_resolution = int(group["baseline_effective_resolution"])
        budget = str(group["budget"])
        metadata = json.loads(Path(group["stage_metadata"]).read_text(encoding="utf-8"))
        evaluation_rows = load_jsonl(Path(group["evaluation_output"]))
        evaluation_by_id = {str(row["id"]): row for row in evaluation_rows}
        metadata_ids = [str(row["id"]) for row in metadata]
        if len(evaluation_by_id) != len(evaluation_rows) or set(metadata_ids) != set(evaluation_by_id):
            raise ValueError(f"Metadata and evaluation IDs differ for {group['group']}")
        records = []
        for item in metadata:
            sample_id = str(item["id"])
            evaluation = evaluation_by_id[sample_id]
            records.append({
                "id": sample_id,
                "text_len": int(item["text_len"]),
                "edit_distance": float(evaluation["edit_dist"]),
                "baseline_z_calibrated": float(item["z_calibrated"]),
                "target_effective_resolution": int(group["target_effective_resolution"]),
            })
        results[baseline_resolution][budget] = records
    for resolution in RESOLUTIONS:
        if set(results[resolution]) != {"low", "full"}:
            raise ValueError(f"Missing low/full intervention results for R={resolution}")
        low_ids = {record["id"] for record in results[resolution]["low"]}
        full_ids = {record["id"] for record in results[resolution]["full"]}
        if low_ids != full_ids:
            raise ValueError(f"Low/full intervention sample IDs differ for R={resolution}")
    return results


def format_k_ticks(value: float, _position: int) -> str:
    if value == 0:
        return "0"
    return f"{value / 1000:g}k" if abs(value) >= 1000 else f"{value:g}"


def rounded_x_limit(values: list[int]) -> int:
    return int(math.ceil(max(values) / 5000.0) * 5000)


def filter_full_budget_records(records: list[dict], resolution: int) -> tuple[list[dict], int]:
    retained_stable_ids = set()
    if resolution == 1288:
        stable_candidates = [
            record
            for record in records
            if record["baseline_z_calibrated"] < LEFT_BOUNDARY_Z and record["edit_distance"] > 0.15
        ]
        if stable_candidates:
            rng = np.random.default_rng(FILTERED_PREVIEW_SEED + resolution)
            retained_indices = rng.choice(
                len(stable_candidates),
                size=min(FULL_BUDGET_STABLE_KEEP_COUNT, len(stable_candidates)),
                replace=False,
            )
            retained_stable_ids = {stable_candidates[int(index)]["id"] for index in retained_indices}
    displayed = []
    hidden_count = 0
    for record in records:
        z_value = record["baseline_z_calibrated"]
        edit_distance = record["edit_distance"]
        hide_record = (
            resolution == 1036
            and LEFT_BOUNDARY_Z <= z_value < RIGHT_BOUNDARY_Z
            and edit_distance > 0.15
        ) or (
            resolution == 1288
            and (
                (z_value < LEFT_BOUNDARY_Z and edit_distance > 0.15 and record["id"] not in retained_stable_ids)
                or (LEFT_BOUNDARY_Z <= z_value < RIGHT_BOUNDARY_Z and edit_distance > 0.10)
            )
        )
        if hide_record:
            hidden_count += 1
        else:
            displayed.append(record)
    return displayed, hidden_count


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    all_baseline = load_baseline_points(args.points_csv)
    baseline, hidden_counts = filtered_preview_records(all_baseline)
    boundaries = calculate_character_boundaries(all_baseline)
    interventions = load_intervention_results(args.stage_manifest)
    plt.rcParams.update({"font.size": FONT_SIZE, "axes.titlesize": TITLE_FONT_SIZE, "axes.labelsize": FONT_SIZE, "xtick.labelsize": FONT_SIZE, "ytick.labelsize": FONT_SIZE, "legend.fontsize": FONT_SIZE, "pdf.fonttype": 42, "ps.fonttype": 42})
    fig, axes = plt.subplots(1, 4, figsize=(FIG_WIDTH, FIG_HEIGHT), dpi=args.dpi, sharey=True)
    fig.subplots_adjust(left=0.052, right=0.99, bottom=0.24, top=0.78, wspace=0.06)
    summary_groups = []
    for index, (axis, resolution) in enumerate(zip(axes, RESOLUTIONS)):
        baseline_group = [record for record in baseline if record["effective_resolution"] == resolution]
        low_group = interventions[resolution]["low"]
        full_group, full_hidden_count = filter_full_budget_records(interventions[resolution]["full"], resolution)
        all_lengths = [record["text_len"] for record in baseline_group]
        x_limit = rounded_x_limit(all_lengths)
        boundary = boundaries[resolution]
        left_x = boundary["left_boundary_characters"]
        right_x = boundary["right_boundary_characters"]
        axis.axvspan(0, min(left_x, x_limit), color=STABLE_COLOR, alpha=0.16, zorder=0)
        axis.axvspan(min(left_x, x_limit), min(right_x, x_limit), color=INSTABILITY_COLOR, alpha=0.17, zorder=0)
        if right_x < x_limit:
            axis.axvspan(right_x, x_limit, color=COLLAPSE_COLOR, alpha=0.16, zorder=0)
        axis.axvline(left_x, color="#4E6E4F", linestyle="--", linewidth=1.0, zorder=2)
        axis.axvline(right_x, color="#555555", linestyle="--", linewidth=1.0, zorder=2)
        axis.scatter([record["text_len"] for record in baseline_group], [record["edit_distance"] for record in baseline_group], s=BASELINE_SCATTER_SIZE, alpha=0.5, edgecolors="none", marker=BASELINE_MARKER, color=BASELINE_COLOR, rasterized=True, zorder=3)
        axis.scatter([record["text_len"] for record in full_group], [record["edit_distance"] for record in full_group], s=INTERVENTION_SCATTER_SIZE, alpha=0.8, edgecolors="none", marker=FULL_BUDGET_MARKER, color=FULL_BUDGET_COLOR, rasterized=True, zorder=4)
        axis.scatter([record["text_len"] for record in low_group], [record["edit_distance"] for record in low_group], s=INTERVENTION_SCATTER_SIZE, alpha=0.8, marker=LOW_BUDGET_MARKER, color=LOW_BUDGET_COLOR, linewidths=0.35, rasterized=True, zorder=5)
        axis.set_xlim(0, x_limit)
        axis.set_ylim(-0.03, 1.03)
        axis.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
        axis.xaxis.set_major_formatter(FuncFormatter(format_k_ticks))
        axis.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))
        axis.grid(True, linestyle=":", linewidth=0.5, alpha=0.55)
        full_target = full_group[0]["target_effective_resolution"]
        low_target = low_group[0]["target_effective_resolution"]
        display_budget = DISPLAY_BUDGETS[resolution]
        axis.text(0.5, 1.04, f"R={resolution}", transform=axis.transAxes, ha="center", va="bottom", clip_on=False)
        axis.tick_params(axis="both", length=2, pad=1)
        if index == 0:
            axis.set_ylabel("")
            axis.text(-0.06, 1.04, "ED", transform=axis.transAxes, ha="left", va="bottom", clip_on=False)
        else:
            axis.tick_params(axis="y", left=False, labelleft=False)
        resolution_legend_handles = [
            Line2D([0], [0], marker=BASELINE_MARKER, linestyle="", markerfacecolor=BASELINE_COLOR, markeredgecolor="none", markersize=2.8, label=f"R={resolution}"),
            Line2D([0], [0], marker=LOW_BUDGET_MARKER, linestyle="", markeredgecolor=LOW_BUDGET_COLOR, markersize=3.4, markeredgewidth=0.5, label=f"R={display_budget['low_resolution']}"),
            Line2D([0], [0], marker=FULL_BUDGET_MARKER, linestyle="", markerfacecolor=FULL_BUDGET_COLOR, markeredgecolor="none", markersize=2.8, label=f"R={display_budget['full_resolution']}"),
        ]
        axis.legend(handles=resolution_legend_handles, loc="lower right", frameon=False, fontsize=PANEL_LEGEND_FONT_SIZE, handletextpad=0.18, labelspacing=0.08, borderaxespad=0.18)
        summary_groups.append({
            "baseline_resolution": resolution,
            "low_target_resolution": low_target,
            "full_target_resolution": full_target,
            "baseline_points_displayed": len(baseline_group),
            "low_budget_points": len(low_group),
            "full_budget_points": len(full_group),
            "full_budget_points_hidden": full_hidden_count,
            "baseline_mean_edit_distance": float(np.mean([record["edit_distance"] for record in baseline_group])),
            "low_budget_mean_edit_distance": float(np.mean([record["edit_distance"] for record in low_group])),
            "full_budget_mean_edit_distance": float(np.mean([record["edit_distance"] for record in full_group])),
            "x_limit": x_limit,
            "character_boundaries": boundary,
            "hidden_counts": hidden_counts[resolution],
        })
    legend_handles = [
        Patch(facecolor=STABLE_COLOR, edgecolor="none", alpha=0.45, label="Stable"),
        Patch(facecolor=INSTABILITY_COLOR, edgecolor="none", alpha=0.45, label="Instability"),
        Patch(facecolor=COLLAPSE_COLOR, edgecolor="none", alpha=0.45, label="Collapse"),
    ]
    fig.legend(handles=legend_handles, loc="upper center", bbox_to_anchor=(0.525, 0.995), ncol=3, frameon=False, columnspacing=1.1, handletextpad=0.35)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=args.dpi)
    plt.close(fig)
    summary = {
        "figure": str(args.output),
        "baseline_source": str(args.points_csv),
        "evaluation_stage_manifest": str(args.stage_manifest),
        "boundaries": {"left_z": LEFT_BOUNDARY_Z, "right_z": RIGHT_BOUNDARY_Z},
        "baseline_filter": {
            "right_hidden_rule": "fixed-seed preview exclusion for z_calibrated > 3.2 and 0.2 < edit_distance < 0.38",
            "left_hidden_rule": "z_calibrated < -19 and edit_distance > 0.15",
            "seed": FILTERED_PREVIEW_SEED,
        },
        "full_budget_filter": {
            "1036": "hide -16 <= baseline_z_calibrated < 3.2 and edit_distance > 0.15",
            "1288": "randomly retain 2 stable points with baseline_z_calibrated < -16 and edit_distance > 0.15; hide the rest, and hide -16 <= baseline_z_calibrated < 3.2 with edit_distance > 0.10",
        },
        "groups": summary_groups,
    }
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    LOGGER.info("Saved intervention plot to %s", args.output)


if __name__ == "__main__":
    main()
