import argparse
import csv
import json
import logging
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.ticker import FormatStrFormatter, FuncFormatter
import numpy as np


NOMINAL_TO_EFFECTIVE = {512: 504, 640: 644, 1024: 1036, 1280: 1288}
LAYOUT = "28_6_0"
X_MAX = 25000
STABLE_COLOR = "#B3E8B8"
INSTABILITY_COLOR = "#EDD894"
COLLAPSE_COLOR = "#C1C0C0"
POINT_COLOR = "#1F77B4"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot Qwen 28_6_0 OCR scatter with expected-ED phase boundaries.")
    parser.add_argument("--eval_dir", type=Path, default=Path("VQAExp/eval_results/ocr/qwen25vl7b/28_6_0"), help="Directory containing per-resolution edit-distance JSONL files.")
    parser.add_argument("--scaling_csv", type=Path, default=Path("VQAExp/scaling_law/qwen_ocr_scaling_points.csv"), help="Qwen OCR scaling-point CSV.")
    parser.add_argument("--fit_root", type=Path, default=Path("VQAExp/scaling_law/Fitting/ocr"), help="Root containing per-resolution EM summaries.")
    parser.add_argument("--stable_ed", type=float, default=0.1, help="Expected edit-distance threshold between Stable and Instability.")
    parser.add_argument("--collapse_ed", type=float, default=0.7, help="Expected edit-distance threshold between Instability and Collapse.")
    parser.add_argument("--manual_collapse_characters", type=str, default=None, help="Optional comma-separated nominal-resolution boundaries, such as 512:7000,640:10000.")
    parser.add_argument("--output", type=Path, default=Path("aaai_resub/qwen_instability_intervention/figures/qwen_ocr_28_6_0_boundaries_ed_0.1_0.7.jpg"), help="Output JPG path.")
    parser.add_argument("--summary", type=Path, default=Path("aaai_resub/qwen_instability_intervention/figures/qwen_ocr_28_6_0_boundaries_ed_0.1_0.7.json"), help="Output boundary calculation JSON path.")
    parser.add_argument("--dpi", type=int, default=300, help="Output resolution in dots per inch.")
    return parser.parse_args()


def load_scatter(path: Path, expected_resolution: int) -> tuple[list[int], list[float], set[str]]:
    text_lengths = []
    edit_distances = []
    sample_ids = set()
    with path.open("r", encoding="utf-8") as file_handle:
        for line_number, line in enumerate(file_handle, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            if int(record["resolution"]) != expected_resolution:
                raise ValueError(f"Unexpected resolution in {path}:{line_number}")
            sample_id = str(record["id"])
            if sample_id in sample_ids:
                raise ValueError(f"Duplicate sample ID in {path}: {sample_id}")
            sample_ids.add(sample_id)
            text_lengths.append(int(record["text_len"]))
            edit_distances.append(float(record["edit_dist"]))
    return text_lengths, edit_distances, sample_ids


def expected_ed_to_z(expected_ed: float, low_mean: float, high_mean: float) -> float:
    probability = (expected_ed - low_mean) / (high_mean - low_mean)
    if not 0.0 < probability < 1.0:
        raise ValueError(f"Expected-ED threshold {expected_ed} is outside fitted component means {low_mean}, {high_mean}")
    return math.log(probability / (1.0 - probability))


def calculate_boundaries(args: argparse.Namespace) -> dict[int, dict]:
    rows_by_resolution = {resolution: [] for resolution in NOMINAL_TO_EFFECTIVE}
    with args.scaling_csv.open("r", newline="", encoding="utf-8") as file_handle:
        for row in csv.DictReader(file_handle):
            resolution = int(row["resolution"])
            if row["layout"] == LAYOUT and resolution in rows_by_resolution:
                rows_by_resolution[resolution].append(row)

    boundaries = {}
    for resolution, rows in rows_by_resolution.items():
        if not rows:
            raise ValueError(f"No scaling rows for layout={LAYOUT}, resolution={resolution}")
        summary = json.loads((args.fit_root / str(resolution) / "em_fit_summary.json").read_text(encoding="utf-8"))
        gating = summary["gating_params"]
        low_mean = float(summary["beta_params"]["low_ed"]["mean"])
        high_mean = float(summary["beta_params"]["high_ed"]["mean"])
        log_text_lengths = np.log(np.array([float(row["text_len"]) for row in rows], dtype=float))
        z_values = np.array([float(gating["w0"]) + float(gating["a"]) * float(row["qwen_logV"]) + float(gating["alpha"]) * float(row["qwen_logG"]) for row in rows], dtype=float)
        slope, intercept = np.polyfit(log_text_lengths, z_values, 1)
        fitted_z = intercept + slope * log_text_lengths
        r_squared = 1.0 - float(np.sum((z_values - fitted_z) ** 2) / np.sum((z_values - np.mean(z_values)) ** 2))
        stable_z = expected_ed_to_z(args.stable_ed, low_mean, high_mean)
        collapse_z = expected_ed_to_z(args.collapse_ed, low_mean, high_mean)
        boundaries[resolution] = {
            "nominal_resolution": resolution,
            "effective_resolution": NOMINAL_TO_EFFECTIVE[resolution],
            "stable_characters": float(math.exp((stable_z - intercept) / slope)),
            "collapse_characters": float(math.exp((collapse_z - intercept) / slope)),
            "stable_z": stable_z,
            "collapse_z": collapse_z,
            "low_ed_mean": low_mean,
            "high_ed_mean": high_mean,
            "log_text_slope": float(slope),
            "log_text_intercept": float(intercept),
            "regression_r_squared": r_squared,
        }
    return boundaries


def format_k_ticks(value: float, _position: int) -> str:
    if value == 0:
        return "0"
    return f"{value / 1000:g}k" if abs(value) >= 1000 else f"{value:g}"


def apply_manual_collapse_boundaries(boundaries: dict[int, dict], value: str | None) -> dict[int, float]:
    if not value:
        return {}
    overrides = {}
    for item in value.split(","):
        resolution_text, characters_text = item.split(":", maxsplit=1)
        resolution = int(resolution_text.strip())
        characters = float(characters_text.strip())
        if resolution not in boundaries:
            raise ValueError(f"Unknown nominal resolution in manual boundary: {resolution}")
        if characters <= 0:
            raise ValueError(f"Manual boundary must be positive: {characters}")
        boundary = boundaries[resolution]
        boundary["reference_collapse_characters"] = boundary["collapse_characters"]
        boundary["reference_collapse_z"] = boundary["collapse_z"]
        manual_z = float(boundary["log_text_intercept"]) + float(boundary["log_text_slope"]) * math.log(characters)
        probability = 1.0 / (1.0 + math.exp(-manual_z))
        expected_ed = float(boundary["low_ed_mean"]) + probability * (float(boundary["high_ed_mean"]) - float(boundary["low_ed_mean"]))
        boundary["collapse_characters"] = characters
        boundary["collapse_z"] = manual_z
        boundary["collapse_expected_ed"] = expected_ed
        boundary["collapse_mode"] = "manual_characters"
        overrides[resolution] = characters
    return overrides


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    boundaries = calculate_boundaries(args)
    manual_overrides = apply_manual_collapse_boundaries(boundaries, args.manual_collapse_characters)
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps({"layout": LAYOUT, "stable_ed": args.stable_ed, "reference_collapse_ed": args.collapse_ed, "manual_collapse_characters": {str(key): value for key, value in manual_overrides.items()}, "x_max": X_MAX, "boundaries": {str(key): value for key, value in boundaries.items()}}, indent=2) + "\n", encoding="utf-8")

    plt.rcParams.update({"font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8, "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 7.5})
    fig, axes = plt.subplots(1, 4, figsize=(7.2, 2.4), dpi=args.dpi, sharex=True, sharey=True)
    fig.subplots_adjust(left=0.065, right=0.982, bottom=0.22, top=0.71, wspace=0.08)

    reference_ids = None
    for index, (axis, resolution) in enumerate(zip(axes, NOMINAL_TO_EFFECTIVE)):
        scatter_path = args.eval_dir / f"ocr_edit_distance_res_{resolution}.jsonl"
        text_lengths, edit_distances, sample_ids = load_scatter(scatter_path, resolution)
        if reference_ids is None:
            reference_ids = sample_ids
        elif sample_ids != reference_ids:
            raise ValueError(f"Sample IDs differ across resolutions: {scatter_path}")

        boundary = boundaries[resolution]
        stable_x = boundary["stable_characters"]
        collapse_x = boundary["collapse_characters"]
        axis.axvspan(0, min(stable_x, X_MAX), color=STABLE_COLOR, alpha=0.22, zorder=0)
        axis.axvspan(min(stable_x, X_MAX), min(collapse_x, X_MAX), color=INSTABILITY_COLOR, alpha=0.22, zorder=0)
        if collapse_x < X_MAX:
            axis.axvspan(collapse_x, X_MAX, color=COLLAPSE_COLOR, alpha=0.22, zorder=0)
        axis.axvline(stable_x, color="#666666", linewidth=0.75, linestyle="--", zorder=2)
        axis.text(stable_x, 0.03, f"{stable_x / 1000:.1f}k", transform=axis.get_xaxis_transform(), rotation=90, ha="right", va="bottom", fontsize=6.5, color="#555555")
        if collapse_x <= X_MAX:
            axis.axvline(collapse_x, color="#666666", linewidth=0.75, linestyle="-.", zorder=2)
            axis.text(collapse_x, 0.03, f"{collapse_x / 1000:.1f}k", transform=axis.get_xaxis_transform(), rotation=90, ha="right", va="bottom", fontsize=6.5, color="#555555")

        axis.scatter(text_lengths, edit_distances, s=3, alpha=0.55, edgecolors="none", color=POINT_COLOR, rasterized=True, zorder=3)
        axis.set_xlim(0, X_MAX)
        axis.set_ylim(-0.03, 1.03)
        axis.set_xticks([0, 5000, 10000, 15000, 20000, 25000])
        axis.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
        axis.xaxis.set_major_formatter(FuncFormatter(lambda value, position, hide_zero=index > 0: "" if hide_zero and value == 0 else format_k_ticks(value, position)))
        axis.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))
        axis.grid(True, linestyle=":", linewidth=0.5, alpha=0.6)
        effective_resolution = NOMINAL_TO_EFFECTIVE[resolution]
        axis.set_title(f"R={effective_resolution}, VT={(effective_resolution // 28) ** 2}", pad=3)
        axis.text(0.97, 0.96, f"mean={sum(edit_distances) / len(edit_distances):.3f}", transform=axis.transAxes, ha="right", va="top", fontsize=7)
        axis.tick_params(axis="both", length=2, pad=1)
        if index == 0:
            axis.set_ylabel("Normalized edit distance")
        else:
            axis.tick_params(axis="y", left=False, labelleft=False)

    if manual_overrides:
        legend_handles = [
            Patch(facecolor=STABLE_COLOR, edgecolor="none", alpha=0.45, label=f"Stable boundary: E[D|Z]={args.stable_ed:g}"),
            Patch(facecolor=INSTABILITY_COLOR, edgecolor="none", alpha=0.45, label="Instability (manual right boundary)"),
            Patch(facecolor=COLLAPSE_COLOR, edgecolor="none", alpha=0.45, label="Collapse (manual boundary)"),
        ]
    else:
        legend_handles = [
            Patch(facecolor=STABLE_COLOR, edgecolor="none", alpha=0.45, label=f"Stable: E[D|Z] < {args.stable_ed:g}"),
            Patch(facecolor=INSTABILITY_COLOR, edgecolor="none", alpha=0.45, label=f"Instability: {args.stable_ed:g} ≤ E[D|Z] < {args.collapse_ed:g}"),
            Patch(facecolor=COLLAPSE_COLOR, edgecolor="none", alpha=0.45, label=f"Collapse: E[D|Z] ≥ {args.collapse_ed:g}"),
        ]
    fig.legend(handles=legend_handles, loc="upper center", bbox_to_anchor=(0.52, 0.985), ncol=3, frameon=False, columnspacing=1.0, handletextpad=0.35)
    fig.supxlabel("Number of characters", x=0.53, y=0.025, fontsize=8)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, format="jpg", dpi=args.dpi)
    plt.close(fig)
    logging.info("Saved boundary plot with %d samples per panel to %s", len(reference_ids or []), args.output)


if __name__ == "__main__":
    main()
