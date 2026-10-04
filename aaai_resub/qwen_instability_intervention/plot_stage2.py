import argparse
import json
import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter, FuncFormatter


FIG_WIDTH = 7.2
FIG_HEIGHT = 2.15
FONT_SIZE = 8
POINT_COLOR = "#1F77B4"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot Stage 2 Qwen OCR intervention edit distances in a 1-by-4 figure.")
    parser.add_argument("--stage2_roots", type=Path, nargs="+", default=[Path("aaai_resub/qwen_instability_intervention/stage2")], help="One or more Stage 2 roots whose disjoint sample sets will be combined.")
    parser.add_argument("--output", type=Path, default=Path("aaai_resub/qwen_instability_intervention/figures/qwen_ocr_instability_to_stable_stage2.jpg"), help="Output JPG path.")
    parser.add_argument("--summary", type=Path, default=Path("aaai_resub/qwen_instability_intervention/stage2/evaluation_summary.json"), help="Output JSON summary path.")
    parser.add_argument("--dpi", type=int, default=300, help="Output resolution in dots per inch.")
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as file_handle:
        return [json.loads(line) for line in file_handle if line.strip()]


def format_k_ticks(value: float, _position: int) -> str:
    if value == 0:
        return "0"
    if abs(value) >= 1000:
        return f"{value / 1000:g}k"
    return f"{value:g}"


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    plt.rcParams.update({
        "font.size": FONT_SIZE,
        "axes.titlesize": FONT_SIZE,
        "axes.labelsize": FONT_SIZE,
        "xtick.labelsize": FONT_SIZE,
        "ytick.labelsize": FONT_SIZE,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })
    manifests = [json.loads((root / "manifest.json").read_text(encoding="utf-8")) for root in args.stage2_roots]
    groups = manifests[0]["groups"]
    if len(groups) != 4:
        raise ValueError(f"Expected four Stage 2 groups, found {len(groups)}")
    group_maps = [{str(group["group"]): group for group in manifest["groups"]} for manifest in manifests]
    reference_group_names = [str(group["group"]) for group in groups]
    for root, group_map in zip(args.stage2_roots, group_maps):
        if list(group_map) != reference_group_names:
            raise ValueError(f"Stage 2 group order differs in {root}")

    fig, axes = plt.subplots(1, 4, figsize=(FIG_WIDTH, FIG_HEIGHT), dpi=args.dpi, sharex=True, sharey=True)
    fig.subplots_adjust(left=0.065, right=0.982, bottom=0.25, top=0.82, wspace=0.08)
    summaries = []

    for index, (axis, group) in enumerate(zip(axes, groups)):
        group_name = str(group["group"])
        baseline_resolution = int(group["baseline_effective_resolution"])
        target_resolution = int(group["target_effective_resolution"])
        records = []
        metadata_by_id = {}
        expected_samples = 0
        for root, group_map in zip(args.stage2_roots, group_maps):
            source_group = group_map[group_name]
            source_records = load_jsonl(Path(source_group["stage2_evaluation"]))
            source_metadata = json.loads(Path(source_group["stage2_metadata"]).read_text(encoding="utf-8"))
            source_metadata_by_id = {str(record["id"]): record for record in source_metadata}
            source_ids = {str(record["id"]) for record in source_records}
            overlap = set(metadata_by_id) & set(source_metadata_by_id)
            if overlap:
                raise ValueError(f"Duplicate sample IDs across Stage 2 roots for {group_name}: {sorted(overlap)[:5]}")
            if source_ids != set(source_metadata_by_id):
                raise ValueError(f"Evaluation and metadata IDs differ in {root} for {group_name}")
            records.extend(source_records)
            metadata_by_id.update(source_metadata_by_id)
            expected_samples += int(source_group["selected_samples"])
        if len(records) != expected_samples:
            raise ValueError(f"Unexpected combined evaluated sample count for {group_name}: {len(records)}")
        if {str(record["id"]) for record in records} != set(metadata_by_id):
            raise ValueError(f"Evaluation and metadata IDs differ for {group_name}")

        text_lengths = [int(record["text_len"]) for record in records]
        edit_distances = [float(record["edit_dist"]) for record in records]
        baseline_distances = [float(metadata_by_id[str(record["id"])]["baseline_edit_distance"]) for record in records]
        mean_edit_distance = sum(edit_distances) / len(edit_distances)
        baseline_mean = sum(baseline_distances) / len(baseline_distances)
        axis.scatter(text_lengths, edit_distances, s=5, alpha=0.6, edgecolors="none", color=POINT_COLOR, rasterized=True)
        axis.set_xlim(0, 25000)
        axis.set_ylim(-0.03, 1.03)
        axis.set_xticks([0, 5000, 10000, 15000, 20000, 25000])
        axis.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
        axis.xaxis.set_major_formatter(FuncFormatter(lambda value, position, hide_zero=index > 0: "" if hide_zero and value == 0 else format_k_ticks(value, position)))
        axis.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))
        axis.grid(True, linestyle=":", linewidth=0.5, alpha=0.6)
        axis.set_title(f"R={baseline_resolution}→{target_resolution}", pad=3)
        axis.text(0.97, 0.96, f"mean={mean_edit_distance:.3f}", transform=axis.transAxes, ha="right", va="top", fontsize=7)
        axis.tick_params(axis="both", length=2, pad=1)
        if index == 0:
            axis.set_ylabel("Normalized edit distance")
        else:
            axis.tick_params(axis="y", left=False, labelleft=False)

        summaries.append({
            "group": group_name,
            "baseline_effective_resolution": baseline_resolution,
            "target_effective_resolution": target_resolution,
            "target_vision_tokens": int(group["target_vision_tokens"]),
            "num_samples": len(records),
            "num_stable_samples": sum(record.get("selection_phase") == "stable_manual_boundary" for record in metadata_by_id.values()),
            "num_instability_samples": sum(record.get("selection_phase") == "instability_manual_boundary" for record in metadata_by_id.values()),
            "baseline_mean_edit_distance": baseline_mean,
            "target_mean_edit_distance": mean_edit_distance,
            "mean_edit_distance_change": mean_edit_distance - baseline_mean,
            "fraction_target_edit_distance_below_0.1": sum(value < 0.1 for value in edit_distances) / len(edit_distances),
            "fraction_improved": sum(target < baseline for target, baseline in zip(edit_distances, baseline_distances)) / len(edit_distances),
        })

    fig.supxlabel("Number of characters", x=0.53, y=0.04, fontsize=FONT_SIZE)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, format=args.output.suffix.lstrip("."), dpi=args.dpi)
    plt.close(fig)
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps({"groups": summaries}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    logging.info("Saved Stage 2 plot to %s", args.output)
    logging.info("Saved Stage 2 evaluation summary to %s", args.summary)


if __name__ == "__main__":
    main()
