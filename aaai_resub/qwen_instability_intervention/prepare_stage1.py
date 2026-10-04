import argparse
import json
import logging
import math
import shutil
from pathlib import Path


NOMINAL_TO_EFFECTIVE = {512: 504, 640: 644, 1024: 1036, 1280: 1288}
QWEN_FACTOR = 28


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare manually bounded Qwen Instability samples for resolution intervention.")
    parser.add_argument("--boundary_json", type=Path, default=Path("aaai_resub/qwen_instability_intervention/figures/qwen_ocr_28_6_0_boundaries_manual.json"), help="Manual-boundary calculation JSON.")
    parser.add_argument("--eval_root", type=Path, default=Path("VQAExp/eval_results/ocr/qwen25vl7b/28_6_0"), help="Baseline Qwen OCR evaluation root.")
    parser.add_argument("--gt_json", type=Path, default=Path("VQAExp/data/Retrieval/retrieval_cropped_0_25k_600.json"), help="Ground-truth metadata JSON.")
    parser.add_argument("--source_images", type=Path, default=Path("VQAExp/data/Retrieval/28_6_0/images"), help="High-quality rendered mother images.")
    parser.add_argument("--fit_summary", type=Path, default=Path("VQAExp/scaling_law/Fitting/ocr/512/em_fit_summary.json"), help="Qwen fit summary containing the shared alpha coefficient.")
    parser.add_argument("--output_root", type=Path, default=Path("aaai_resub/qwen_instability_intervention/stage1"), help="Stage-one output root.")
    parser.add_argument("--selection_phase", choices=["stable", "instability"], default="instability", help="Select samples from either side of the manually specified Stable boundary.")
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as file_handle:
        return [json.loads(line) for line in file_handle if line.strip()]


def ceil_to_factor(value: float, factor: int) -> int:
    return int(math.ceil(value / factor) * factor)


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    boundary_config = json.loads(args.boundary_json.read_text(encoding="utf-8"))
    fit_summary = json.loads(args.fit_summary.read_text(encoding="utf-8"))
    alpha = float(fit_summary["gating_params"]["alpha"])
    gt_records = json.loads(args.gt_json.read_text(encoding="utf-8"))
    gt_by_id = {str(record["id"]): record for record in gt_records}
    if len(gt_by_id) != len(gt_records):
        raise ValueError(f"Duplicate IDs in {args.gt_json}")

    groups = []
    for nominal_resolution, effective_resolution in NOMINAL_TO_EFFECTIVE.items():
        boundary = boundary_config["boundaries"][str(nominal_resolution)]
        stable_characters = float(boundary["stable_characters"])
        collapse_characters = float(boundary["collapse_characters"])
        stable_z = float(boundary["stable_z"])
        collapse_z = float(boundary["collapse_z"])
        required_size = effective_resolution * math.exp((collapse_z - stable_z) / (2.0 * alpha))
        target_size = ceil_to_factor(required_size, QWEN_FACTOR)
        target_z_conservative = collapse_z - 2.0 * alpha * math.log(target_size / effective_resolution)
        if target_z_conservative > stable_z + 1e-12:
            raise ValueError(f"Rounded target does not reach Stable for R={effective_resolution}")

        baseline_path = args.eval_root / f"ocr_edit_distance_res_{nominal_resolution}.jsonl"
        baseline_rows = load_jsonl(baseline_path)
        if args.selection_phase == "stable":
            selected_rows = [record for record in baseline_rows if int(record["text_len"]) < stable_characters]
        else:
            selected_rows = [record for record in baseline_rows if stable_characters <= int(record["text_len"]) < collapse_characters]
        if not selected_rows:
            raise ValueError(f"No {args.selection_phase} samples selected for R={effective_resolution}")

        group_name = f"{effective_resolution}_to_{target_size}"
        group_root = args.output_root / "data" / group_name
        raw_images_dir = group_root / "raw_images"
        raw_images_dir.mkdir(parents=True, exist_ok=True)
        for old_image in raw_images_dir.glob("*.jpg"):
            old_image.unlink()

        selected_metadata = []
        selected_manifest = []
        for baseline_record in sorted(selected_rows, key=lambda item: (int(item["text_len"]), str(item["id"]))):
            sample_id = str(baseline_record["id"])
            if sample_id not in gt_by_id:
                raise KeyError(f"Missing GT record for {sample_id}")
            source_image = args.source_images / f"{sample_id}.jpg"
            if not source_image.is_file():
                raise FileNotFoundError(f"Missing mother image: {source_image}")
            shutil.copy2(source_image, raw_images_dir / source_image.name)
            metadata_record = dict(gt_by_id[sample_id])
            metadata_record.update({
                "baseline_nominal_resolution": nominal_resolution,
                "baseline_effective_resolution": effective_resolution,
                "target_effective_resolution": target_size,
                "baseline_edit_distance": float(baseline_record["edit_dist"]),
                "selection_phase": f"{args.selection_phase}_manual_boundary",
            })
            selected_metadata.append(metadata_record)
            selected_manifest.append({
                "id": sample_id,
                "text_len": int(baseline_record["text_len"]),
                "baseline_edit_distance": float(baseline_record["edit_dist"]),
                "source_image": str(source_image),
            })

        metadata_path = group_root / "metadata.json"
        selection_path = group_root / "selection.json"
        metadata_path.write_text(json.dumps(selected_metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        selection_path.write_text(json.dumps(selected_manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        group_summary = {
            "group": group_name,
            "baseline_nominal_resolution": nominal_resolution,
            "baseline_effective_resolution": effective_resolution,
            "target_effective_resolution": target_size,
            "baseline_vision_tokens": (effective_resolution // QWEN_FACTOR) ** 2,
            "target_vision_tokens": (target_size // QWEN_FACTOR) ** 2,
            "stable_characters": stable_characters,
            "collapse_characters": collapse_characters,
            "stable_z": stable_z,
            "collapse_z": collapse_z,
            "shared_alpha": alpha,
            "continuous_required_size": required_size,
            "target_z_conservative": target_z_conservative,
            "selected_samples": len(selected_metadata),
            "metadata": str(metadata_path),
            "raw_images": str(raw_images_dir),
        }
        groups.append(group_summary)
        logging.info("Prepared %s with %d samples; required %.2f, rounded target %d", group_name, len(selected_metadata), required_size, target_size)

    args.output_root.mkdir(parents=True, exist_ok=True)
    manifest = {
        "layout": "28_6_0",
        "selection_phase": args.selection_phase,
        "selection": "text_len < stable_characters" if args.selection_phase == "stable" else "stable_characters <= text_len < manual_collapse_characters",
        "target_rule": "ceil28(R * exp((Z_right - Z_stable) / (2 * alpha))) with source-resolution intercept held fixed",
        "boundary_source": str(args.boundary_json),
        "groups": groups,
        "total_selected_group_memberships": sum(group["selected_samples"] for group in groups),
    }
    (args.output_root / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    logging.info("Saved stage-one manifest to %s", args.output_root / "manifest.json")


if __name__ == "__main__":
    main()
