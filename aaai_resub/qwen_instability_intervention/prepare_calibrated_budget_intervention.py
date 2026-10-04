import argparse
import csv
import json
import logging
import os
import shutil
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


LOGGER = logging.getLogger(__name__)
RIGHT_BOUNDARY_Z = 3.2
LEFT_HIDDEN_Z = -19.0
LEFT_HIDDEN_EDIT_DISTANCE = 0.15
FILTERED_PREVIEW_SEED = 20260724
FILTERED_PREVIEW_KEEP_COUNTS = {504: 1, 644: 1, 1036: 2}
GROUPS = [
    {"baseline_resolution": 504, "budget": "low", "target_resolution": 672},
    {"baseline_resolution": 504, "budget": "full", "target_resolution": 868},
    {"baseline_resolution": 644, "budget": "low", "target_resolution": 868},
    {"baseline_resolution": 644, "budget": "full", "target_resolution": 1120},
    {"baseline_resolution": 1036, "budget": "low", "target_resolution": 1400},
    {"baseline_resolution": 1036, "budget": "full", "target_resolution": 1792},
    {"baseline_resolution": 1288, "budget": "low", "target_resolution": 1764},
    {"baseline_resolution": 1288, "budget": "full", "target_resolution": 2212},
]
EFFECTIVE_TO_NOMINAL = {504: 512, 644: 640, 1036: 1024, 1288: 1280}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Select calibrated Qwen OCR Stable plus Instability samples and prepare shared resize inputs.")
    parser.add_argument("--points_csv", type=Path, default=Path("aaai_resub/qwen_instability_intervention/all_parameter_calibration_28_6_0/qwen_ocr_28_6_0_all_parameters_calibrated_points.csv"), help="All-parameter calibrated Qwen point CSV.")
    parser.add_argument("--original_gt", type=Path, default=Path("VQAExp/data/Retrieval/retrieval_cropped_0_25k_600.json"), help="Original VQA OCR ground-truth JSON.")
    parser.add_argument("--pride_gt", type=Path, default=Path("aaai_resub/qwen_instability_intervention/pride_long_context/data/pride_25k_48k_all_gt.json"), help="Pride long-context ground-truth JSON.")
    parser.add_argument("--original_images", type=Path, default=Path("VQAExp/data/Retrieval/28_6_0/images"), help="Original VQA mother-image directory.")
    parser.add_argument("--pride_images", type=Path, default=Path("aaai_resub/qwen_instability_intervention/pride_long_context/rendered/mother_images/28_6_0/images"), help="Pride long-context mother-image directory.")
    parser.add_argument("--output_root", type=Path, default=Path("aaai_resub/qwen_instability_intervention/calibrated_budget_intervention"), help="Output root for manifests and shared resize inputs.")
    return parser.parse_args()


def load_records(path: Path) -> list[dict]:
    with path.open("r", newline="", encoding="utf-8") as file_handle:
        return list(csv.DictReader(file_handle))


def load_gt(path: Path) -> dict[str, dict]:
    records = json.loads(path.read_text(encoding="utf-8"))
    result = {str(record["id"]): record for record in records}
    if len(result) != len(records):
        raise ValueError(f"Duplicate IDs in ground-truth file: {path}")
    return result


def canonical_id(source: str, image_id: str) -> str:
    return f"{source}__{image_id}"


def select_right_hidden_records(records: list[dict]) -> set[tuple[int, str, str]]:
    excluded = set()
    for effective_resolution, keep_count in FILTERED_PREVIEW_KEEP_COUNTS.items():
        candidates = [record for record in records if int(record["effective_resolution"]) == effective_resolution and float(record["z_calibrated"]) > RIGHT_BOUNDARY_Z and 0.2 < float(record["edit_distance"]) < 0.38]
        rng = np.random.default_rng(FILTERED_PREVIEW_SEED + effective_resolution)
        retained_indices = set(rng.choice(len(candidates), size=min(keep_count, len(candidates)), replace=False).tolist())
        for index, record in enumerate(candidates):
            if index not in retained_indices:
                excluded.add((int(record["nominal_resolution"]), str(record["source"]), str(record["image_id"])))
    return excluded


def link_or_copy(source: Path, destination: Path) -> None:
    if destination.exists():
        if destination.stat().st_size != source.stat().st_size:
            raise ValueError(f"Existing raw image differs from expected source: {destination}")
        return
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def build_metadata(record: dict, gt_record: dict, source_image: Path, target_resolution: int, budget: str | None) -> dict:
    image_id = str(record["image_id"])
    source = str(record["source"])
    return {
        "id": canonical_id(source, image_id),
        "source": source,
        "source_image_id": image_id,
        "context": str(gt_record["context"]),
        "text_len": int(gt_record.get("text_len", len(str(gt_record["context"])))),
        "point_text_len": int(record["text_len"]),
        "baseline_nominal_resolution": int(record["nominal_resolution"]),
        "baseline_effective_resolution": int(record["effective_resolution"]),
        "target_effective_resolution": target_resolution,
        "budget": budget,
        "baseline_edit_distance": float(record["edit_distance"]),
        "z_calibrated": float(record["z_calibrated"]),
        "mother_image": str(source_image),
    }


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    records = load_records(args.points_csv)
    required_columns = {"image_id", "nominal_resolution", "effective_resolution", "text_len", "edit_distance", "source", "z_calibrated"}
    if not records or required_columns - set(records[0]):
        raise ValueError(f"Calibrated points CSV is missing required columns: {args.points_csv}")
    original_gt = load_gt(args.original_gt)
    pride_gt = load_gt(args.pride_gt)
    gt_by_source = {"original": original_gt, "pride_long_context": pride_gt}
    image_root_by_source = {"original": args.original_images, "pride_long_context": args.pride_images}
    unexpected_sources = {str(record["source"]) for record in records} - set(gt_by_source)
    if unexpected_sources:
        raise ValueError(f"Unexpected point sources: {sorted(unexpected_sources)}")
    right_hidden = select_right_hidden_records(records)
    left_hidden = {
        (int(record["nominal_resolution"]), str(record["source"]), str(record["image_id"]))
        for record in records
        if float(record["z_calibrated"]) < LEFT_HIDDEN_Z and float(record["edit_distance"]) > LEFT_HIDDEN_EDIT_DISTANCE
    }
    selected_by_resolution = defaultdict(list)
    exclusions = []
    for record in records:
        key = (int(record["nominal_resolution"]), str(record["source"]), str(record["image_id"]))
        if float(record["z_calibrated"]) >= RIGHT_BOUNDARY_Z:
            exclusions.append({"nominal_resolution": key[0], "source": key[1], "image_id": key[2], "reason": "z_not_less_than_right_boundary"})
            continue
        if key in left_hidden:
            exclusions.append({"nominal_resolution": key[0], "source": key[1], "image_id": key[2], "reason": "left_hidden_outlier"})
            continue
        selected_by_resolution[int(record["effective_resolution"])].append(record)
    for effective_resolution in EFFECTIVE_TO_NOMINAL:
        if not selected_by_resolution[effective_resolution]:
            raise ValueError(f"No selected samples for R={effective_resolution}")

    group_entries = []
    target_entries: dict[int, dict[str, dict]] = defaultdict(dict)
    group_metadata: dict[str, list[dict]] = {}
    for group in GROUPS:
        baseline_resolution = int(group["baseline_resolution"])
        target_resolution = int(group["target_resolution"])
        budget = str(group["budget"])
        group_name = f"{baseline_resolution}_to_{target_resolution}"
        metadata = []
        for record in selected_by_resolution[baseline_resolution]:
            source = str(record["source"])
            image_id = str(record["image_id"])
            gt_record = gt_by_source[source].get(image_id)
            if gt_record is None:
                raise KeyError(f"Missing GT for {source}:{image_id}")
            source_image = image_root_by_source[source] / f"{image_id}.jpg"
            if not source_image.is_file():
                raise FileNotFoundError(f"Missing mother image for {source}:{image_id}: {source_image}")
            item = build_metadata(record, gt_record, source_image, target_resolution, budget)
            metadata.append(item)
            target_item = dict(item)
            target_item["group_memberships"] = [group_name]
            existing = target_entries[target_resolution].get(item["id"])
            if existing is None:
                target_entries[target_resolution][item["id"]] = target_item
            else:
                existing["group_memberships"].append(group_name)
                if existing["context"] != target_item["context"] or existing["mother_image"] != target_item["mother_image"]:
                    raise ValueError(f"Inconsistent duplicated target record: {item['id']}")
        group_metadata[group_name] = metadata
        group_entries.append({
            "group": group_name,
            "budget": budget,
            "baseline_nominal_resolution": EFFECTIVE_TO_NOMINAL[baseline_resolution],
            "baseline_effective_resolution": baseline_resolution,
            "target_effective_resolution": target_resolution,
            "baseline_vision_tokens": (baseline_resolution // 28) ** 2,
            "target_vision_tokens": (target_resolution // 28) ** 2,
            "selected_samples": len(metadata),
            "metadata": str(args.output_root / "groups" / group_name / "metadata.json"),
            "target_metadata": str(args.output_root / "targets" / f"target_{target_resolution}" / "metadata.json"),
            "target_images": str(args.output_root / "targets" / f"target_{target_resolution}" / "images"),
        })

    for group in group_entries:
        metadata_path = Path(group["metadata"])
        metadata_path.parent.mkdir(parents=True, exist_ok=True)
        metadata_path.write_text(json.dumps(group_metadata[group["group"]], ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    target_summaries = []
    for target_resolution in sorted(target_entries):
        target_root = args.output_root / "targets" / f"target_{target_resolution}"
        raw_images = target_root / "raw_images"
        raw_images.mkdir(parents=True, exist_ok=True)
        metadata = [target_entries[target_resolution][sample_id] for sample_id in sorted(target_entries[target_resolution])]
        for item in metadata:
            link_or_copy(Path(item["mother_image"]), raw_images / f"{item['id']}.jpg")
        metadata_path = target_root / "metadata.json"
        metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        target_summaries.append({
            "target_effective_resolution": target_resolution,
            "vision_tokens": (target_resolution // 28) ** 2,
            "unique_samples": len(metadata),
            "metadata": str(metadata_path),
            "raw_images": str(raw_images),
            "images": str(target_root / "images"),
            "group_memberships": sorted({group_name for item in metadata for group_name in item["group_memberships"]}),
        })

    selection_summary = {
        "points_csv": str(args.points_csv),
        "selection_rule": f"z_calibrated < {RIGHT_BOUNDARY_Z:g}",
        "left_hidden_rule": f"z_calibrated < {LEFT_HIDDEN_Z:g} and edit_distance > {LEFT_HIDDEN_EDIT_DISTANCE:g}",
        "right_hidden_rule": "reproduced fixed-seed plotting exclusion; these samples are all outside z_calibrated < 3.2",
        "right_hidden_seed": FILTERED_PREVIEW_SEED,
        "right_hidden_excluded_count": len(right_hidden),
        "left_hidden_excluded_count": len(left_hidden),
        "selected_by_effective_resolution": {str(resolution): len(items) for resolution, items in sorted(selected_by_resolution.items())},
        "selected_by_source_and_resolution": {f"{resolution}:{source}": count for (resolution, source), count in sorted(Counter((int(item['effective_resolution']), str(item['source'])) for items in selected_by_resolution.values() for item in items).items())},
        "excluded_records": exclusions,
    }
    selection_path = args.output_root / "selection.json"
    selection_path.parent.mkdir(parents=True, exist_ok=True)
    selection_path.write_text(json.dumps(selection_summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    manifest = {
        "experiment": "calibrated_budget_intervention",
        "layout": "28_6_0",
        "boundaries": {"left_z": -16.0, "right_z": RIGHT_BOUNDARY_Z},
        "groups": group_entries,
        "targets": target_summaries,
        "selection": str(selection_path),
        "total_group_memberships": sum(group["selected_samples"] for group in group_entries),
        "total_unique_resize_inputs": sum(target["unique_samples"] for target in target_summaries),
    }
    manifest_path = args.output_root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    LOGGER.info("Prepared %d logical groups and %d shared resize targets", len(group_entries), len(target_summaries))
    LOGGER.info("Selected samples by original resolution: %s", selection_summary["selected_by_effective_resolution"])
    LOGGER.info("Saved manifest to %s", manifest_path)


if __name__ == "__main__":
    main()
