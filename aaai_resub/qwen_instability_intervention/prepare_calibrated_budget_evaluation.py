import argparse
import json
import logging
import shutil
from collections import Counter
from pathlib import Path


LOGGER = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare per-intervention-group OCR prediction and GT JSON files for edit-distance evaluation.")
    parser.add_argument("--root", type=Path, default=Path("aaai_resub/qwen_instability_intervention/calibrated_budget_intervention"), help="Calibrated budget intervention root.")
    parser.add_argument("--output_root", type=Path, default=Path("aaai_resub/qwen_instability_intervention/calibrated_budget_intervention/evaluation_stage"), help="Output root for group-specific prediction and metadata files.")
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict]:
    records = []
    with path.open("r", encoding="utf-8") as file_handle:
        for line_number, line in enumerate(file_handle, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            if "id" not in record:
                raise ValueError(f"Missing id in {path}:{line_number}")
            records.append(record)
    return records


def load_prediction_map(path: Path) -> dict[str, dict]:
    records = load_jsonl(path)
    prediction_map = {str(record["id"]): record for record in records}
    if len(prediction_map) != len(records):
        raise ValueError(f"Duplicate prediction IDs in {path}")
    return prediction_map


def write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file_handle:
        for record in records:
            file_handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    manifest_path = args.root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    prediction_maps = {}
    for target in manifest["targets"]:
        resolution = int(target["target_effective_resolution"])
        path = args.root / "ocr_predictions" / f"target_{resolution}" / "ocr_predictions_md_res.jsonl"
        prediction_maps[resolution] = load_prediction_map(path)
    prepared_groups = []
    for group in manifest["groups"]:
        group_name = str(group["group"])
        target_resolution = int(group["target_effective_resolution"])
        metadata_path = Path(group["metadata"])
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        metadata_ids = [str(record["id"]) for record in metadata]
        if len(metadata_ids) != len(set(metadata_ids)):
            raise ValueError(f"Duplicate metadata IDs for {group_name}")
        prediction_map = prediction_maps[target_resolution]
        missing = sorted(set(metadata_ids) - set(prediction_map))
        if missing:
            raise ValueError(f"Missing predictions for {group_name}: {missing[:3]}")
        group_predictions = [prediction_map[sample_id] for sample_id in metadata_ids]
        prediction_path = args.output_root / "predictions" / group_name / f"ocr_predictions_res_{target_resolution}.jsonl"
        prepared_metadata_path = args.output_root / "metadata" / f"{group_name}.json"
        write_jsonl(prediction_path, group_predictions)
        prepared_metadata_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(metadata_path, prepared_metadata_path)
        finish_reasons = Counter(str(record.get("finish_reason", "")) for record in group_predictions)
        prepared_groups.append({
            **group,
            "stage_predictions": str(prediction_path),
            "stage_metadata": str(prepared_metadata_path),
            "evaluation_output": str(args.output_root / "evaluation" / group_name / f"ocr_edit_distance_res_{target_resolution}.jsonl"),
            "finish_reasons": dict(finish_reasons),
        })
        LOGGER.info("Prepared %s with %d predictions at target %d", group_name, len(group_predictions), target_resolution)
    stage_manifest = {
        "source_manifest": str(manifest_path),
        "groups": prepared_groups,
        "total_group_memberships": sum(int(group["selected_samples"]) for group in prepared_groups),
    }
    output_path = args.output_root / "manifest.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(stage_manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    LOGGER.info("Saved evaluation-stage manifest to %s", output_path)


if __name__ == "__main__":
    main()
