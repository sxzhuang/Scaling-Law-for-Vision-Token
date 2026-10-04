import argparse
import json
import logging
import shutil
from pathlib import Path


LOGGER = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare Stage 1 Qwen OCR predictions for edit-distance evaluation.")
    parser.add_argument("--stage1_root", type=Path, default=Path("aaai_resub/qwen_instability_intervention/stage1"), help="Stage 1 data and OCR prediction root.")
    parser.add_argument("--output_root", type=Path, default=Path("aaai_resub/qwen_instability_intervention/stage2"), help="Stage 2 output root.")
    parser.add_argument("--allow_non_stop", action="store_true", help="Include length-limited predictions as model failures instead of rejecting the dataset.")
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


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    manifest_path = args.stage1_root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    prepared_groups = []

    for group in manifest["groups"]:
        group_name = str(group["group"])
        target_resolution = int(group["target_effective_resolution"])
        metadata_path = args.stage1_root / "data" / group_name / "metadata.json"
        source_predictions = args.stage1_root / "ocr_predictions" / group_name / "ocr_predictions_md_res.jsonl"
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        predictions = load_jsonl(source_predictions)
        metadata_ids = [str(record["id"]) for record in metadata]
        prediction_ids = [str(record["id"]) for record in predictions]
        if len(metadata_ids) != len(set(metadata_ids)):
            raise ValueError(f"Duplicate metadata IDs in {metadata_path}")
        if len(prediction_ids) != len(set(prediction_ids)):
            raise ValueError(f"Duplicate prediction IDs in {source_predictions}")
        if set(metadata_ids) != set(prediction_ids):
            missing = sorted(set(metadata_ids) - set(prediction_ids))
            extra = sorted(set(prediction_ids) - set(metadata_ids))
            raise ValueError(f"ID mismatch for {group_name}: missing={missing[:5]} extra={extra[:5]}")
        non_stop = [record["id"] for record in predictions if record.get("finish_reason") != "stop"]
        if non_stop and not args.allow_non_stop:
            raise ValueError(f"Non-stop predictions for {group_name}: {non_stop[:5]}")

        prediction_dir = args.output_root / "predictions" / group_name
        metadata_dir = args.output_root / "metadata"
        prediction_dir.mkdir(parents=True, exist_ok=True)
        metadata_dir.mkdir(parents=True, exist_ok=True)
        prepared_predictions = prediction_dir / f"ocr_predictions_res_{target_resolution}.jsonl"
        prepared_metadata = metadata_dir / f"{group_name}.json"
        shutil.copy2(source_predictions, prepared_predictions)
        shutil.copy2(metadata_path, prepared_metadata)
        prepared_groups.append({
            **group,
            "stage2_predictions": str(prepared_predictions),
            "stage2_metadata": str(prepared_metadata),
            "stage2_evaluation": str(args.output_root / "evaluation" / group_name / f"ocr_edit_distance_res_{target_resolution}.jsonl"),
            "non_stop_predictions": len(non_stop),
            "non_stop_ids": non_stop,
        })
        LOGGER.info("Prepared %s: %d matched predictions at target resolution %d", group_name, len(predictions), target_resolution)

    stage2_manifest = {
        "source_manifest": str(manifest_path),
        "groups": prepared_groups,
        "total_samples": sum(int(group["selected_samples"]) for group in prepared_groups),
    }
    output_manifest = args.output_root / "manifest.json"
    output_manifest.parent.mkdir(parents=True, exist_ok=True)
    output_manifest.write_text(json.dumps(stage2_manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    LOGGER.info("Saved Stage 2 manifest to %s", output_manifest)


if __name__ == "__main__":
    main()
