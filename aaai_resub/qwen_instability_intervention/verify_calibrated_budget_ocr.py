import argparse
import json
import logging
from collections import Counter
from pathlib import Path


LOGGER = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify that every calibrated-budget Qwen OCR target has complete prediction artifacts.")
    parser.add_argument("--root", type=Path, default=Path("aaai_resub/qwen_instability_intervention/calibrated_budget_intervention"), help="Calibrated budget intervention root.")
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
    manifest_path = args.root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for target in manifest["targets"]:
        resolution = int(target["target_effective_resolution"])
        expected_count = int(target["unique_samples"])
        metadata_path = Path(target["metadata"])
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        expected_ids = {str(record["id"]) for record in metadata}
        if len(metadata) != expected_count or len(expected_ids) != expected_count:
            raise ValueError(f"Invalid metadata for target {resolution}: expected={expected_count} metadata={len(metadata)} unique={len(expected_ids)}")
        prediction_root = args.root / "ocr_predictions" / f"target_{resolution}"
        markdown_dir = prediction_root / "md_res"
        prediction_path = prediction_root / "ocr_predictions_md_res.jsonl"
        markdown_ids = {path.stem for path in markdown_dir.glob("*.md")} if markdown_dir.is_dir() else set()
        if markdown_ids != expected_ids:
            raise ValueError(f"Markdown mismatch for target {resolution}: missing={sorted(expected_ids - markdown_ids)[:3]} extra={sorted(markdown_ids - expected_ids)[:3]}")
        if not prediction_path.is_file():
            raise FileNotFoundError(f"Missing prediction JSONL for target {resolution}: {prediction_path}")
        predictions = load_jsonl(prediction_path)
        prediction_ids = [str(record["id"]) for record in predictions]
        if len(prediction_ids) != len(set(prediction_ids)):
            duplicates = [sample_id for sample_id, count in Counter(prediction_ids).items() if count > 1]
            raise ValueError(f"Duplicate prediction IDs for target {resolution}: {duplicates[:3]}")
        if set(prediction_ids) != expected_ids:
            raise ValueError(f"Prediction mismatch for target {resolution}: missing={sorted(expected_ids - set(prediction_ids))[:3]} extra={sorted(set(prediction_ids) - expected_ids)[:3]}")
        finish_reasons = Counter(str(record.get("finish_reason", "")) for record in predictions)
        LOGGER.info("Verified target=%d samples=%d finish_reasons=%s", resolution, expected_count, dict(finish_reasons))
    LOGGER.info("All calibrated-budget OCR targets are complete and ID-consistent")


if __name__ == "__main__":
    main()
