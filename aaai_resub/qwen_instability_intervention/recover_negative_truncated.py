import argparse
import csv
import json
import logging
import shutil
from pathlib import Path


LOGGER = logging.getLogger(__name__)
GROUPS = ("504_to_728", "644_to_840", "1036_to_1260", "1288_to_1652")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare or merge negative-control OCR samples truncated by max_tokens.")
    parser.add_argument("--mode", choices=["prepare", "merge"], required=True, help="Prepare rerun inputs or merge successful rerun outputs.")
    parser.add_argument("--stage_root", type=Path, default=Path("aaai_resub/qwen_instability_intervention/negative_control/stage1"), help="Negative-control Stage 1 root.")
    parser.add_argument("--rerun_root", type=Path, default=Path("aaai_resub/qwen_instability_intervention/negative_control/rerun_truncated"), help="Root for isolated truncated-sample reruns.")
    parser.add_argument("--merge_successful_only", action="store_true", help="Merge only rerun records with finish_reason=stop and leave other source records unchanged.")
    parser.add_argument("--merge_all", action="store_true", help="Merge all rerun records while preserving their actual finish reasons.")
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as file_handle:
        return [json.loads(line) for line in file_handle if line.strip()]


def find_non_stop_ids(stage_root: Path, group: str) -> list[str]:
    records_path = stage_root / "ocr_predictions" / group / "generation_records_md_res.csv"
    with records_path.open(newline="", encoding="utf-8") as file_handle:
        rows = list(csv.DictReader(file_handle))
    return [str(row["image_id"]) for row in rows if row["finish_reason"] != "stop"]


def prepare(args: argparse.Namespace) -> None:
    summaries = []
    for group in GROUPS:
        truncated_ids = find_non_stop_ids(args.stage_root, group)
        if not truncated_ids:
            continue
        group_root = args.rerun_root / group
        images_dir = group_root / "images"
        output_dir = group_root / "md_res"
        images_dir.mkdir(parents=True, exist_ok=True)
        output_dir.mkdir(parents=True, exist_ok=True)
        for path in images_dir.glob("*.jpg"):
            path.unlink()
        for path in output_dir.glob("*.md"):
            path.unlink()
        for path in (group_root / "ocr_predictions_md_res.jsonl", group_root / "generation_records_md_res.csv"):
            if path.exists():
                path.unlink()

        source_metadata_path = args.stage_root / "data" / group / "metadata.json"
        source_metadata = json.loads(source_metadata_path.read_text(encoding="utf-8"))
        metadata_by_id = {str(record["id"]): record for record in source_metadata}
        for sample_id in truncated_ids:
            source_image = args.stage_root / "data" / group / "images" / f"{sample_id}.jpg"
            if not source_image.is_file():
                raise FileNotFoundError(f"Missing rerun image: {source_image}")
            (images_dir / source_image.name).symlink_to(source_image.resolve())
        rerun_metadata = [metadata_by_id[sample_id] for sample_id in truncated_ids]
        metadata_path = group_root / "metadata.json"
        metadata_path.write_text(json.dumps(rerun_metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        summaries.append({"group": group, "ids": truncated_ids, "count": len(truncated_ids), "metadata": str(metadata_path)})
        LOGGER.info("Prepared %d truncated samples for %s", len(truncated_ids), group)

    manifest_path = args.rerun_root / "manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps({"groups": summaries, "total_samples": sum(item["count"] for item in summaries)}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    LOGGER.info("Saved rerun manifest with %d samples to %s", sum(item["count"] for item in summaries), manifest_path)


def merge(args: argparse.Namespace) -> None:
    if args.merge_successful_only and args.merge_all:
        raise ValueError("--merge_successful_only and --merge_all are mutually exclusive")
    manifest = json.loads((args.rerun_root / "manifest.json").read_text(encoding="utf-8"))
    for summary in manifest["groups"]:
        group = str(summary["group"])
        expected_ids = set(str(sample_id) for sample_id in summary["ids"])
        group_root = args.rerun_root / group
        replacement_records = load_jsonl(group_root / "ocr_predictions_md_res.jsonl")
        replacement_by_id = {str(record["id"]): record for record in replacement_records}
        if set(replacement_by_id) != expected_ids:
            raise ValueError(f"Rerun ID mismatch for {group}")
        non_stop = [sample_id for sample_id, record in replacement_by_id.items() if record.get("finish_reason") != "stop"]
        if non_stop and not args.merge_successful_only and not args.merge_all:
            raise ValueError(f"Rerun still truncated for {group}: {non_stop}")
        if args.merge_successful_only:
            replacement_by_id = {sample_id: record for sample_id, record in replacement_by_id.items() if record.get("finish_reason") == "stop"}

        prediction_root = args.stage_root / "ocr_predictions" / group
        main_jsonl = prediction_root / "ocr_predictions_md_res.jsonl"
        main_records = load_jsonl(main_jsonl)
        main_ids = [str(record["id"]) for record in main_records]
        if len(main_ids) != len(set(main_ids)):
            raise ValueError(f"Duplicate main prediction IDs for {group}")
        merged_records = [replacement_by_id.get(str(record["id"]), record) for record in main_records]
        main_jsonl.write_text("".join(json.dumps(record, ensure_ascii=False) + "\n" for record in merged_records), encoding="utf-8")

        for sample_id in replacement_by_id:
            shutil.copy2(group_root / "md_res" / f"{sample_id}.md", prediction_root / "md_res" / f"{sample_id}.md")

        generation_path = prediction_root / "generation_records_md_res.csv"
        with generation_path.open(newline="", encoding="utf-8") as file_handle:
            rows = list(csv.DictReader(file_handle))
            fieldnames = list(rows[0]) if rows else ["image_id", "finish_reason"]
        for row in rows:
            sample_id = str(row["image_id"])
            if sample_id in replacement_by_id:
                row["finish_reason"] = str(replacement_by_id[sample_id].get("finish_reason", "stop"))
        with generation_path.open("w", newline="", encoding="utf-8") as file_handle:
            writer = csv.DictWriter(file_handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        LOGGER.info("Merged %d rerun records into %s", len(replacement_by_id), group)


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    if args.mode == "prepare":
        prepare(args)
    else:
        merge(args)


if __name__ == "__main__":
    main()
