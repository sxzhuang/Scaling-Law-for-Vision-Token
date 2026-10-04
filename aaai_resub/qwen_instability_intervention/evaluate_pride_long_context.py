import argparse
import json
import logging
from pathlib import Path

from VQAExp.cal_ocr_edit_distance import calc_edit_distance, load_gt_by_id, load_predictions_by_id


LOGGER = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate the added Pride long-context Qwen OCR predictions.")
    parser.add_argument("--root", type=Path, default=Path("aaai_resub/qwen_instability_intervention/pride_long_context"), help="Long-context experiment root.")
    return parser.parse_args()


def merge_predictions(paths: list[Path]) -> dict[str, dict]:
    merged = {}
    for path in paths:
        records = load_predictions_by_id(path)
        overlap = set(merged) & set(records)
        if overlap:
            raise ValueError(f"Duplicate prediction IDs across shards: {sorted(overlap)[:5]}")
        merged.update(records)
    return merged


def evaluate_group(gt_path: Path, prediction_paths: list[Path], output_path: Path, resolution: int) -> dict:
    gt_by_id = load_gt_by_id(gt_path, "context")
    predictions_by_id = merge_predictions(prediction_paths)
    missing = sorted(set(gt_by_id) - set(predictions_by_id))
    extra = sorted(set(predictions_by_id) - set(gt_by_id))
    if missing or extra:
        raise ValueError(f"ID mismatch for resolution={resolution}: missing={missing[:5]} extra={extra[:5]}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    finish_reasons = {}
    with output_path.open("w", encoding="utf-8") as file_handle:
        for sample_id, sample in sorted(gt_by_id.items()):
            prediction_record = predictions_by_id[sample_id]
            edit_raw, edit_dist, gt_len, prediction_len = calc_edit_distance(str(prediction_record["prediction"]), str(sample["context"]))
            finish_reason = str(prediction_record.get("finish_reason", ""))
            finish_reasons[finish_reason] = finish_reasons.get(finish_reason, 0) + 1
            result = {
                "id": sample_id,
                "layout": "28_6_0",
                "resolution": resolution,
                "edit_dist": edit_dist,
                "edit_raw": edit_raw,
                "gt_len": gt_len,
                "prediction_len": prediction_len,
                "text_len": sample.get("text_len"),
                "finish_reason": finish_reason,
                "max_tokens": prediction_record.get("max_tokens"),
            }
            file_handle.write(json.dumps(result, ensure_ascii=False) + "\n")

    summary = {
        "resolution": resolution,
        "num_gt": len(gt_by_id),
        "num_predictions": len(predictions_by_id),
        "num_evaluated": len(gt_by_id),
        "finish_reasons": finish_reasons,
        "output": str(output_path),
    }
    LOGGER.info("Evaluated resolution=%d samples=%d finish_reasons=%s", resolution, len(gt_by_id), finish_reasons)
    return summary


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    groups = [
        {
            "resolution": 1024,
            "gt": args.root / "data/pride_25k_34k_for_1036_gt.json",
            "predictions": [args.root / "ocr_predictions/1036/ocr_predictions_md_res.jsonl"],
        },
        {
            "resolution": 1280,
            "gt": args.root / "data/pride_25k_48k_for_1288_gt.json",
            "predictions": [
                args.root / "ocr_predictions/1288/slot0/ocr_predictions_md_res.jsonl",
                args.root / "ocr_predictions/1288/slot1/ocr_predictions_md_res.jsonl",
            ],
        },
    ]
    summaries = []
    for group in groups:
        resolution = int(group["resolution"])
        output_path = args.root / "evaluation" / f"ocr_edit_distance_res_{resolution}.jsonl"
        summaries.append(evaluate_group(group["gt"], group["predictions"], output_path, resolution))

    summary_path = args.root / "evaluation/summary.json"
    summary_path.write_text(json.dumps({"groups": summaries, "total_evaluated": sum(item["num_evaluated"] for item in summaries)}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    LOGGER.info("Saved evaluation summary to %s", summary_path)


if __name__ == "__main__":
    main()
