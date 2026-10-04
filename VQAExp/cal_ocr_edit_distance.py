import argparse
import json
import logging
import re
from pathlib import Path

from Levenshtein import distance as levenshtein_distance


LOGGER = logging.getLogger(__name__)
RESOLUTION_PATTERN = re.compile(r"ocr_predictions_res_(\d+)\.jsonl$")
WHITESPACE_PATTERN = re.compile(r"\s+")


def normalize_whitespace(text: str) -> str:
    return WHITESPACE_PATTERN.sub(" ", text).strip()


def load_gt_by_id(gt_json: Path, context_field: str) -> dict[str, dict]:
    with gt_json.open("r", encoding="utf-8") as file_handle:
        samples = json.load(file_handle)
    if not isinstance(samples, list):
        raise ValueError(f"GT JSON must contain a list: {gt_json}")

    gt_by_id = {}
    for sample in samples:
        if not isinstance(sample, dict):
            continue
        sample_id = str(sample.get("id", "")).strip()
        if not sample_id:
            continue
        if context_field not in sample:
            raise ValueError(f"GT sample {sample_id} is missing {context_field} field.")
        gt_by_id[sample_id] = sample
    return gt_by_id


def load_predictions_by_id(predictions_jsonl: Path) -> dict[str, dict]:
    predictions_by_id = {}
    with predictions_jsonl.open("r", encoding="utf-8") as file_handle:
        for line_number, line in enumerate(file_handle, start=1):
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            sample_id = str(record.get("id", "")).strip()
            if not sample_id:
                raise ValueError(f"{predictions_jsonl}:{line_number} is missing id field.")
            if "prediction" not in record:
                raise ValueError(f"{predictions_jsonl}:{line_number} is missing prediction field.")
            predictions_by_id[sample_id] = record
    return predictions_by_id


def parse_resolution(predictions_jsonl: Path) -> int:
    match = RESOLUTION_PATTERN.match(predictions_jsonl.name)
    if match is None:
        raise ValueError(f"Cannot parse resolution from filename: {predictions_jsonl}")
    return int(match.group(1))


def calc_edit_distance(prediction: str, gt: str) -> tuple[int, float, int, int]:
    normalized_prediction = normalize_whitespace(prediction)
    normalized_gt = normalize_whitespace(gt)
    edit_raw = levenshtein_distance(normalized_prediction, normalized_gt)
    denom = max(len(normalized_prediction), len(normalized_gt))
    edit_dist = 0.0 if denom == 0 else edit_raw / denom
    return edit_raw, edit_dist, len(normalized_gt), len(normalized_prediction)


def evaluate_one_file(gt_by_id: dict[str, dict], predictions_jsonl: Path, output_jsonl: Path, context_field: str) -> dict:
    layout = predictions_jsonl.parent.name
    resolution = parse_resolution(predictions_jsonl)
    predictions_by_id = load_predictions_by_id(predictions_jsonl)
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)

    missing = []
    evaluated = 0
    extra_predictions = sorted(set(predictions_by_id) - set(gt_by_id))
    with output_jsonl.open("w", encoding="utf-8") as file_handle:
        for sample_id, sample in sorted(gt_by_id.items()):
            prediction_record = predictions_by_id.get(sample_id)
            if prediction_record is None:
                missing.append(sample_id)
                continue

            prediction = str(prediction_record["prediction"])
            gt_text = str(sample[context_field])
            edit_raw, edit_dist, gt_len, prediction_len = calc_edit_distance(prediction, gt_text)
            result = {
                "id": sample_id,
                "layout": layout,
                "resolution": resolution,
                "edit_dist": edit_dist,
                "edit_raw": edit_raw,
                "gt_len": gt_len,
                "prediction_len": prediction_len,
                "text_len": sample.get("text_len"),
            }
            file_handle.write(json.dumps(result, ensure_ascii=False) + "\n")
            evaluated += 1

    summary = {
        "layout": layout,
        "resolution": resolution,
        "num_gt": len(gt_by_id),
        "num_prediction_records": len(predictions_by_id),
        "num_evaluated": evaluated,
        "num_missing_predictions": len(missing),
        "num_extra_predictions": len(extra_predictions),
        "output_jsonl": str(output_jsonl),
    }
    LOGGER.info(
        "%s res_%d: total=%d success=%d missing=%d extra=%d output=%s",
        layout,
        resolution,
        len(gt_by_id),
        evaluated,
        len(missing),
        len(extra_predictions),
        output_jsonl,
    )
    return summary


def iter_prediction_files(predictions_root: Path, layouts: set[str] | None, resolutions: set[int] | None) -> list[Path]:
    prediction_files = []
    for predictions_jsonl in sorted(predictions_root.glob("*/ocr_predictions_res_*.jsonl")):
        layout = predictions_jsonl.parent.name
        resolution = parse_resolution(predictions_jsonl)
        if layouts is not None and layout not in layouts:
            continue
        if resolutions is not None and resolution not in resolutions:
            continue
        prediction_files.append(predictions_jsonl)
    return prediction_files


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Calculate OCR edit distance for VQAExp OCR prediction JSONL files.")
    parser.add_argument("--gt_json", type=Path, default=Path("VQAExp/data/Retrieval/retrieval_cropped_0_25k_600.json"), help="Ground-truth JSON containing id, context, and text_len fields.")
    parser.add_argument("--predictions_root", type=Path, default=Path("VQAExp/ocr_predictions/Qwen/Qwen2.5-VL-7B-Instruct/OCR"), help="Root directory containing layout OCR prediction JSONL files.")
    parser.add_argument("--output_root", type=Path, default=Path("VQAExp/eval_results/ocr/qwen25vl7b"), help="Root directory for per-layout edit distance JSONL files.")
    parser.add_argument("--context_field", type=str, default="context", help="GT text field used for OCR edit distance.")
    parser.add_argument("--layouts", type=str, nargs="*", default=None, help="Optional layout names to evaluate, such as 28_6_0.")
    parser.add_argument("--resolutions", type=int, nargs="*", default=None, help="Optional resolutions to evaluate, such as 512 640 1024 1280.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    if not args.gt_json.is_file():
        raise FileNotFoundError(f"GT JSON not found: {args.gt_json}")
    if not args.predictions_root.is_dir():
        raise FileNotFoundError(f"Predictions root not found: {args.predictions_root}")

    layouts = set(args.layouts) if args.layouts else None
    resolutions = set(args.resolutions) if args.resolutions else None
    gt_by_id = load_gt_by_id(args.gt_json, args.context_field)
    prediction_files = iter_prediction_files(args.predictions_root, layouts, resolutions)
    if not prediction_files:
        LOGGER.warning("No OCR prediction JSONL files found under %s", args.predictions_root)
        return

    total_evaluated = 0
    total_missing = 0
    for predictions_jsonl in prediction_files:
        layout = predictions_jsonl.parent.name
        resolution = parse_resolution(predictions_jsonl)
        output_jsonl = args.output_root / layout / f"ocr_edit_distance_res_{resolution}.jsonl"
        summary = evaluate_one_file(gt_by_id, predictions_jsonl, output_jsonl, args.context_field)
        total_evaluated += int(summary["num_evaluated"])
        total_missing += int(summary["num_missing_predictions"])

    LOGGER.info("Finished %d files. Total success=%d missing=%d", len(prediction_files), total_evaluated, total_missing)


if __name__ == "__main__":
    main()
