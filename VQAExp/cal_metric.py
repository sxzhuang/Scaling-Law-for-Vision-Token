import argparse
import json
import logging
import re
from pathlib import Path


LOGGER = logging.getLogger(__name__)
THINK_TAG_PATTERN = re.compile(r"<think>.*?</think>", flags=re.DOTALL | re.IGNORECASE)
RESOLUTION_PATTERN = re.compile(r"res[_-]?(\d+)")


def remove_think_tags(text: str) -> str:
    return THINK_TAG_PATTERN.sub("", text)


def normalize_text(text: str) -> str:
    return remove_think_tags(text).strip().lower()


def calc_contains_all(response: str, gold_answers: list[str]) -> float:
    normalized_response = normalize_text(response)
    normalized_answers = [normalize_text(str(answer)) for answer in gold_answers]
    if not normalized_answers:
        return 0.0
    matched = sum(1 for answer in normalized_answers if answer in normalized_response)
    return matched / len(normalized_answers)


def infer_resolution(predictions_jsonl: Path, output_prefix: str) -> str:
    for value in (predictions_jsonl.stem, output_prefix):
        match = RESOLUTION_PATTERN.search(value)
        if match:
            return match.group(1)
    raise ValueError("Could not infer resolution from predictions filename or output_prefix. Use --resolution.")


def load_gt_by_id(json_path: Path) -> dict[str, dict]:
    with json_path.open("r", encoding="utf-8") as f:
        samples = json.load(f)
    if not isinstance(samples, list):
        raise ValueError("GT JSON must contain a list of samples.")

    by_id = {}
    for sample in samples:
        sample_id = str(sample.get("id", "")).strip()
        if not sample_id:
            continue
        if "answers" not in sample:
            raise ValueError(f"Sample {sample_id} is missing answers field.")
        by_id[sample_id] = sample
    return by_id


def load_predictions_by_id(jsonl_path: Path) -> dict[str, dict]:
    predictions = {}
    with jsonl_path.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            sample_id = str(record.get("id", "")).strip()
            if not sample_id:
                raise ValueError(f"Prediction JSONL line {line_number} is missing id field.")
            if "prediction" not in record:
                raise ValueError(f"Prediction JSONL line {line_number} is missing prediction field.")
            predictions[sample_id] = record
    return predictions


def evaluate(gt_by_id: dict[str, dict], predictions_by_id: dict[str, dict]) -> tuple[dict, list[dict]]:
    details = []
    missing_predictions = []
    correct = 0
    evaluated = 0

    for sample_id, sample in sorted(gt_by_id.items()):
        prediction_record = predictions_by_id.get(sample_id)
        if prediction_record is None:
            missing_predictions.append(sample_id)
            continue

        prediction = str(prediction_record["prediction"])
        contains_all = calc_contains_all(prediction, sample["answers"])
        is_correct = contains_all == 1.0
        correct += int(is_correct)
        evaluated += 1

        details.append(
            {
                "id": sample_id,
                "answers": sample["answers"],
                "prediction": prediction.strip(),
                "contains_all": contains_all,
                "correct": is_correct,
                "text_len": sample.get("text_len"),
            }
        )

    summary = {
        "num_gt": len(gt_by_id),
        "num_evaluated": evaluated,
        "num_missing_predictions": len(missing_predictions),
        "accuracy_contains_all": correct / evaluated if evaluated else 0.0,
        "num_correct": correct,
        "missing_prediction_ids": missing_predictions,
    }
    return summary, details


def save_outputs(summary: dict, details: list[dict], output_dir: Path, metric_name: str, resolution: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    details_path = output_dir / f"{metric_name}_res_{resolution}.jsonl"
    output = {
        "summary": summary,
        "details": details,
    }
    details_path.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    LOGGER.info("Saved details to %s", details_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate VQA predictions with VTCBench-style contains_all accuracy.")
    parser.add_argument("--gt_json", type=Path, default=Path("VQAExp/data/Retrieval/retrieval_cropped_0_25k_600.json"), help="Ground-truth JSON file containing id and answers.")
    parser.add_argument("--predictions_jsonl", type=Path, required=True, help="JSONL file containing id and prediction fields.")
    parser.add_argument("--output_dir", type=Path, default=Path("VQAExp/eval_results"), help="Directory to save evaluation results.")
    parser.add_argument("--output_prefix", type=str, default="retrieval_vqa", help="Prefix for saved result files.")
    parser.add_argument("--resolution", type=str, default=None, help="Image resolution used in the prediction file. Inferred from filename by default.")
    parser.add_argument("--metric_name", type=str, default="retrieval_acc", help="Metric name used in the output filename.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    if not args.predictions_jsonl.is_file():
        raise FileNotFoundError(f"Predictions JSONL not found: {args.predictions_jsonl}")

    gt_by_id = load_gt_by_id(args.gt_json)
    predictions_by_id = load_predictions_by_id(args.predictions_jsonl)
    summary, details = evaluate(gt_by_id, predictions_by_id)
    resolution = args.resolution or infer_resolution(args.predictions_jsonl, args.output_prefix)
    save_outputs(summary, details, args.output_dir, args.metric_name, resolution)
    LOGGER.info("Evaluated %d samples, accuracy_contains_all=%.6f", summary["num_evaluated"], summary["accuracy_contains_all"])


if __name__ == "__main__":
    main()


# python VQAExp/cal_metric.py \
# --gt_json VQAExp/data/Retrieval/retrieval_cropped_0_25k_600.json \
# --predictions_jsonl VQAExp/vqa_predictions/Qwen/Qwen2.5-VL-7B-Instruct/Retrieval/28_6_0/vqa_predictions_res_640.jsonl \
# --output_dir VQAExp/eval_results/retrieval/qwen25vl7b/28_6_0 \
# --metric_name retrieval_acc
#
# Output:
# VQAExp/eval_results/retrieval/qwen25vl7b/28_6_0/retrieval_acc_res_640.jsonl
