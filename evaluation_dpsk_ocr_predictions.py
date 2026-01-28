import argparse
import json
import logging
import math
import re
from pathlib import Path

import jieba
import nltk
from nltk.metrics import f_measure, precision, recall
from nltk.translate import meteor_score

from Levenshtein import distance as levenshtein_distance


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def contain_chinese_string(text):
    chinese_pattern = re.compile(r"[\u4e00-\u9fa5]")
    return bool(chinese_pattern.search(text))


def cal_per_metrics(pred, gt):
    metrics = {}

    if contain_chinese_string(gt) or contain_chinese_string(pred):
        reference = jieba.lcut(gt)
        hypothesis = jieba.lcut(pred)
    else:
        reference = gt.split()
        hypothesis = pred.split()

    metrics["bleu"] = nltk.translate.bleu([reference], hypothesis)

    # metrics["meteor"] = meteor_score.meteor_score([reference], hypothesis)

    reference_set = set(reference)
    hypothesis_set = set(hypothesis)

    metrics["f_measure"] = f_measure(reference_set, hypothesis_set)
    metrics["precision"] = precision(reference_set, hypothesis_set)
    metrics["recall"] = recall(reference_set, hypothesis_set)

    denom = max(len(pred), len(gt))
    if denom == 0:
        metrics["edit_dist"] = 0.0
    else:
        edit_raw = levenshtein_distance(pred, gt)
        metrics["edit_dist"] = edit_raw / denom
    return metrics


def _load_existing(path: Path) -> list:
    if not path.exists():
        return []
    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except json.JSONDecodeError:
        logger.warning("Existing metrics file is not valid JSON. Starting fresh.")
        return []
    if isinstance(data, list):
        return data
    logger.warning("Existing metrics file is not a list. Starting fresh.")
    return []


def _flush_metrics(metrics_records: list, output_path: Path, processed: int, total: int) -> None:
    if not metrics_records:
        return
    existing = _load_existing(output_path)
    existing.extend(metrics_records)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(existing, f, ensure_ascii=False, indent=2)
    metrics_records.clear()
    percent = (processed / total) * 100 if total else 100
    logger.info("Processed %d/%d predictions (%.1f%%). Saved checkpoint to %s", processed, total, percent, output_path)


def evaluate_predictions(input_file, output_file):
    with open(input_file, encoding="utf-8") as f:
        predictions = json.load(f)

    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        output_path.unlink()

    total = len(predictions)
    if total == 0:
        output_path.write_text("[]", encoding="utf-8")
        logger.info("No predictions found. Wrote empty list to %s", output_path)
        return

    checkpoints = sorted({min(total, math.ceil(total * i / 4)) for i in range(1, 5)})
    metrics_records = []
    for index, record in enumerate(predictions):
        pred_text = record.get("label", "")
        gt_text = record.get("answer", "")
        metrics = cal_per_metrics(pred_text, gt_text)

        text_length = record.get("text_token_len")
        if text_length is None:
            text_length = len(gt_text)
        metrics["text_token_len"] = text_length
        metrics["text_len"] = record.get("text_len")
        metrics["id"] = record.get("id")
        metrics_records.append(metrics)

        processed = index + 1
        if processed in checkpoints:
            _flush_metrics(metrics_records, output_path, processed, total)

    if metrics_records:
        _flush_metrics(metrics_records, output_path, total, total)


def main():
    parser = argparse.ArgumentParser(description="Evaluate OCR predictions without aggregation.")
    parser.add_argument("--json_for_eval", type=str, required=True, help="Path to OCR prediction JSON.") # tmp_eval.json
    parser.add_argument("--output_file", type=str, required=True, help="Path to store metrics JSON.") # pride_640_eval_dpsk_ocr.json
    args = parser.parse_args()

    evaluate_predictions(args.json_for_eval, args.output_file)


if __name__ == "__main__":
    main()
