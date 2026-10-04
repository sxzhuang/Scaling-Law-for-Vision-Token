import argparse
import csv
import json
import logging
import math
from pathlib import Path


LOGGER = logging.getLogger(__name__)
FONT_SIZES = [20, 28, 36]
LINE_SPACINGS = [0, 6, 24, 42]
LETTER_SPACINGS = [-1, 0, 7]
RESOLUTIONS = [512, 640, 1024, 1280]
OCR_GEOMETRY_FIELDS = [
    "image_id",
    "layout",
    "font_size",
    "line_spacing",
    "letter_spacing",
    "resolution",
    "input_W",
    "input_H",
    "qwen_W",
    "qwen_H",
    "qwen_factor",
    "qwen_vision_token_num",
    "raw_image_W",
    "raw_image_H",
    "character_width",
    "character_height",
    "qwen_character_width",
    "qwen_character_height",
    "qwen_letter_spacing",
    "qwen_line_spacing",
    "qwen_V",
    "qwen_G",
    "qwen_logV",
    "qwen_logG",
    "text_len",
]


def iter_layouts() -> list[str]:
    return [f"{font_size}_{line_spacing}_{letter_spacing}" for font_size in FONT_SIZES for line_spacing in LINE_SPACINGS for letter_spacing in LETTER_SPACINGS]


def load_ocr_geometry_rows(ocr_csv: Path) -> dict[tuple[str, int, str], dict]:
    rows = {}
    with ocr_csv.open("r", newline="", encoding="utf-8") as file_handle:
        reader = csv.DictReader(file_handle)
        missing = set(OCR_GEOMETRY_FIELDS) - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"OCR CSV is missing required fields: {sorted(missing)}")
        for row in reader:
            key = (row["layout"], int(row["resolution"]), row["image_id"])
            rows[key] = {field: row[field] for field in OCR_GEOMETRY_FIELDS}
    LOGGER.info("Loaded %d geometry rows from %s", len(rows), ocr_csv)
    return rows


def load_eval_details(eval_json: Path) -> list[dict]:
    with eval_json.open("r", encoding="utf-8") as file_handle:
        payload = json.load(file_handle)
    if isinstance(payload, dict) and isinstance(payload.get("details"), list):
        return payload["details"]
    if isinstance(payload, list):
        return payload
    raise ValueError(f"Unsupported eval JSON format: {eval_json}")


def build_rows(args: argparse.Namespace) -> list[dict]:
    geometry_rows = load_ocr_geometry_rows(args.ocr_csv)
    rows = []
    missing_geometry = 0
    for layout in iter_layouts():
        for resolution in args.resolutions:
            eval_json = args.eval_root / layout / f"retrieval_acc_res_{resolution}.jsonl"
            if not eval_json.is_file():
                LOGGER.warning("Skip missing Retrieval eval file: %s", eval_json)
                continue
            details = load_eval_details(eval_json)
            for detail in details:
                sample_id = str(detail["id"])
                geometry = geometry_rows.get((layout, int(resolution), sample_id))
                if geometry is None:
                    missing_geometry += 1
                    continue
                answers = detail.get("answers", [])
                num_gold_answers = len(answers)
                task_complexity = max(num_gold_answers, 1)
                contains_all = float(detail.get("contains_all", 0.0))
                correct = 1 if bool(detail.get("correct", contains_all == 1.0)) else 0
                row = dict(geometry)
                row.update(
                    {
                        "task": "retrieval",
                        "num_gold_answers": num_gold_answers,
                        "task_complexity": task_complexity,
                        "log_complexity": math.log(task_complexity),
                        "contains_all": contains_all,
                        "correct": correct,
                    }
                )
                rows.append(row)
            LOGGER.info("Loaded %d Retrieval eval rows from %s", len(details), eval_json)
    if missing_geometry:
        LOGGER.warning("Skipped %d Retrieval rows missing OCR geometry.", missing_geometry)
    return rows


def write_csv(rows: list[dict], output_csv: Path) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = OCR_GEOMETRY_FIELDS + ["task", "num_gold_answers", "task_complexity", "log_complexity", "contains_all", "correct"]
    with output_csv.open("w", newline="", encoding="utf-8") as file_handle:
        writer = csv.DictWriter(file_handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    LOGGER.info("Saved %d rows to %s", len(rows), output_csv)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build Qwen Retrieval scaling-law CSV by joining Retrieval accuracy with OCR geometry cache.")
    parser.add_argument("--ocr_csv", type=Path, default=Path("VQAExp/scaling_law/qwen_ocr_scaling_points.csv"), help="Qwen OCR scaling CSV used as Retrieval geometry cache.")
    parser.add_argument("--eval_root", type=Path, default=Path("VQAExp/eval_results/retrieval/qwen25vl7b"), help="Root directory containing Retrieval accuracy JSON files.")
    parser.add_argument("--output_csv", type=Path, default=Path("VQAExp/scaling_law/qwen_retrieval_scaling_points.csv"), help="Output CSV path.")
    parser.add_argument("--resolutions", type=int, nargs="+", default=RESOLUTIONS, help="Resolutions to include.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    rows = build_rows(args)
    if not rows:
        raise ValueError("No Retrieval scaling rows were built. Check eval_root and ocr_csv.")
    write_csv(rows, args.output_csv)


if __name__ == "__main__":
    main()
