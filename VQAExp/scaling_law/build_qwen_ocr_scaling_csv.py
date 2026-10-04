import argparse
import csv
import json
import logging
import math
from pathlib import Path

from PIL import Image


LOGGER = logging.getLogger(__name__)
FONT_SIZES = [20, 28, 36]
LINE_SPACINGS = [0, 6, 24, 42]
LETTER_SPACINGS = [-1, 0, 7]
RESOLUTIONS = [512, 640, 1024, 1280]
QWEN_FACTOR = 28
CHARACTER_SIZE_BY_FONT = {
    20: (10.0369, 19.0),
    28: (14.0533, 26.0),
    36: (18.0624, 33.0),
}


def iter_layouts() -> list[str]:
    return [f"{font_size}_{line_spacing}_{letter_spacing}" for font_size in FONT_SIZES for line_spacing in LINE_SPACINGS for letter_spacing in LETTER_SPACINGS]


def parse_layout(layout: str) -> tuple[int, int, int]:
    parts = layout.split("_")
    if len(parts) != 3:
        raise ValueError(f"Invalid layout name: {layout}")
    return int(parts[0]), int(parts[1]), int(parts[2])


def round_to_multiple(value: int, factor: int) -> int:
    return max(factor, int(round(float(value) / float(factor))) * factor)


def floor_to_multiple(value: float, factor: int) -> int:
    return max(factor, int(math.floor(value / float(factor))) * factor)


def ceil_to_multiple(value: float, factor: int) -> int:
    return max(factor, int(math.ceil(value / float(factor))) * factor)


def qwen_smart_resize(height: int, width: int, factor: int, min_pixels: int | None, max_pixels: int | None) -> tuple[int, int]:
    if height <= 0 or width <= 0:
        raise ValueError(f"Invalid image size: {width}x{height}")
    resized_height = round_to_multiple(height, factor)
    resized_width = round_to_multiple(width, factor)
    if max_pixels is not None and resized_height * resized_width > max_pixels:
        scale = math.sqrt((height * width) / float(max_pixels))
        resized_height = floor_to_multiple(height / scale, factor)
        resized_width = floor_to_multiple(width / scale, factor)
    if min_pixels is not None and resized_height * resized_width < min_pixels:
        scale = math.sqrt(float(min_pixels) / (height * width))
        resized_height = ceil_to_multiple(height * scale, factor)
        resized_width = ceil_to_multiple(width * scale, factor)
    return resized_height, resized_width


def load_jsonl_by_id(jsonl_path: Path) -> dict[str, dict]:
    records = {}
    with jsonl_path.open("r", encoding="utf-8") as file_handle:
        for line_number, line in enumerate(file_handle, start=1):
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            sample_id = str(record.get("id", "")).strip()
            if not sample_id:
                raise ValueError(f"{jsonl_path}:{line_number} is missing id field.")
            records[sample_id] = record
    return records


def get_image_size(image_path: Path, cache: dict[Path, tuple[int, int]]) -> tuple[int, int]:
    if image_path in cache:
        return cache[image_path]
    with Image.open(image_path) as image:
        size = image.size
    cache[image_path] = size
    return size


def build_row(
    record: dict,
    layout: str,
    resolution: int,
    data_root: Path,
    factor: int,
    min_pixels: int | None,
    max_pixels: int | None,
    image_size_cache: dict[Path, tuple[int, int]],
) -> dict:
    sample_id = str(record["id"])
    font_size, line_spacing, letter_spacing = parse_layout(layout)
    base_character_width, base_character_height = CHARACTER_SIZE_BY_FONT[font_size]
    raw_image_path = data_root / layout / "images" / f"{sample_id}.jpg"
    input_image_path = data_root / layout / f"images_{resolution}" / f"{sample_id}.jpg"
    if not raw_image_path.is_file():
        raise FileNotFoundError(f"Raw rendered image not found: {raw_image_path}")
    if not input_image_path.is_file():
        raise FileNotFoundError(f"Qwen input image not found: {input_image_path}")

    raw_image_w, raw_image_h = get_image_size(raw_image_path, image_size_cache)
    input_w, input_h = get_image_size(input_image_path, image_size_cache)
    qwen_h, qwen_w = qwen_smart_resize(input_h, input_w, factor, min_pixels, max_pixels)
    qwen_vision_token_num = (qwen_w // factor) * (qwen_h // factor)

    qwen_character_width = base_character_width * qwen_w / raw_image_w
    qwen_character_height = base_character_height * qwen_h / raw_image_h
    qwen_letter_spacing = letter_spacing * qwen_w / raw_image_w
    qwen_line_spacing = line_spacing * qwen_h / raw_image_h
    qwen_v_denominator = qwen_character_height + qwen_line_spacing
    qwen_v = ((qwen_character_width + qwen_letter_spacing) * float(record["text_len"])) / qwen_v_denominator if qwen_v_denominator != 0 else float("nan")
    qwen_g = float(record["text_len"]) / float(qwen_vision_token_num)

    return {
        "image_id": sample_id,
        "layout": layout,
        "font_size": font_size,
        "line_spacing": line_spacing,
        "letter_spacing": letter_spacing,
        "resolution": resolution,
        "input_W": input_w,
        "input_H": input_h,
        "qwen_W": qwen_w,
        "qwen_H": qwen_h,
        "qwen_factor": factor,
        "qwen_vision_token_num": qwen_vision_token_num,
        "raw_image_W": raw_image_w,
        "raw_image_H": raw_image_h,
        "character_width": base_character_width,
        "character_height": base_character_height,
        "qwen_character_width": qwen_character_width,
        "qwen_character_height": qwen_character_height,
        "qwen_letter_spacing": qwen_letter_spacing,
        "qwen_line_spacing": qwen_line_spacing,
        "qwen_V": qwen_v,
        "qwen_G": qwen_g,
        "qwen_logV": math.log(max(qwen_v, 1e-12)),
        "qwen_logG": math.log(max(qwen_g, 1e-12)),
        "text_len": int(record["text_len"]),
        "gt_len": int(record["gt_len"]),
        "prediction_len": int(record["prediction_len"]),
        "edit_raw": int(record["edit_raw"]),
        "edit_distance": float(record["edit_dist"]),
    }


def build_rows(args: argparse.Namespace) -> list[dict]:
    rows = []
    image_size_cache: dict[Path, tuple[int, int]] = {}
    for layout in iter_layouts():
        for resolution in args.resolutions:
            edit_distance_jsonl = args.eval_root / layout / f"ocr_edit_distance_res_{resolution}.jsonl"
            if not edit_distance_jsonl.is_file():
                LOGGER.warning("Skip missing edit-distance JSONL: %s", edit_distance_jsonl)
                continue
            records_by_id = load_jsonl_by_id(edit_distance_jsonl)
            for sample_id in sorted(records_by_id):
                rows.append(
                    build_row(
                        records_by_id[sample_id],
                        layout,
                        resolution,
                        args.data_root,
                        args.qwen_factor,
                        args.min_pixels,
                        args.max_pixels,
                        image_size_cache,
                    )
                )
            LOGGER.info("Loaded %d rows from %s", len(records_by_id), edit_distance_jsonl)
    return rows


def write_csv(rows: list[dict], output_csv: Path) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
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
        "gt_len",
        "prediction_len",
        "edit_raw",
        "edit_distance",
    ]
    with output_csv.open("w", newline="", encoding="utf-8") as file_handle:
        writer = csv.DictWriter(file_handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    LOGGER.info("Saved %d rows to %s", len(rows), output_csv)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a Qwen-specific OCR scaling-law CSV from VQAExp edit-distance JSONL files.")
    parser.add_argument("--eval_root", type=Path, default=Path("VQAExp/eval_results/ocr/qwen25vl7b"), help="Root directory containing OCR edit-distance JSONL files.")
    parser.add_argument("--data_root", type=Path, default=Path("VQAExp/data/Retrieval"), help="Root directory containing rendered and resized Retrieval images.")
    parser.add_argument("--output_csv", type=Path, default=Path("VQAExp/scaling_law/qwen_ocr_scaling_points.csv"), help="Output CSV path.")
    parser.add_argument("--resolutions", type=int, nargs="+", default=RESOLUTIONS, help="Resolutions to include.")
    parser.add_argument("--qwen_factor", type=int, default=QWEN_FACTOR, help="Qwen image resize factor.")
    parser.add_argument("--min_pixels", type=int, default=None, help="Optional Qwen min_pixels constraint. Leave unset to only round to qwen_factor.")
    parser.add_argument("--max_pixels", type=int, default=None, help="Optional Qwen max_pixels constraint. Leave unset to only round to qwen_factor.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    rows = build_rows(args)
    if not rows:
        raise ValueError("No rows were built. Check eval_root and data_root.")
    write_csv(rows, args.output_csv)


if __name__ == "__main__":
    main()
