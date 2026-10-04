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
INTERNVL_TILE_SIZE = 448
INTERNVL_PATCH_SIZE = 14
INTERNVL_DOWNSAMPLE_RATIO = 0.5
INTERNVL_MIN_DYNAMIC_PATCH = 1
INTERNVL_MAX_DYNAMIC_PATCH = 12
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


def find_closest_aspect_ratio(aspect_ratio: float, target_ratios: list[tuple[int, int]], width: int, height: int, image_size: int) -> tuple[int, int]:
    best_ratio_diff = float("inf")
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio


def internvl_dynamic_tiles(width: int, height: int, min_num: int, max_num: int, image_size: int, use_thumbnail: bool) -> dict:
    if width <= 0 or height <= 0:
        raise ValueError(f"Invalid image size: {width}x{height}")
    aspect_ratio = width / height
    target_ratios = set((i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if i * j <= max_num and i * j >= min_num)
    target_ratios = sorted(target_ratios, key=lambda item: item[0] * item[1])
    tile_cols, tile_rows = find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size)
    detail_tile_num = tile_cols * tile_rows
    thumbnail_tile_num = 1 if use_thumbnail and detail_tile_num != 1 else 0
    total_tile_num = detail_tile_num + thumbnail_tile_num
    return {
        "tile_cols": tile_cols,
        "tile_rows": tile_rows,
        "detail_tile_num": detail_tile_num,
        "thumbnail_tile_num": thumbnail_tile_num,
        "total_tile_num": total_tile_num,
        "detail_canvas_w": tile_cols * image_size,
        "detail_canvas_h": tile_rows * image_size,
    }


def internvl_tokens_per_tile(tile_size: int, patch_size: int, downsample_ratio: float) -> int:
    patch_grid = tile_size // patch_size
    return int(round((patch_grid * patch_grid) * (downsample_ratio * downsample_ratio)))


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


def build_row(record: dict, layout: str, resolution: int, data_root: Path, tile_size: int, patch_size: int, downsample_ratio: float, min_dynamic_patch: int, max_dynamic_patch: int, use_thumbnail: bool, image_size_cache: dict[Path, tuple[int, int]]) -> dict:
    sample_id = str(record["id"])
    font_size, line_spacing, letter_spacing = parse_layout(layout)
    base_character_width, base_character_height = CHARACTER_SIZE_BY_FONT[font_size]
    raw_image_path = data_root / layout / "images" / f"{sample_id}.jpg"
    input_image_path = data_root / layout / f"images_{resolution}" / f"{sample_id}.jpg"
    if not raw_image_path.is_file():
        raise FileNotFoundError(f"Raw rendered image not found: {raw_image_path}")
    if not input_image_path.is_file():
        raise FileNotFoundError(f"InternVL input image not found: {input_image_path}")

    raw_image_w, raw_image_h = get_image_size(raw_image_path, image_size_cache)
    input_w, input_h = get_image_size(input_image_path, image_size_cache)
    tile_info = internvl_dynamic_tiles(input_w, input_h, min_dynamic_patch, max_dynamic_patch, tile_size, use_thumbnail)
    tokens_per_tile = internvl_tokens_per_tile(tile_size, patch_size, downsample_ratio)
    detail_vision_token_num = tile_info["detail_tile_num"] * tokens_per_tile
    thumbnail_vision_token_num = tile_info["thumbnail_tile_num"] * tokens_per_tile
    total_vision_token_num = tile_info["total_tile_num"] * tokens_per_tile

    internvl_character_width = base_character_width * tile_info["detail_canvas_w"] / raw_image_w
    internvl_character_height = base_character_height * tile_info["detail_canvas_h"] / raw_image_h
    internvl_letter_spacing = letter_spacing * tile_info["detail_canvas_w"] / raw_image_w
    internvl_line_spacing = line_spacing * tile_info["detail_canvas_h"] / raw_image_h
    internvl_v_denominator = internvl_character_height + internvl_line_spacing
    internvl_v = ((internvl_character_width + internvl_letter_spacing) * float(record["text_len"])) / internvl_v_denominator if internvl_v_denominator != 0 else float("nan")
    internvl_g = float(record["text_len"]) / float(total_vision_token_num)

    return {
        "image_id": sample_id,
        "layout": layout,
        "font_size": font_size,
        "line_spacing": line_spacing,
        "letter_spacing": letter_spacing,
        "resolution": resolution,
        "input_W": input_w,
        "input_H": input_h,
        "raw_image_W": raw_image_w,
        "raw_image_H": raw_image_h,
        "character_width": base_character_width,
        "character_height": base_character_height,
        "internvl_tile_size": tile_size,
        "internvl_patch_size": patch_size,
        "internvl_downsample_ratio": downsample_ratio,
        "internvl_min_dynamic_patch": min_dynamic_patch,
        "internvl_max_dynamic_patch": max_dynamic_patch,
        "internvl_use_thumbnail": int(use_thumbnail),
        "internvl_tile_cols": tile_info["tile_cols"],
        "internvl_tile_rows": tile_info["tile_rows"],
        "internvl_detail_tile_num": tile_info["detail_tile_num"],
        "internvl_thumbnail_tile_num": tile_info["thumbnail_tile_num"],
        "internvl_total_tile_num": tile_info["total_tile_num"],
        "internvl_detail_canvas_W": tile_info["detail_canvas_w"],
        "internvl_detail_canvas_H": tile_info["detail_canvas_h"],
        "internvl_tokens_per_tile": tokens_per_tile,
        "internvl_detail_vision_token_num": detail_vision_token_num,
        "internvl_thumbnail_vision_token_num": thumbnail_vision_token_num,
        "internvl_vision_token_num": total_vision_token_num,
        "internvl_character_width": internvl_character_width,
        "internvl_character_height": internvl_character_height,
        "internvl_letter_spacing": internvl_letter_spacing,
        "internvl_line_spacing": internvl_line_spacing,
        "internvl_V": internvl_v,
        "internvl_G": internvl_g,
        "internvl_logV": math.log(max(internvl_v, 1e-12)),
        "internvl_logG": math.log(max(internvl_g, 1e-12)),
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
                        args.tile_size,
                        args.patch_size,
                        args.downsample_ratio,
                        args.min_dynamic_patch,
                        args.max_dynamic_patch,
                        args.use_thumbnail,
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
        "raw_image_W",
        "raw_image_H",
        "character_width",
        "character_height",
        "internvl_tile_size",
        "internvl_patch_size",
        "internvl_downsample_ratio",
        "internvl_min_dynamic_patch",
        "internvl_max_dynamic_patch",
        "internvl_use_thumbnail",
        "internvl_tile_cols",
        "internvl_tile_rows",
        "internvl_detail_tile_num",
        "internvl_thumbnail_tile_num",
        "internvl_total_tile_num",
        "internvl_detail_canvas_W",
        "internvl_detail_canvas_H",
        "internvl_tokens_per_tile",
        "internvl_detail_vision_token_num",
        "internvl_thumbnail_vision_token_num",
        "internvl_vision_token_num",
        "internvl_character_width",
        "internvl_character_height",
        "internvl_letter_spacing",
        "internvl_line_spacing",
        "internvl_V",
        "internvl_G",
        "internvl_logV",
        "internvl_logG",
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
    parser = argparse.ArgumentParser(description="Build an InternVL-specific OCR scaling-law CSV from VQAExp edit-distance JSONL files.")
    parser.add_argument("--eval_root", type=Path, default=Path("VQAExp/eval_results/ocr/internvl3_5_8b"), help="Root directory containing InternVL OCR edit-distance JSONL files.")
    parser.add_argument("--data_root", type=Path, default=Path("VQAExp/data/Retrieval"), help="Root directory containing rendered and resized Retrieval images.")
    parser.add_argument("--output_csv", type=Path, default=Path("VQAExp/scaling_law/internvl3_5_8b_ocr_scaling_points.csv"), help="Output CSV path.")
    parser.add_argument("--resolutions", type=int, nargs="+", default=RESOLUTIONS, help="Resolutions to include.")
    parser.add_argument("--tile_size", type=int, default=INTERNVL_TILE_SIZE, help="InternVL tile size / force_image_size.")
    parser.add_argument("--patch_size", type=int, default=INTERNVL_PATCH_SIZE, help="InternVL vision patch size.")
    parser.add_argument("--downsample_ratio", type=float, default=INTERNVL_DOWNSAMPLE_RATIO, help="InternVL pixel-shuffle downsample ratio.")
    parser.add_argument("--min_dynamic_patch", type=int, default=INTERNVL_MIN_DYNAMIC_PATCH, help="Minimum number of dynamic image patches.")
    parser.add_argument("--max_dynamic_patch", type=int, default=INTERNVL_MAX_DYNAMIC_PATCH, help="Maximum number of dynamic image patches.")
    parser.add_argument("--use_thumbnail", action=argparse.BooleanOptionalAction, default=True, help="Whether InternVL appends a thumbnail tile when detail tiles > 1.")
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
