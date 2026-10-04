#!/usr/bin/env python3
"""Build Chinese analysis CSV for EM fitting from metadata and eval JSON files."""

from __future__ import annotations

import argparse
import csv
import json
import logging
import re
from pathlib import Path

from PIL import ImageFont

DEFAULT_METADATA_PATH = Path("build_dataset_from_ebook/jianai_dataset_square/jinaia_square_ge.json")
DEFAULT_EVAL_ROOT = Path("eval_results/jianai_dataset_square")
DEFAULT_OUTPUT_CSV = Path("data_analysis/data4analysis_chinese.csv")
DEFAULT_RESOLUTIONS = "512,640"
DEFAULT_FONT_SIZE = 28
DEFAULT_LINE_SPACING = 6
DEFAULT_LETTER_SPACING = 0
DEFAULT_FONT_PATH = Path("build_dataset_from_ebook/fonts/SourceHanSansSC-Regular.otf")
DEFAULT_FONT_SIZES = "20,28,36"
DEFAULT_LINE_SPACINGS = "0,6,24,42"
DEFAULT_LETTER_SPACINGS = "-1,0,7"
DEFAULT_GRID_METADATA_ROOT = Path("appendix_exp/jianai_dataset_square")
DEFAULT_GRID_EVAL_ROOT = Path("appendix_exp/eval_results/jianai_dataset_square")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build Chinese CSV fields required by EM fitting.")
    parser.add_argument("--metadata_json", type=Path, default=DEFAULT_METADATA_PATH, help="Path to metadata JSON containing gt/text_len/text_token_len.")
    parser.add_argument("--eval_root", type=Path, default=DEFAULT_EVAL_ROOT, help="Directory containing resolution subfolders with eval JSON files.")
    parser.add_argument("--resolutions", type=str, default=DEFAULT_RESOLUTIONS, help="Comma-separated resolutions to include, e.g. 512,640.")
    parser.add_argument("--output_csv", type=Path, default=DEFAULT_OUTPUT_CSV, help="Output CSV path.")
    parser.add_argument("--font_path", type=Path, default=DEFAULT_FONT_PATH, help="Font used to estimate Chinese glyph size.")
    parser.add_argument("--font_size", type=int, default=DEFAULT_FONT_SIZE, help="Fixed font_size value written to CSV.")
    parser.add_argument("--line_spacing", type=int, default=DEFAULT_LINE_SPACING, help="Fixed line_spacing value written to CSV.")
    parser.add_argument("--letter_spacing", type=int, default=DEFAULT_LETTER_SPACING, help="Fixed letter_spacing value written to CSV.")
    parser.add_argument("--max_unique_chars", type=int, default=4096, help="Maximum unique Chinese chars sampled for width/height estimation.")
    parser.add_argument("--grid_mode", action="store_true", help="Build CSV by traversing appendix grid layout folders.")
    parser.add_argument("--grid_metadata_root", type=Path, default=DEFAULT_GRID_METADATA_ROOT, help="Root with <font>_<line>_<letter>/jianai_square_gt.json metadata files.")
    parser.add_argument("--grid_eval_root", type=Path, default=DEFAULT_GRID_EVAL_ROOT, help="Root with <font>_<line>_<letter>/<resolution>/eval JSON files.")
    parser.add_argument("--font_sizes", type=str, default=DEFAULT_FONT_SIZES, help="Comma-separated font sizes for grid mode.")
    parser.add_argument("--line_spacings", type=str, default=DEFAULT_LINE_SPACINGS, help="Comma-separated line spacings for grid mode.")
    parser.add_argument("--letter_spacings", type=str, default=DEFAULT_LETTER_SPACINGS, help="Comma-separated letter spacings for grid mode.")
    return parser.parse_args()


def load_json(path: Path) -> object:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def parse_resolutions(resolutions_arg: str) -> list[int]:
    resolutions: list[int] = []
    for token in resolutions_arg.split(","):
        token = token.strip()
        if not token:
            continue
        try:
            resolutions.append(int(token))
        except ValueError as exc:
            raise ValueError(f"Invalid resolution: {token}") from exc
    if not resolutions:
        raise ValueError("No valid resolutions provided.")
    return resolutions


def parse_int_list(values_arg: str, name: str) -> list[int]:
    values: list[int] = []
    for token in values_arg.split(","):
        token = token.strip()
        if not token:
            continue
        try:
            values.append(int(token))
        except ValueError as exc:
            raise ValueError(f"Invalid {name}: {token}") from exc
    if not values:
        raise ValueError(f"No valid {name} provided.")
    return values


def resolve_metadata_path(path: Path) -> Path:
    if path.exists():
        return path
    candidates = [
        path.with_name("jianai_square_gt.json"),
        Path("build_dataset_from_ebook/jianai_dataset_square/jianai_square_gt.json"),
    ]
    for candidate in candidates:
        if candidate.exists():
            logging.warning("metadata_json %s not found; fallback to %s", path, candidate)
            return candidate
    raise FileNotFoundError(f"Metadata JSON not found: {path}")


def load_font(font_path: Path | None, font_size: int) -> ImageFont.ImageFont:
    if font_path is not None and font_path.exists():
        return ImageFont.truetype(str(font_path), font_size)
    try:
        return ImageFont.truetype("DejaVuSans.ttf", font_size)
    except OSError:
        return ImageFont.load_default()


def build_meta_map(meta_data: object) -> dict[str, dict]:
    if not isinstance(meta_data, list):
        raise ValueError("metadata_json must contain a list.")
    meta_map: dict[str, dict] = {}
    for entry in meta_data:
        if not isinstance(entry, dict):
            continue
        image_name = entry.get("image")
        if not isinstance(image_name, str) or not image_name.strip():
            continue
        image_id = Path(image_name).stem
        meta_map[image_id] = entry
    return meta_map


def collect_chinese_chars(meta_map: dict[str, dict], max_unique_chars: int) -> list[str]:
    unique_chars: set[str] = set()
    for entry in meta_map.values():
        text = entry.get("gt", "")
        if not isinstance(text, str):
            continue
        for char in re.findall(r"[\u4e00-\u9fff]", text):
            unique_chars.add(char)
            if len(unique_chars) >= max_unique_chars:
                return sorted(unique_chars)
    return sorted(unique_chars)


def estimate_character_size(chars: list[str], font: ImageFont.ImageFont) -> tuple[float, float]:
    widths: list[float] = []
    heights: list[float] = []
    sample_chars = chars if chars else ["汉"]
    for char in sample_chars:
        bbox = font.getbbox(char)
        width = float(max(bbox[2] - bbox[0], 0))
        height = float(max(bbox[3] - bbox[1], 0))
        if width > 0 and height > 0:
            widths.append(width)
            heights.append(height)
    if not widths or not heights:
        fallback = font.getbbox("汉")
        fallback_w = float(max(fallback[2] - fallback[0], 1))
        fallback_h = float(max(fallback[3] - fallback[1], 1))
        return fallback_w, fallback_h
    return float(sum(widths) / len(widths)), float(sum(heights) / len(heights))


def to_int(value: object) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def to_float(value: object) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def build_entry_map(data: object) -> dict[str, dict]:
    if isinstance(data, list):
        entries = data
    elif isinstance(data, dict):
        entries = list(data.values())
    else:
        return {}
    result: dict[str, dict] = {}
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        sample_id = entry.get("id")
        if not isinstance(sample_id, str) or not sample_id.strip():
            continue
        result[sample_id.strip()] = entry
    return result


def merge_rows_for_resolution(
    resolution: int,
    metric_map: dict[str, dict],
    prepare_map: dict[str, dict],
    meta_map: dict[str, dict],
    font_size: int,
    line_spacing: int,
    letter_spacing: int,
    character_width: float,
    character_height: float,
) -> list[dict[str, object]]:
    merged_rows: list[dict[str, object]] = []
    all_ids = sorted(set(metric_map.keys()) | set(prepare_map.keys()) | set(meta_map.keys()))
    for sample_id in all_ids:
        metric_entry = metric_map.get(sample_id, {})
        prepare_entry = prepare_map.get(sample_id, {})
        meta_entry = meta_map.get(sample_id, {})
        text_len = to_int(metric_entry.get("text_len"))
        if text_len is None:
            text_len = to_int(prepare_entry.get("text_len"))
        if text_len is None and isinstance(meta_entry, dict):
            text_len = to_int(meta_entry.get("text_len"))
        edit_distance = to_float(metric_entry.get("edit_distance"))
        if edit_distance is None:
            edit_distance = to_float(metric_entry.get("edit_dist"))
        if text_len is None or edit_distance is None:
            continue
        text_token_len = to_int(metric_entry.get("text_token_len"))
        if text_token_len is None:
            text_token_len = to_int(prepare_entry.get("text_token_len"))
        if text_token_len is None and isinstance(meta_entry, dict):
            text_token_len = to_int(meta_entry.get("text_token_len"))
        merged_rows.append(
            {
                "image_id": sample_id,
                "resolution": int(resolution),
                "font_size": int(font_size),
                "line_spacing": int(line_spacing),
                "letter_spacing": int(letter_spacing),
                "text_len": int(text_len),
                "text_token_len": text_token_len,
                "character_width": float(character_width),
                "character_height": float(character_height),
                "edit_distance": float(edit_distance),
            }
        )
    return merged_rows


def write_csv(rows: list[dict[str, object]], output_csv: Path) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "image_id",
        "resolution",
        "font_size",
        "line_spacing",
        "letter_spacing",
        "text_len",
        "text_token_len",
        "character_width",
        "character_height",
        "edit_distance",
    ]
    with output_csv.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_rows_for_layout(
    metadata_path: Path,
    eval_dir: Path,
    resolution: int,
    font_size: int,
    line_spacing: int,
    letter_spacing: int,
    font_path: Path,
    max_unique_chars: int,
) -> list[dict[str, object]]:
    if not metadata_path.exists():
        logging.warning("Skipping %s/%s: missing metadata %s", metadata_path.parent.name, resolution, metadata_path)
        return []
    metric_path = eval_dir / "dpsk_eval_metric.json"
    prepare_path = eval_dir / "prepare_json_for_eval.json"
    if not metric_path.exists() or not prepare_path.exists():
        logging.warning(
            "Skipping %s/%s: missing files metric=%s prepare=%s",
            metadata_path.parent.name,
            resolution,
            metric_path.exists(),
            prepare_path.exists(),
        )
        return []

    meta_data = load_json(metadata_path)
    meta_map = build_meta_map(meta_data)
    font = load_font(font_path, font_size)
    sampled_chars = collect_chinese_chars(meta_map, max_unique_chars)
    character_width, character_height = estimate_character_size(sampled_chars, font)
    metric_map = build_entry_map(load_json(metric_path))
    prepare_map = build_entry_map(load_json(prepare_path))
    rows = merge_rows_for_resolution(
        resolution=resolution,
        metric_map=metric_map,
        prepare_map=prepare_map,
        meta_map=meta_map,
        font_size=font_size,
        line_spacing=line_spacing,
        letter_spacing=letter_spacing,
        character_width=character_width,
        character_height=character_height,
    )
    logging.info(
        "Layout %s @ %s: char_width=%.4f char_height=%.4f metric=%d prepare=%d merged_rows=%d",
        metadata_path.parent.name,
        resolution,
        character_width,
        character_height,
        len(metric_map),
        len(prepare_map),
        len(rows),
    )
    return rows


def build_grid_rows(args: argparse.Namespace, resolutions: list[int]) -> list[dict[str, object]]:
    font_sizes = parse_int_list(args.font_sizes, "font_size")
    line_spacings = parse_int_list(args.line_spacings, "line_spacing")
    letter_spacings = parse_int_list(args.letter_spacings, "letter_spacing")
    all_rows: list[dict[str, object]] = []
    for font_size in font_sizes:
        for line_spacing in line_spacings:
            for letter_spacing in letter_spacings:
                layout_name = f"{font_size}_{line_spacing}_{letter_spacing}"
                metadata_path = args.grid_metadata_root / layout_name / "jianai_square_gt.json"
                for resolution in resolutions:
                    eval_dir = args.grid_eval_root / layout_name / str(resolution)
                    all_rows.extend(
                        build_rows_for_layout(
                            metadata_path=metadata_path,
                            eval_dir=eval_dir,
                            resolution=resolution,
                            font_size=font_size,
                            line_spacing=line_spacing,
                            letter_spacing=letter_spacing,
                            font_path=args.font_path,
                            max_unique_chars=args.max_unique_chars,
                        )
                    )
    return all_rows


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    resolutions = parse_resolutions(args.resolutions)

    if args.grid_mode:
        all_rows = build_grid_rows(args, resolutions)
        all_rows.sort(
            key=lambda row: (
                int(row["resolution"]),
                int(row["font_size"]),
                int(row["line_spacing"]),
                int(row["letter_spacing"]),
                str(row["image_id"]),
            )
        )
        if not all_rows:
            raise RuntimeError("No rows generated. Check grid input paths and eval files.")
        write_csv(all_rows, args.output_csv)
        logging.info("Wrote %d grid rows to %s", len(all_rows), args.output_csv)
        return

    metadata_path = resolve_metadata_path(args.metadata_json)
    meta_data = load_json(metadata_path)
    meta_map = build_meta_map(meta_data)
    logging.info("Loaded metadata entries: %d from %s", len(meta_map), metadata_path)

    font = load_font(args.font_path, args.font_size)
    sampled_chars = collect_chinese_chars(meta_map, args.max_unique_chars)
    character_width, character_height = estimate_character_size(sampled_chars, font)
    logging.info(
        "Estimated Chinese char size from %d chars: width=%.4f height=%.4f",
        len(sampled_chars),
        character_width,
        character_height,
    )

    all_rows: list[dict[str, object]] = []
    for resolution in resolutions:
        resolution_dir = args.eval_root / str(resolution)
        metric_path = resolution_dir / "dpsk_eval_metric.json"
        prepare_path = resolution_dir / "prepare_json_for_eval.json"
        if not metric_path.exists() or not prepare_path.exists():
            logging.warning(
                "Skipping resolution %s: missing files metric=%s prepare=%s",
                resolution,
                metric_path.exists(),
                prepare_path.exists(),
            )
            continue
        metric_map = build_entry_map(load_json(metric_path))
        prepare_map = build_entry_map(load_json(prepare_path))
        rows = merge_rows_for_resolution(
            resolution=resolution,
            metric_map=metric_map,
            prepare_map=prepare_map,
            meta_map=meta_map,
            font_size=args.font_size,
            line_spacing=args.line_spacing,
            letter_spacing=args.letter_spacing,
            character_width=character_width,
            character_height=character_height,
        )
        logging.info(
            "Resolution %s: metric=%d prepare=%d merged_rows=%d",
            resolution,
            len(metric_map),
            len(prepare_map),
            len(rows),
        )
        all_rows.extend(rows)

    all_rows.sort(key=lambda row: (int(row["resolution"]), str(row["image_id"])))
    if not all_rows:
        raise RuntimeError("No rows generated. Check input paths and JSON content.")
    write_csv(all_rows, args.output_csv)
    logging.info("Wrote %d rows to %s", len(all_rows), args.output_csv)


if __name__ == "__main__":
    main()
