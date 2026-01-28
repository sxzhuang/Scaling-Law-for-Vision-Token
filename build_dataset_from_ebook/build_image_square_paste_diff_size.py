#!/usr/bin/env python3
"""Render specific GT ids and paste them onto larger canvases derived from existing metadata sizes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable

import logging
from PIL import Image

from utils.generate_square_images import render_text_assets

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render ids and paste them onto all larger canvas sizes from pride_square_gt metadata.")
    parser.add_argument(
        "--ids",
        type=str,
        nargs="+",
        default=["pride_prejudice_double_C14B1_C20B4", "pride_prejudice_double_C20B1_C25B3", "pride_prejudice_double_C6B2_C40B1"],
        help="List of ids to render (e.g., pride_prejudice_double_C14B1_C20B4).",
    )
    parser.add_argument("--metadata", type=Path, default=Path("build_dataset_from_ebook/pride_dataset_square/pride_square_gt.json"), help="Path to pride_square_gt.json.")
    parser.add_argument("--output_dir", type=Path, default=Path("build_dataset_from_ebook/pride_dataset_square_paste2larger/images"), help="Directory to save pasted images.")
    parser.add_argument("--font_path", type=Path, default=None, help="Optional TTF/OTF font path.")
    parser.add_argument("--font_size", type=int, default=28, help="Font size (matches test.py default).")
    parser.add_argument("--line_spacing", type=int, default=24, help="Line spacing (matches test.py default).")
    parser.add_argument("--letter_spacing", type=int, default=6, help="Letter spacing (matches test.py default).")
    parser.add_argument("--padding", type=int, default=20, help="Padding around text.")
    parser.add_argument("--dpi", type=int, default=300, help="DPI for PDF rendering (unused for paste but kept for parity).")
    parser.add_argument("--ratio_min", type=float, default=0.9, help="Minimum width/height ratio.")
    parser.add_argument("--ratio_max", type=float, default=1.1, help="Maximum width/height ratio.")
    return parser.parse_args()


def load_metadata(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Metadata file not found: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"{path} must be a list of records.")
    return data


def load_gt_text(records: Iterable[dict[str, Any]], sample_id: str) -> str:
    target = Path(sample_id.strip()).stem
    for record in records:
        if not isinstance(record, dict):
            continue
        image_name = str(record.get("image", "")).strip()
        if not image_name:
            continue
        if Path(image_name).stem == target:
            gt_text = str(record.get("gt", "")).strip()
            if not gt_text:
                raise ValueError(f"Found {target} but gt is empty.")
            return gt_text
    raise KeyError(f"id not found in metadata: {target}")


def collect_canvas_sizes(records: Iterable[dict[str, Any]]) -> set[tuple[int, int]]:
    sizes: set[tuple[int, int]] = set()
    for record in records:
        if not isinstance(record, dict):
            continue
        width = record.get("width")
        height = record.get("height")
        if isinstance(width, (int, float)) and isinstance(height, (int, float)):
            w_i, h_i = int(width), int(height)
            if w_i > 0 and h_i > 0:
                sizes.add((w_i, h_i))
    return sizes


def filter_sizes_for_image(canvas_sizes: set[tuple[int, int]], img_width: int, img_height: int) -> list[tuple[int, int]]:
    candidates = [(w, h) for w, h in canvas_sizes if w >= img_width and h >= img_height]
    candidates.sort(key=lambda item: (item[0] * item[1], item[0], item[1]))
    return candidates


def paste_center(base_image: Image.Image, target_size: tuple[int, int]) -> Image.Image:
    target_w, target_h = target_size
    canvas = Image.new("RGB", (target_w, target_h), "white")
    base_w, base_h = base_image.size
    offset = ((target_w - base_w) // 2, (target_h - base_h) // 2)
    canvas.paste(base_image, offset)
    return canvas


def main() -> None:
    args = parse_args()
    records = load_metadata(args.metadata)
    canvas_sizes = collect_canvas_sizes(records)
    if not canvas_sizes:
        raise RuntimeError("No available width/height entries in metadata. Please populate sizes first.")

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    total_ids = len(args.ids)
    for idx, sample_id in enumerate(args.ids, start=1):
        logger.info("Processing %s (%d/%d)", sample_id, idx, total_ids)
        gt_text = load_gt_text(records, sample_id)
        render_result = render_text_assets(
            text=gt_text,
            font_size=args.font_size,
            line_spacing=args.line_spacing,
            letter_spacing=args.letter_spacing,
            font_path=args.font_path,
            pdf_dpi=args.dpi,
            padding=args.padding,
            ratio_min=args.ratio_min,
            ratio_max=args.ratio_max,
        )
        base_image = render_result["image"].convert("RGB")
        base_w, base_h = base_image.size
        candidates = filter_sizes_for_image(canvas_sizes, base_w, base_h)
        if not candidates:
            logger.info("[skip] %s has no fitting canvas (rendered %dx%d).", sample_id, base_w, base_h)
            continue

        for target_w, target_h in candidates:
            canvas = paste_center(base_image, (target_w, target_h))
            out_name = f"{Path(sample_id).stem}_{target_w}_{target_h}.jpg"
            out_path = output_dir / out_name
            canvas.save(out_path, "JPEG", quality=95, dpi=(args.dpi, args.dpi))
        logger.info("[done] %s produced %d images (base %dx%d).", sample_id, len(candidates), base_w, base_h)


if __name__ == "__main__":
    main()
