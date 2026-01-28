#!/usr/bin/env python3
"""Render metadata texts into square JPGs for font/spacing combinations."""

from __future__ import annotations

import argparse
import json
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from itertools import product
from pathlib import Path

from utils.generate_square_images import render_text_assets

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render metadata texts into square JPGs for font/spacing combinations.")
    parser.add_argument("--metadata", type=Path, default=Path("build_dataset_from_ebook/pride_dataset_square/pride_square_gt.json"), help="Path to metadata JSON containing gt text and image names.")
    parser.add_argument("--output_root", type=Path, default=Path("build_dataset_from_ebook/pride_dataset_square_density_experiment"), help="Root directory to save rendered images.")
    parser.add_argument("--font_sizes", type=int, nargs="+", default=[20, 28, 36], help="List of font sizes, e.g., 24 28 32.")
    parser.add_argument("--line_spacings", type=int, nargs="+", default=[0, 6, 24, 42], help="List of line spacings.")
    parser.add_argument("--letter_spacings", type=int, nargs="+", default=[-1, 0, 7], help="List of letter spacings.")
    parser.add_argument("--font_path", type=Path, default=None, help="Optional path to a TTF/OTF font file.")
    parser.add_argument("--padding", type=int, default=20, help="Pixel padding around rendered text.")
    parser.add_argument("--dpi", type=int, default=300, help="DPI for saved JPGs.")
    parser.add_argument("--ratio_min", type=float, default=0.9, help="Minimum acceptable width/height ratio.")
    parser.add_argument("--ratio_max", type=float, default=1.1, help="Maximum acceptable width/height ratio.")
    parser.add_argument("--num_workers", type=int, default=8, help="Number of threads for parallel rendering.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing images instead of skipping.")
    return parser.parse_args()


def load_metadata(path: Path) -> list[tuple[str, str]]:
    records = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(records, list):
        raise ValueError("Metadata JSON must be a list of records.")
    items: list[tuple[str, str]] = []
    for record in records:
        if not isinstance(record, dict):
            continue
        image_name = str(record.get("image", "")).strip()
        text = str(record.get("gt", "")).strip()
        if not image_name or not text:
            continue
        items.append((image_name, text))
    if not items:
        raise RuntimeError("No valid entries found in metadata.")
    return items


def render_one(text: str, image_name: str, font_size: int, line_spacing: int, letter_spacing: int, output_root: Path, args: argparse.Namespace) -> Path | None:
    subdir = output_root / f"{font_size}_{line_spacing}_{letter_spacing}" / "images"
    subdir.mkdir(parents=True, exist_ok=True)
    image_path = subdir / image_name
    if image_path.exists() and not args.overwrite:
        logger.debug("Skipping existing file %s", image_path)
        return None
    result = render_text_assets(
        text=text,
        font_size=font_size,
        line_spacing=line_spacing,
        letter_spacing=letter_spacing,
        font_path=args.font_path,
        pdf_dpi=args.dpi,
        padding=args.padding,
        ratio_min=args.ratio_min,
        ratio_max=args.ratio_max,
    )
    image = result["image"].convert("RGB")
    image.save(image_path, "JPEG", quality=95, dpi=(args.dpi, args.dpi))
    return image_path


def main() -> None:
    args = parse_args()
    entries = load_metadata(args.metadata)
    combos = list(product(args.font_sizes, args.line_spacings, args.letter_spacings))
    logger.info("Loaded %d metadata entries.", len(entries))
    logger.info("Rendering %d combinations.", len(combos))

    tasks = []
    for image_name, text in entries:
        for font_size, line_spacing, letter_spacing in combos:
            tasks.append((image_name, text, font_size, line_spacing, letter_spacing))

    rendered = 0
    skipped = 0
    with ThreadPoolExecutor(max_workers=max(1, args.num_workers)) as executor:
        future_map = {
            executor.submit(
                render_one,
                text,
                image_name,
                font_size,
                line_spacing,
                letter_spacing,
                args.output_root,
                args,
            ): (image_name, font_size, line_spacing, letter_spacing)
            for image_name, text, font_size, line_spacing, letter_spacing in tasks
        }
        for future in as_completed(future_map):
            image_path = future.result()
            if image_path is None:
                skipped += 1
            else:
                rendered += 1

    logger.info("Rendering complete. Created %d images, skipped %d existing.", rendered, skipped)


if __name__ == "__main__":
    main()
