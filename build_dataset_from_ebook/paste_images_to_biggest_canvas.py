#!/usr/bin/env python3
"""Paste images that fit into the largest canvas determined from a folder."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Iterable

from PIL import Image

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Paste images that fit into the largest canvas from a reference folder.")
    parser.add_argument("--canvas_dir", type=Path, default=Path("build_dataset_from_ebook/pride_dataset_square_density_experiment/GroupA/image"), help="Directory used to determine the largest canvas size.")
    parser.add_argument("--input_dir", type=Path, default=Path("build_dataset_from_ebook/pride_dataset_square_density_experiment/GroupC/image"), help="Directory containing source images to paste.")
    parser.add_argument("--output_dir", type=Path, default=Path("build_dataset_from_ebook/pride_dataset_square_density_experiment/GroupC_paste2biggest/image"), help="Directory to save centered images.")
    return parser.parse_args()


def list_image_files(input_dir: Path) -> list[Path]:
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")
    image_paths = [p for p in input_dir.iterdir() if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}]
    if not image_paths:
        raise ValueError(f"No images found in {input_dir}")
    return sorted(image_paths)


def find_max_size(image_paths: Iterable[Path]) -> tuple[int, int]:
    max_width, max_height = 0, 0
    for path in image_paths:
        with Image.open(path) as img:
            width, height = img.size
        max_width = max(max_width, width)
        max_height = max(max_height, height)
    return max_width, max_height


def filter_images_by_canvas(image_paths: Iterable[Path], canvas_size: tuple[int, int]) -> tuple[list[Path], list[tuple[Path, int, int]]]:
    fit_paths: list[Path] = []
    skipped: list[tuple[Path, int, int]] = []
    canvas_width, canvas_height = canvas_size
    for path in image_paths:
        with Image.open(path) as img:
            width, height = img.size
        if width <= canvas_width and height <= canvas_height:
            fit_paths.append(path)
        else:
            skipped.append((path, width, height))
    return fit_paths, skipped


def paste_images(image_paths: Iterable[Path], canvas_size: tuple[int, int], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    canvas_width, canvas_height = canvas_size
    for path in image_paths:
        with Image.open(path) as img:
            base_image = img.convert("RGB")
        base_width, base_height = base_image.size
        offset_x = (canvas_width - base_width) // 2
        offset_y = (canvas_height - base_height) // 2
        canvas = Image.new("RGB", (canvas_width, canvas_height), "white")
        canvas.paste(base_image, (offset_x, offset_y))
        out_path = output_dir / path.name
        canvas.save(out_path)
        # logger.info("Saved %s", out_path)


def main() -> None:
    args = parse_args()
    canvas_paths = list_image_files(args.canvas_dir)
    # canvas_size = find_max_size(canvas_paths)
    canvas_size = (3584, 3584)  # Fixed canvas size
    logger.info("Canvas determined as %dx%d from %s", canvas_size[0], canvas_size[1], args.canvas_dir)

    input_paths = list_image_files(args.input_dir)
    fit_paths, skipped = filter_images_by_canvas(input_paths, canvas_size)
    if skipped:
        logger.info("Skipped %d images that exceed canvas: %s", len(skipped), ", ".join([p.name for p, _, _ in skipped]))
    if not fit_paths:
        logger.warning("No images fit within the canvas. Nothing to paste.")
        return

    paste_images(fit_paths, canvas_size, args.output_dir)
    logger.info("Finished saving %d images to %s", len(fit_paths), args.output_dir)


if __name__ == "__main__":
    main()
