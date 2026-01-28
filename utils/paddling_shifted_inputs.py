#!/usr/bin/env python3
"""Generate shifted versions of images on padded canvases."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from PIL import Image

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
LOGGER = logging.getLogger(__name__)


def generate_shifted_images(folder: Path, resolution: int, stride: int = 2) -> None:
    """Create shifted variants of images inside *folder*.

    Args:
        folder: Directory containing input JPG/PNG images.
        resolution: Target square resolution for the padded canvas.
        stride: Step size, in pixels, for both horizontal and vertical shifts.
    """

    if resolution <= 16:
        raise ValueError("resolution must be greater than 16 to allow padding.")
    if stride <= 0:
        raise ValueError("stride must be a positive integer.")

    inner_size = resolution - 16
    image_paths = [
        path
        for path in sorted(folder.iterdir())
        if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png"}
    ]

    if not image_paths:
        LOGGER.warning("No image files (.jpg/.jpeg/.png) found under %s", folder)
        return

    LOGGER.info(
        "Processing %d images at resolution %d with stride %d",
        len(image_paths),
        resolution,
        stride,
    )

    positions = sorted({*range(0, 17, stride), 16})
    output_folder = folder.parent / f"{folder.name}_paddling_shift"
    output_folder.mkdir(parents=True, exist_ok=True)
    LOGGER.info("Saving shifted images to %s", output_folder)

    for image_path in image_paths:
        with Image.open(image_path) as image:
            resized = image.resize((inner_size, inner_size))
        base_name = image_path.stem
        for offset_x in positions:
            for offset_y in positions:
                canvas = Image.new("RGB", (resolution, resolution), color="white")
                canvas.paste(resized, (offset_x, offset_y))
                output_name = f"{base_name}_{offset_x}_{offset_y}.jpg"
                output_path = output_folder / output_name
                canvas.save(output_path, "JPEG", quality=95)
                LOGGER.debug("Saved %s", output_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pad and shift images to a fixed canvas.")
    parser.add_argument("--folder", type=Path, default=Path("build_dataset_from_ebook/pride_dataset_square_large_variance/selected_from1024/images"), help="Directory that holds source images.")
    parser.add_argument("--resolution", type=int, required=True, help="Target square resolution for outputs.")
    parser.add_argument("--stride", type=int, default=4, help="Shift step for both axes in pixels.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    generate_shifted_images(args.folder, args.resolution, args.stride)


if __name__ == "__main__":
    main()
