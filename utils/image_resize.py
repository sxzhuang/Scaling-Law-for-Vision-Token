import argparse
import logging
from pathlib import Path

from PIL import Image

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Resize JPG images to a square resolution.")
    parser.add_argument("--input_path", type=Path, required=True, help="Directory containing JPG images.")
    parser.add_argument("--output_path", type=Path, required=True, help="Directory to write resized JPG images.")
    parser.add_argument("--resolution", type=int, required=True, help="Target square resolution (pixels).")
    return parser.parse_args()


def resize_images(input_path: Path, output_path: Path, resolution: int) -> None:
    if not input_path.exists():
        raise FileNotFoundError(f"Input directory not found: {input_path}")
    if not input_path.is_dir():
        raise NotADirectoryError(f"Input path is not a directory: {input_path}")
    if resolution <= 0:
        raise ValueError("Resolution must be a positive integer.")
    if input_path.resolve() == output_path.resolve():
        raise ValueError("Output path must differ from input path to avoid overwriting images.")

    output_path.mkdir(parents=True, exist_ok=True)
    image_paths = sorted(
        path for path in input_path.iterdir() if path.is_file() and path.suffix.lower() == ".jpg"
    )
    if not image_paths:
        raise RuntimeError(f"No JPG images found in {input_path}")

    for image_path in image_paths:
        with Image.open(image_path) as image:
            resized = image.resize((resolution, resolution))
            resized.save(output_path / image_path.name)
    logger.info("Resized %d images to %dx%d in %s", len(image_paths), resolution, resolution, output_path)


def main() -> None:
    args = parse_args()
    resize_images(args.input_path, args.output_path, args.resolution)


if __name__ == "__main__":
    main()
