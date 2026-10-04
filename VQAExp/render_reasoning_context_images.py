import argparse
import json
import logging
import re
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.generate_square_images import render_text_assets


LOGGER = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render cropped Reasoning context fields into JPG images.")
    parser.add_argument("--input_json", type=Path, default=Path("VQAExp/data/Reasoning/reasoning_cropped_0_25k_600.json"), help="Input JSON dataset with context fields.")
    parser.add_argument("--output_root", type=Path, default=Path("VQAExp/data"), help="Root directory for rendered image datasets.")
    parser.add_argument("--dataset_name", type=str, default="Reasoning", help="Dataset name used under output_root.")
    parser.add_argument("--font_path", type=Path, default=None, help="Optional path to a TTF/OTF font file.")
    parser.add_argument("--font_size", type=int, default=28, help="Font size for rendered text.")
    parser.add_argument("--line_spacing", type=int, default=6, help="Line spacing in pixels.")
    parser.add_argument("--letter_spacing", type=int, default=0, help="Letter spacing in pixels.")
    parser.add_argument("--padding", type=int, default=20, help="Pixel padding around rendered text.")
    parser.add_argument("--dpi", type=int, default=300, help="DPI metadata for saved JPGs.")
    parser.add_argument("--ratio_min", type=float, default=0.9, help="Minimum acceptable width/height ratio during layout search.")
    parser.add_argument("--ratio_max", type=float, default=1.1, help="Maximum acceptable width/height ratio during layout search.")
    parser.add_argument("--jpeg_quality", type=int, default=95, help="JPEG quality.")
    parser.add_argument("--limit", type=int, default=None, help="Optional maximum number of samples to render.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing JPG files.")
    return parser.parse_args()


def load_samples(input_json: Path) -> list[dict]:
    with input_json.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("Input JSON must contain a list of samples.")
    return data


def output_images_dir(output_root: Path, dataset_name: str, font_size: int, line_spacing: int, letter_spacing: int) -> Path:
    render_name = f"{font_size}_{line_spacing}_{letter_spacing}"
    return output_root / dataset_name / render_name / "images"


def sample_image_name(sample: dict, index: int) -> str:
    sample_id = str(sample.get("id") or f"sample_{index:06d}")
    return f"{sample_id}.jpg"


def normalize_context(context: str) -> str:
    return re.sub(r"\s+", " ", context).strip()


def render_sample(sample: dict, output_path: Path, args: argparse.Namespace) -> None:
    context = sample.get("context")
    if not isinstance(context, str) or not context:
        raise ValueError(f"Sample {sample.get('id')} does not have a non-empty context field.")
    context = normalize_context(context)
    if not context:
        raise ValueError(f"Sample {sample.get('id')} has an empty context after whitespace normalization.")

    render_result = render_text_assets(
        text=context,
        font_size=args.font_size,
        line_spacing=args.line_spacing,
        letter_spacing=args.letter_spacing,
        font_path=args.font_path,
        pdf_dpi=args.dpi,
        padding=args.padding,
        ratio_min=args.ratio_min,
        ratio_max=args.ratio_max,
    )
    image = render_result["image"].convert("RGB")
    image.save(output_path, "JPEG", quality=args.jpeg_quality, dpi=(args.dpi, args.dpi))


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    samples = load_samples(args.input_json)
    if args.limit is not None:
        samples = samples[:args.limit]

    images_dir = output_images_dir(args.output_root, args.dataset_name, args.font_size, args.line_spacing, args.letter_spacing)
    images_dir.mkdir(parents=True, exist_ok=True)

    rendered = 0
    skipped = 0
    for index, sample in enumerate(samples):
        image_path = images_dir / sample_image_name(sample, index)
        if image_path.exists() and not args.overwrite:
            skipped += 1
            continue
        render_sample(sample, image_path, args)
        rendered += 1
        if rendered % 25 == 0:
            LOGGER.info("Rendered %d images", rendered)

    LOGGER.info("Rendered %d images, skipped %d existing images", rendered, skipped)
    LOGGER.info("Images saved under %s", images_dir)


if __name__ == "__main__":
    main()
