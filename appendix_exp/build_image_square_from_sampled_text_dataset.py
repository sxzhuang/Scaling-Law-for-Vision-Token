#!/usr/bin/env python3
"""Render appendix sampled text records into square-ish JPG images.

This script keeps the image-generation path from
build_dataset_from_ebook/build_image_square_from_json_languages.py, but uses
pre-sampled text records instead of sampling blocks from the source book JSON.
It writes JPG files only; no PDFs are generated.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import random
import re
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageFont

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.generate_square_images import (
    MAX_WRAP_WIDTH,
    MIN_WRAP_WIDTH,
    adjust_wrap_width,
    attempt_layout,
    estimate_image_size,
    precise_line_width,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

WRAP_WIDTH_BY_BLOCKS = {1: 70, 2: 120, 3: 200, 4: 220}


@dataclass(frozen=True)
class TextSample:
    sample_id: str
    gt: str
    text_len: int
    group_label: str
    blocks: list[str]


@dataclass(frozen=True)
class CharacterMetrics:
    width: float
    height: float
    sample_count: int


class WrapHistory:
    """Store successful (text_len, wrap_width) pairs to match the source renderer API."""

    def __init__(self) -> None:
        self.records: list[tuple[int, int]] = []

    def add(self, text_len: int, wrap_width: int) -> None:
        self.records.append((text_len, wrap_width))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render pre-sampled appendix text records into square-ish JPG images.")
    parser.add_argument("--dataset", type=Path, default=Path("appendix_exp/text_samples/jianai_sampled_text_dataset.json"), help="Path to the sampled text dataset JSON.")
    parser.add_argument("--output_dir", type=Path, default=Path("appendix_exp/jianai_dataset_square"), help="Root directory where parameter-specific JPG images and metadata JSON are stored.")
    parser.add_argument("--font_path", type=Path, default=None, help="Optional path to a TTF/OTF font file. Defaults to DejaVuSans or PIL font.")
    parser.add_argument("--font_size", type=int, default=28, help="Font size for rendered text.")
    parser.add_argument("--padding", type=int, default=20, help="Pixel padding around rendered text.")
    parser.add_argument("--wrap_width", type=int, default=60, help="Fallback character count per line when no preset applies.")
    parser.add_argument("--dpi", type=int, default=300, help="DPI metadata used when saving JPGs.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducible character-metric sampling.")
    parser.add_argument("--ratio_min", type=float, default=0.9, help="Minimum acceptable width/height ratio.")
    parser.add_argument("--ratio_max", type=float, default=1.1, help="Maximum acceptable width/height ratio.")
    parser.add_argument("--metadata_name", type=str, default="jianai_square_gt.json", help="Filename for the output metadata JSON.")
    parser.add_argument("--line_spacing", type=int, default=6, help="Line spacing (pixels) used during rendering.")
    parser.add_argument("--letter_spacing", type=int, default=0, help="Letter spacing (pixels) used during rendering.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing JPG files instead of reusing them.")
    parser.add_argument("--limit", type=int, default=0, help="Optional limit for quick smoke tests. A value <= 0 renders all records.")
    return parser.parse_args()


def load_samples(path: Path) -> list[TextSample]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError("Sampled text dataset JSON must contain a list of records.")

    samples: list[TextSample] = []
    for idx, record in enumerate(data):
        if not isinstance(record, dict):
            continue
        gt = str(record.get("gt", "")).strip()
        if not gt:
            continue
        sample_id = str(record.get("sample_id", f"sample_{idx:06d}")).strip() or f"sample_{idx:06d}"
        safe_sample_id = re.sub(r"[^0-9A-Za-z_.-]+", "_", sample_id)
        samples.append(
            TextSample(
                sample_id=safe_sample_id,
                gt=gt,
                text_len=int(record.get("text_len", len(gt))),
                group_label=str(record.get("group_label", "")),
                blocks=[str(block) for block in record.get("blocks", [])],
            )
        )
    return samples


def load_font(font_path: Path | None, font_size: int) -> ImageFont.ImageFont:
    if font_path is not None:
        return ImageFont.truetype(str(font_path), font_size)
    try:
        return ImageFont.truetype("DejaVuSans.ttf", font_size)
    except OSError:
        return ImageFont.load_default()


def initial_wrap_width_from_metrics(text: str, char_width: float, char_height: float) -> int:
    normalized = re.sub(r"\s+", " ", text.strip())
    if not normalized:
        return MIN_WRAP_WIDTH
    char_count = len(normalized)
    safe_width = max(char_width, 1e-6)
    safe_height = max(char_height, 1e-6)
    approx = int(round(math.sqrt((char_count * safe_height) / safe_width)))
    return max(MIN_WRAP_WIDTH, min(MAX_WRAP_WIDTH, approx or MIN_WRAP_WIDTH))


def estimate_image_size_multilingual(
    text: str,
    font: ImageFont.ImageFont,
    line_spacing: int,
    letter_spacing: int,
    padding: int,
    ratio_min: float,
    ratio_max: float,
    char_width: float,
    char_height: float,
    max_layout_attempts: int = 3,
) -> dict:
    normalized_text = text.strip()
    wrap_width = initial_wrap_width_from_metrics(normalized_text, char_width, char_height)
    best_result: dict | None = None
    best_delta = float("inf")
    tried: set[int] = set()

    for _ in range(max_layout_attempts):
        tried.add(wrap_width)
        result, ratio = attempt_layout(text=normalized_text, wrap_width=wrap_width, font=font, line_spacing=line_spacing, letter_spacing=letter_spacing, padding=padding)
        delta = abs(1.0 - ratio)
        if delta < best_delta:
            best_delta = delta
            best_result = result
        if ratio_min <= ratio <= ratio_max:
            return result
        next_wrap = adjust_wrap_width(wrap_width, ratio, ratio_min=ratio_min, ratio_max=ratio_max)
        if next_wrap in tried or next_wrap == wrap_width:
            break
        wrap_width = next_wrap

    if best_result is None:
        raise RuntimeError("Unable to generate multilingual layout for the provided text.")
    return best_result


def estimate_sample_character_metrics(
    text: str,
    font: ImageFont.ImageFont,
    font_size: int,
    line_spacing: int,
    letter_spacing: int,
    padding: int,
    ratio_min: float,
    ratio_max: float,
) -> tuple[float, float]:
    size_result = estimate_image_size(text=text, font_size=font_size, line_spacing=line_spacing, letter_spacing=letter_spacing, font=font, padding=padding, ratio_min=ratio_min, ratio_max=ratio_max)
    lines = size_result["lines"]
    non_empty_lines = [line for line in lines if line]
    if non_empty_lines:
        total_chars = sum(len(line) for line in non_empty_lines)
        total_width = sum(precise_line_width(line, font, letter_spacing) for line in non_empty_lines)
    else:
        normalized = re.sub(r"\s+", " ", text.strip())
        total_chars = max(len(normalized), 1)
        total_width = float(max(size_result["actual_size"][0] - padding * 2, 1))
    char_width = max(total_width / max(total_chars, 1), 1e-6)
    line_count = max(1, len(lines))
    usable_height = max(size_result["actual_size"][1] - padding * 2, 1)
    char_height = max(usable_height / line_count, 1e-6)
    return char_width, char_height


def estimate_dataset_character_metrics(
    samples: Sequence[TextSample],
    rng: random.Random,
    font: ImageFont.ImageFont,
    font_size: int,
    line_spacing: int,
    letter_spacing: int,
    padding: int,
    ratio_min: float,
    ratio_max: float,
) -> CharacterMetrics:
    sample_count = min(3, len(samples))
    sampled_indices = list(range(len(samples))) if len(samples) <= 3 else rng.sample(range(len(samples)), sample_count)
    widths: list[float] = []
    heights: list[float] = []
    for idx in sampled_indices:
        char_width, char_height = estimate_sample_character_metrics(
            text=samples[idx].gt,
            font=font,
            font_size=font_size,
            line_spacing=line_spacing,
            letter_spacing=letter_spacing,
            padding=padding,
            ratio_min=ratio_min,
            ratio_max=ratio_max,
        )
        widths.append(char_width)
        heights.append(char_height)
    if not widths or not heights:
        raise RuntimeError("Unable to estimate character metrics from sampled text records.")
    return CharacterMetrics(width=sum(widths) / len(widths), height=sum(heights) / len(heights), sample_count=len(sampled_indices))


def render_squareish_image(
    text: str,
    font: ImageFont.ImageFont,
    padding: int,
    line_spacing: int,
    letter_spacing: int,
    text_len: int,
    base_wrap: int,
    history: WrapHistory,
    ratio_min: float,
    ratio_max: float,
    character_metrics: CharacterMetrics,
) -> tuple[Image.Image, int, float]:
    size_result = estimate_image_size_multilingual(
        text=text,
        font=font,
        line_spacing=line_spacing,
        letter_spacing=letter_spacing,
        padding=padding,
        ratio_min=ratio_min,
        ratio_max=ratio_max,
        char_width=character_metrics.width,
        char_height=character_metrics.height,
    )
    image = size_result["image"]
    width, height = size_result["actual_size"]
    aspect_ratio = (width / height) if height else 0.0
    history.add(text_len, base_wrap)
    return image, base_wrap, aspect_ratio


def save_jpg(image: Image.Image, image_path: Path, dpi: int) -> None:
    image_path.parent.mkdir(parents=True, exist_ok=True)
    image.convert("RGB").save(image_path, "JPEG", quality=95, dpi=(dpi, dpi))


def remove_unused_files(directory: Path, keep: set[str]) -> int:
    removed = 0
    for path in directory.glob("*.jpg"):
        if path.is_file() and path.name not in keep:
            path.unlink()
            removed += 1
    return removed


def layout_output_dir(args: argparse.Namespace) -> Path:
    return args.output_dir / f"{args.font_size}_{args.line_spacing}_{args.letter_spacing}"


def render_samples(args: argparse.Namespace, samples: Sequence[TextSample], font: ImageFont.ImageFont, character_metrics: CharacterMetrics, image_dir: Path) -> tuple[list[dict], set[str]]:
    image_dir.mkdir(parents=True, exist_ok=True)
    history = WrapHistory()
    metadata: list[dict] = []
    used_jpgs: set[str] = set()

    for idx, sample in enumerate(samples, start=1):
        jpg_name = f"{sample.sample_id}.jpg"
        jpg_path = image_dir / jpg_name
        used_jpgs.add(jpg_name)
        block_count = len(sample.blocks)
        dynamic_wrap = WRAP_WIDTH_BY_BLOCKS.get(block_count, args.wrap_width)
        image, wrap_used, aspect_ratio = render_squareish_image(
            text=sample.gt,
            font=font,
            padding=args.padding,
            line_spacing=args.line_spacing,
            letter_spacing=args.letter_spacing,
            text_len=sample.text_len,
            base_wrap=dynamic_wrap,
            history=history,
            ratio_min=args.ratio_min,
            ratio_max=args.ratio_max,
            character_metrics=character_metrics,
        )
        if args.overwrite or not jpg_path.exists():
            save_jpg(image=image, image_path=jpg_path, dpi=args.dpi)
        else:
            logger.info("Skipping existing image for %s", sample.sample_id)

        metadata_record = {
            "image": jpg_name,
            "gt": sample.gt,
            "text_len": len(sample.gt),
            "wrap_width": wrap_used,
            "aspect_ratio": aspect_ratio,
            "group_label": sample.group_label,
            "blocks": sample.blocks,
        }
        metadata.append(metadata_record)
        if idx % 50 == 0 or idx == len(samples):
            logger.info("Rendered %d/%d images.", idx, len(samples))

    return metadata, used_jpgs


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)
    config_dir = layout_output_dir(args)
    config_dir.mkdir(parents=True, exist_ok=True)

    samples = load_samples(args.dataset)
    if args.limit > 0:
        samples = samples[: args.limit]
    if not samples:
        raise RuntimeError("No non-empty sampled text records found.")

    image_dir = config_dir / "images"
    existing_jpgs = {path.name for path in image_dir.glob("*.jpg")} if image_dir.exists() else set()
    font = load_font(args.font_path, args.font_size)
    character_metrics = estimate_dataset_character_metrics(
        samples=samples,
        rng=rng,
        font=font,
        font_size=args.font_size,
        line_spacing=args.line_spacing,
        letter_spacing=args.letter_spacing,
        padding=args.padding,
        ratio_min=args.ratio_min,
        ratio_max=args.ratio_max,
    )
    logger.info("Estimated character metrics from %d sampled records: width=%.4f, height=%.4f", character_metrics.sample_count, character_metrics.width, character_metrics.height)

    metadata, used_jpgs = render_samples(args=args, samples=samples, font=font, character_metrics=character_metrics, image_dir=image_dir)
    output_json = config_dir / args.metadata_name
    output_json.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")

    jpg_new = len(used_jpgs - existing_jpgs)
    jpg_reused = len(used_jpgs & existing_jpgs)
    removed_jpgs = remove_unused_files(image_dir, used_jpgs)
    logger.info("Image files - new: %d, reused/skipped: %d, removed stale: %d", jpg_new, jpg_reused, removed_jpgs)
    logger.info("Created %d square-ish JPG files under %s and wrote metadata to %s", len(metadata), image_dir, output_json)


if __name__ == "__main__":
    main()

# Run:
# python appendix_exp/build_image_square_from_sampled_text_dataset.py --font_path build_dataset_from_ebook/fonts/SourceHanSansSC-Regular.otf
