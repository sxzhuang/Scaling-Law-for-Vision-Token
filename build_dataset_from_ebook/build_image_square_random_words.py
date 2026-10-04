#!/usr/bin/env python3
"""Build random-word-perturbed square metadata and render fixed-width images."""

from __future__ import annotations

import argparse
import json
import logging
import random
import re
import string
import textwrap
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Randomly mutate words in gt and render fixed-width images.")
    parser.add_argument("--input_json", type=Path, default=Path("build_dataset_from_ebook/pride_dataset_square/pride_square_gt.json"), help="Path to source pride_square_gt.json.")
    parser.add_argument("--output_json", type=Path, default=Path("build_dataset_from_ebook/pride_dataset_square_random/pride_square_gt.json"), help="Path to output mutated pride_square_gt.json.")
    parser.add_argument("--images_dir", type=Path, default=Path("build_dataset_from_ebook/pride_dataset_square_random/images"), help="Directory for rendered output images.")
    parser.add_argument("--font_path", type=Path, default=None, help="Optional path to a TTF/OTF font file.")
    parser.add_argument("--font_size", type=int, default=28, help="Font size for rendering.")
    parser.add_argument("--padding", type=int, default=20, help="Image padding in pixels.")
    parser.add_argument("--line_spacing", type=int, default=6, help="Line spacing in pixels.")
    parser.add_argument("--dpi", type=int, default=300, help="DPI for saved JPG images.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for deterministic mutations.")
    return parser.parse_args()


def load_records(path: Path) -> list[dict]:
    records = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(records, list):
        raise ValueError("Input JSON must be a list.")
    return records


def load_font(font_path: Path | None, font_size: int) -> ImageFont.ImageFont:
    if font_path is not None:
        return ImageFont.truetype(str(font_path), font_size)
    try:
        return ImageFont.truetype("DejaVuSans.ttf", font_size)
    except OSError:
        return ImageFont.load_default()


def line_width(text: str, font: ImageFont.ImageFont) -> float:
    if hasattr(font, "getlength"):
        return float(font.getlength(text))
    bbox = font.getbbox(text or " ")
    return float(bbox[2] - bbox[0])


def draw_justified_line(draw: ImageDraw.ImageDraw, line: str, font: ImageFont.ImageFont, x: int, y: int, max_width: float) -> None:
    stripped = line.rstrip()
    if not stripped:
        draw.text((x, y), line, font=font, fill="black")
        return
    words = stripped.split()
    if len(words) == 1:
        draw.text((x, y), stripped, font=font, fill="black")
        return
    space_width = line_width(" ", font)
    total_word_width = sum(line_width(word, font) for word in words)
    gaps = len(words) - 1
    base_line_width = total_word_width + space_width * gaps
    extra_space = max(max_width - base_line_width, 0.0)
    per_gap = space_width + (extra_space / gaps if gaps else 0.0)

    cursor = float(x)
    for idx, word in enumerate(words):
        draw.text((cursor, y), word, font=font, fill="black")
        cursor += line_width(word, font)
        if idx < gaps:
            cursor += per_gap


def wrap_text(text: str, width: int) -> list[str]:
    cleaned = text.replace("\n", " ").strip()
    paragraphs = [p.strip() for p in re.split(r"\s{2,}", cleaned) if p.strip()]
    if not paragraphs:
        paragraphs = [cleaned]

    lines: list[str] = []
    for paragraph in paragraphs:
        wrapped = textwrap.wrap(paragraph, width=width)
        lines.extend(wrapped or [""])
        lines.append("")
    if lines:
        lines.pop()
    return lines or [""]


def average_character_width(font: ImageFont.ImageFont) -> float:
    sample = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
    widths = [line_width(ch, font) for ch in sample]
    return max(sum(widths) / len(widths), 1.0)


def fit_lines_for_target_width(text: str, font: ImageFont.ImageFont, target_text_width: int) -> tuple[list[str], int]:
    avg_char = average_character_width(font)
    wrap_width = max(10, int(target_text_width / avg_char))
    wrap_width = min(wrap_width, 400)

    lines = wrap_text(text, wrap_width)
    max_line = max((line_width(line, font) for line in lines), default=0.0)

    while max_line > target_text_width and wrap_width > 5:
        wrap_width -= 1
        lines = wrap_text(text, wrap_width)
        max_line = max((line_width(line, font) for line in lines), default=0.0)

    while wrap_width < 400:
        next_lines = wrap_text(text, wrap_width + 1)
        next_max = max((line_width(line, font) for line in next_lines), default=0.0)
        if next_max > target_text_width:
            break
        wrap_width += 1
        lines = next_lines

    return lines, wrap_width


def render_text_fixed_width(text: str, target_width: int, font: ImageFont.ImageFont, padding: int, line_spacing: int) -> tuple[Image.Image, int]:
    safe_width = max(target_width, padding * 2 + 20)
    target_text_width = max(safe_width - padding * 2, 20)
    lines, wrap_used = fit_lines_for_target_width(text, font, target_text_width)
    bbox = font.getbbox("Hg")
    line_height = (bbox[3] - bbox[1]) + line_spacing
    height = line_height * max(1, len(lines)) + padding * 2

    image = Image.new("RGB", (safe_width, height), "white")
    draw = ImageDraw.Draw(image)
    y = padding
    for idx, line in enumerate(lines):
        is_last = idx == len(lines) - 1
        if not line.strip() or is_last:
            draw.text((padding, y), line, font=font, fill="black")
        else:
            draw_justified_line(draw=draw, line=line, font=font, x=padding, y=y, max_width=float(target_text_width))
        y += line_height

    return image, wrap_used


def random_replacement_char(original: str, rng: random.Random) -> str:
    if original.isascii() and original.isalpha():
        alphabet = string.ascii_lowercase if original.islower() else string.ascii_uppercase
        replacement = rng.choice(alphabet)
        while replacement == original:
            replacement = rng.choice(alphabet)
        return replacement
    alphabet = string.ascii_lowercase
    replacement = rng.choice(alphabet)
    while replacement == original:
        replacement = rng.choice(alphabet)
    return replacement


def mutate_one_letter(word: str, rng: random.Random) -> tuple[str, bool]:
    letter_positions = [idx for idx, char in enumerate(word) if char.isalpha()]
    if not letter_positions:
        return word, False
    target_idx = rng.choice(letter_positions)
    replacement = random_replacement_char(word[target_idx], rng)
    mutated = word[:target_idx] + replacement + word[target_idx + 1 :]
    return mutated, True


def mutate_gt_text(gt_text: str, rng: random.Random) -> str:
    tokens = re.findall(r"\s+|\S+", gt_text)
    word_positions = [idx for idx, token in enumerate(tokens) if not token.isspace()]

    for pair_start in range(0, len(word_positions) - 1, 2):
        first = word_positions[pair_start]
        second = word_positions[pair_start + 1]
        chosen = rng.choice([first, second])
        mutated, success = mutate_one_letter(tokens[chosen], rng)
        if success:
            tokens[chosen] = mutated
            continue

        other = second if chosen == first else first
        mutated, success = mutate_one_letter(tokens[other], rng)
        if success:
            tokens[other] = mutated

    return "".join(tokens)


def build_randomized_records(records: list[dict], rng: random.Random) -> list[dict]:
    randomized: list[dict] = []
    for record in records:
        if not isinstance(record, dict):
            continue
        updated = dict(record)
        original_gt = str(record.get("gt", ""))
        updated["gt"] = mutate_gt_text(original_gt, rng)
        randomized.append(updated)
    return randomized


def render_images(records: list[dict], images_dir: Path, font: ImageFont.ImageFont, padding: int, line_spacing: int, dpi: int) -> list[dict]:
    images_dir.mkdir(parents=True, exist_ok=True)
    rendered_records: list[dict] = []
    skipped = 0

    for idx, record in enumerate(records, start=1):
        image_name = str(record.get("image", "")).strip()
        gt_text = str(record.get("gt", ""))
        width_value = record.get("width", 0)
        try:
            target_width = int(width_value)
        except (TypeError, ValueError):
            target_width = 0

        if not image_name or target_width <= 0:
            skipped += 1
            rendered_records.append(record)
            continue

        image_path = images_dir / image_name
        image_path.parent.mkdir(parents=True, exist_ok=True)
        image, wrap_used = render_text_fixed_width(text=gt_text, target_width=target_width, font=font, padding=padding, line_spacing=line_spacing)
        image.convert("RGB").save(image_path, "JPEG", quality=95, dpi=(dpi, dpi))

        updated = dict(record)
        updated["width"] = image.width
        updated["height"] = image.height
        updated["wrap_width"] = wrap_used
        updated["aspect_ratio"] = (image.width / image.height) if image.height else 0.0
        rendered_records.append(updated)

        if idx % 100 == 0:
            logger.info("Rendered %d/%d images.", idx, len(records))

    logger.info("Image rendering done. Rendered %d entries, skipped %d invalid entries.", len(records) - skipped, skipped)
    return rendered_records


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)

    source_records = load_records(args.input_json)
    mutated_records = build_randomized_records(source_records, rng)

    font = load_font(args.font_path, args.font_size)
    rendered_records = render_images(
        records=mutated_records,
        images_dir=args.images_dir,
        font=font,
        padding=args.padding,
        line_spacing=args.line_spacing,
        dpi=args.dpi,
    )

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(rendered_records, ensure_ascii=False, indent=2), encoding="utf-8")

    logger.info("Saved randomized metadata to %s", args.output_json)
    logger.info("Images saved under %s", args.images_dir)


if __name__ == "__main__":
    main()
