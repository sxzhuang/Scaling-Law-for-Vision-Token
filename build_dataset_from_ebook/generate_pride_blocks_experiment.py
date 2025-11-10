#!/usr/bin/env python3
"""Generate randomized Pride & Prejudice block images via PDF conversion.

Workflow:
1. Parse chapters from a plain-text ebook (lines beginning with ``CHAPTER``).
2. Split each chapter into four sentence-aligned blocks.
3. Randomly sample block groups of different sizes (1, 2, 3, 4 blocks).
4. Render each group into a tightly padded PDF, then convert to JPG.
5. Record each group's ground-truth text alongside the corresponding image.

Dependencies: Pillow, pdf2image, and a working Poppler installation for
pdf2image. Install via:
    pip install pillow pdf2image
"""

from __future__ import annotations

import argparse
import json
import math
import random
import re
import textwrap
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

from PIL import Image, ImageDraw, ImageFont

CHAPTER_PATTERN = re.compile(r"^\s*CHAPTER\s+([A-Z0-9]+)[^\r\n]*", re.MULTILINE)
SENTENCE_SPLIT_PATTERN = re.compile(r"(?<=[.!?])\s+")


@dataclass(frozen=True)
class Block:
    key: str
    text: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sample chapter blocks, create PDFs, and convert them to JPGs."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("pride-and-prejudice.txt"),
        help="Path to the Pride and Prejudice plain-text file.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("pride_blocks_experiment"),
        help="Directory where PDFs, JPGs, and metadata JSON are stored.",
    )
    parser.add_argument(
        "--font-path",
        type=Path,
        default=None,
        help="Optional path to a TTF/OTF font file. Defaults to DejaVuSans or PIL font.",
    )
    parser.add_argument(
        "--font-size",
        type=int,
        default=28,
        help="Font size for rendered text.",
    )
    parser.add_argument(
        "--wrap-width",
        type=int,
        default=60,
        help="Approximate character count per line before wrapping.",
    )
    parser.add_argument(
        "--padding",
        type=int,
        default=20,
        help="Pixel padding around rendered text.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="DPI used when converting PDFs to JPGs.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible sampling.",
    )
    parser.add_argument(
        "--single-count",
        type=int,
        default=100,
        help="Number of single-block samples.",
    )
    parser.add_argument(
        "--double-count",
        type=int,
        default=200,
        help="Number of double-block samples.",
    )
    parser.add_argument(
        "--triple-count",
        type=int,
        default=200,
        help="Number of triple-block samples.",
    )
    parser.add_argument(
        "--quadruple-count",
        type=int,
        default=300,
        help="Number of quadruple-block samples.",
    )
    return parser.parse_args()


def load_text(path: Path) -> str:
    text = path.read_text(encoding="utf-8")
    return text.replace("\\ufeff", "")


def extract_chapters(text: str) -> List[tuple[str, str]]:
    matches = list(CHAPTER_PATTERN.finditer(text))
    chapters: List[tuple[str, str]] = []
    for idx, match in enumerate(matches):
        start = match.end()
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(text)
        body = text[start:end].strip()
        page_id = match.group(1).strip(". ")
        if not body:
            continue
        chapters.append((page_id, body))
    return chapters


def normalize_paragraph(text: str) -> str:
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def split_sentences(paragraph: str) -> List[str]:
    if not paragraph:
        return []
    sentences = SENTENCE_SPLIT_PATTERN.split(paragraph)
    return [sentence.strip() for sentence in sentences if sentence.strip()]


def chunk_sentences(sentences: Sequence[str], block_count: int) -> List[str]:
    if block_count <= 0:
        raise ValueError("block_count must be greater than zero.")
    if not sentences:
        return [""] * block_count

    total_len = sum(len(sentence) for sentence in sentences)
    target_len = total_len / block_count

    blocks: List[str] = []
    current: List[str] = []
    current_len = 0

    for idx, sentence in enumerate(sentences):
        current.append(sentence)
        current_len += len(sentence)

        remaining_sentences = len(sentences) - idx - 1
        remaining_blocks = block_count - len(blocks) - 1
        must_split = remaining_blocks > 0 and remaining_sentences == remaining_blocks
        should_split = (
            remaining_blocks > 0
            and remaining_sentences > remaining_blocks
            and current_len >= target_len
        )
        if must_split or should_split:
            blocks.append(" ".join(current).strip())
            current = []
            current_len = 0

    blocks.append(" ".join(current).strip())

    if len(blocks) < block_count:
        blocks.extend([""] * (block_count - len(blocks)))
    elif len(blocks) > block_count:
        tail = " ".join(blocks[block_count - 1 :]).strip()
        blocks = blocks[: block_count - 1]
        blocks.append(tail)
    return blocks


def create_blocks(
    chapters: Sequence[Tuple[str, str]], block_count: int = 4
) -> List[Block]:
    blocks: List[Block] = []
    for page_id, raw_body in chapters:
        normalized = normalize_paragraph(raw_body)
        sentences = split_sentences(normalized)
        chunked = chunk_sentences(sentences, block_count)
        for idx, text in enumerate(chunked, start=1):
            cleaned = text.strip()
            if not cleaned:
                continue
            key = f"{page_id}_block{idx}"
            blocks.append(Block(key=key, text=cleaned))
    return blocks


def wrap_text(text: str, width: int) -> List[str]:
    text = text.replace("\\n", " ").strip()
    paragraphs = [p.strip() for p in re.split(r"\\s{2,}", text) if p.strip()]
    if not paragraphs:
        paragraphs = [text]

    lines: List[str] = []
    for paragraph in paragraphs:
        wrapped = textwrap.wrap(paragraph, width=width)
        lines.extend(wrapped or [""])
        lines.append("")
    if lines:
        lines.pop()  # remove trailing blank line
    return lines or [""]


def load_font(font_path: Path | None, font_size: int) -> ImageFont.ImageFont:
    if font_path is not None:
        return ImageFont.truetype(str(font_path), font_size)
    try:
        return ImageFont.truetype("DejaVuSans.ttf", font_size)
    except OSError:
        return ImageFont.load_default()


def line_width(line: str, font: ImageFont.ImageFont) -> float:
    if hasattr(font, "getlength"):
        return float(font.getlength(line))
    bbox = font.getbbox(line or " ")
    return float(bbox[2] - bbox[0])


def draw_justified_line(
    draw: ImageDraw.ImageDraw,
    line: str,
    font: ImageFont.ImageFont,
    x: int,
    y: int,
    max_width: float,
    fill: str = "black",
) -> None:
    stripped = line.rstrip()
    if not stripped:
        draw.text((x, y), line, font=font, fill=fill)
        return
    words = stripped.split()
    if len(words) == 1:
        draw.text((x, y), stripped, font=font, fill=fill)
        return
    space_width = line_width(" ", font)
    total_word_width = sum(line_width(word, font) for word in words)
    gaps = len(words) - 1
    base_line_width = total_word_width + space_width * gaps
    extra_space = max(max_width - base_line_width, 0)
    per_gap = space_width + (extra_space / gaps if gaps else 0)

    cursor = x
    for idx, word in enumerate(words):
        draw.text((cursor, y), word, font=font, fill=fill)
        cursor += line_width(word, font)
        if idx < gaps:
            cursor += per_gap


def render_text_image(
    text: str,
    font: ImageFont.ImageFont,
    wrap_width: int,
    padding: int,
) -> Image.Image:
    lines = wrap_text(text, wrap_width)
    bbox = font.getbbox("Hg")
    line_height = (bbox[3] - bbox[1]) + 6
    max_width = max((line_width(line, font) for line in lines), default=0)
    width = int(max_width) + padding * 2 or padding * 2
    height = line_height * max(1, len(lines)) + padding * 2

    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    y = padding
    for idx, line in enumerate(lines):
        is_last = idx == len(lines) - 1
        if not line.strip() or is_last:
            draw.text((padding, y), line, font=font, fill="black")
        else:
            draw_justified_line(
                draw=draw,
                line=line,
                font=font,
                x=padding,
                y=y,
                max_width=max_width,
            )
        y += line_height
    return image


def ensure_output_dirs(base_dir: Path) -> Dict[str, Path]:
    pdf_dir = base_dir / "pdf"
    jpg_dir = base_dir / "images"
    base_dir.mkdir(parents=True, exist_ok=True)
    pdf_dir.mkdir(parents=True, exist_ok=True)
    jpg_dir.mkdir(parents=True, exist_ok=True)
    return {"base": base_dir, "pdf": pdf_dir, "jpg": jpg_dir}


def unique_suffix(rng: random.Random) -> str:
    timestamp = datetime.utcnow().strftime("%Y%m%d%H%M%S%f")
    extra = rng.randint(0, 99999)
    return f"{timestamp}_{extra:05d}"


def sample_unique_groups(
    indices: Sequence[int], group_size: int, count: int, rng: random.Random
) -> List[Tuple[int, ...]]:
    if group_size > len(indices):
        raise ValueError("group_size cannot exceed number of available blocks.")
    seen: set[Tuple[int, ...]] = set()
    groups: List[Tuple[int, ...]] = []
    available = list(indices)
    max_attempts = count * 20
    attempts = 0

    while len(groups) < count and attempts < max_attempts:
        selection = tuple(sorted(rng.sample(available, group_size)))
        if selection not in seen:
            seen.add(selection)
            groups.append(selection)
        attempts += 1

    if len(groups) < count:
        raise RuntimeError(
            f"Unable to sample {count} unique groups of size {group_size}. "
            "Reduce the requested count or ensure more blocks exist."
        )
    return groups


def effective_sample_count(total_blocks: int, group_size: int, requested: int) -> int:
    if group_size <= 0:
        raise ValueError("group_size must be positive.")
    if total_blocks < group_size:
        raise ValueError(
            f"Not enough blocks ({total_blocks}) to sample group size {group_size}."
        )
    if group_size == 1:
        available = total_blocks
    else:
        try:
            available = math.comb(total_blocks, group_size)
        except ValueError as exc:  # pragma: no cover
            raise ValueError(
                f"Unable to compute combinations for n={total_blocks}, k={group_size}"
            ) from exc
    if requested > available:
        print(
            f"[warn] Requested {requested} groups of size {group_size}, "
            f"but only {available} unique combinations exist. "
            f"Using {available} instead."
        )
    return min(requested, available)


def save_pdf_and_jpg(
    image: Image.Image,
    pdf_path: Path,
    jpg_path: Path,
    dpi: int,
) -> None:
    rgb_image = image.convert("RGB")
    rgb_image.save(pdf_path, "PDF", resolution=dpi)
    rgb_image.save(jpg_path, "JPEG", quality=95, dpi=(dpi, dpi))


def process_groups(
    label: str,
    groups: Sequence[Tuple[int, ...]],
    blocks: Sequence[Block],
    font: ImageFont.ImageFont,
    dirs: Dict[str, Path],
    wrap_width: int,
    padding: int,
    dpi: int,
    rng: random.Random,
) -> List[dict]:
    records: List[dict] = []
    for combo in groups:
        combined_text = " ".join(blocks[idx].text for idx in combo)
        suffix = unique_suffix(rng)
        base_name = f"pride_prejudice_{label}_block_{suffix}"
        pdf_path = dirs["pdf"] / f"{base_name}.pdf"
        jpg_path = dirs["jpg"] / f"{base_name}.jpg"
        image = render_text_image(
            combined_text,
            font=font,
            wrap_width=wrap_width,
            padding=padding,
        )
        save_pdf_and_jpg(image, pdf_path, jpg_path, dpi=dpi)
        records.append({"image": jpg_path.name, "gt": combined_text})
    return records


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)

    text = load_text(args.input)
    chapters = extract_chapters(text)
    if not chapters:
        raise RuntimeError("No chapters found. Ensure the text contains 'CHAPTER' headings.")
    blocks = create_blocks(chapters, block_count=4)
    if not blocks:
        raise RuntimeError("No non-empty blocks extracted from the text.")

    dirs = ensure_output_dirs(args.output_dir)
    font = load_font(args.font_path, args.font_size)
    all_indices = list(range(len(blocks)))

    sampling_plan = [
        ("single", 1, args.single_count),
        ("double", 2, args.double_count),
        ("triple", 3, args.triple_count),
        ("quadruple", 4, args.quadruple_count),
    ]

    metadata: List[dict] = []
    for label, size, count in sampling_plan:
        if count <= 0:
            continue
        actual_count = effective_sample_count(len(all_indices), size, count)
        if actual_count == 0:
            continue
        if size == 1:
            sampled = rng.sample(all_indices, actual_count)
            groups = [tuple([idx]) for idx in sampled]
        else:
            groups = sample_unique_groups(all_indices, size, actual_count, rng)
        metadata.extend(
            process_groups(
                label=label,
                groups=groups,
                blocks=blocks,
                font=font,
                dirs=dirs,
                wrap_width=args.wrap_width,
                padding=args.padding,
                dpi=args.dpi,
                rng=rng,
            )
        )

    output_json = dirs["base"] / "pride_prejudice_blocks_gt.json"
    output_json.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(
        f"Created {len(metadata)} JPG/PDF pairs under {dirs['base']} "
        f"and wrote metadata to {output_json}"
    )


if __name__ == "__main__":
    main()
