#!/usr/bin/env python3
"""Generate square-ish OCR samples from dataset blocks via shared square render helpers."""

from __future__ import annotations

import argparse
import json
import logging
import math
import random
import re
import textwrap
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont
from transformers import AutoTokenizer
from utils.generate_square_images import render_text_assets, save_image_and_pdf

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

WRAP_WIDTH_BY_BLOCKS = {1: 70, 2: 120, 3: 200, 4: 220}


@dataclass(frozen=True)
class Block:
    key: str
    text: str


class WrapHistory:
    """Store successful (token_len, wrap_width) pairs to guide future guesses."""

    def __init__(self) -> None:
        self.records: list[tuple[int, int]] = []

    def add(self, token_len: int, wrap_width: int) -> None:
        self.records.append((token_len, wrap_width))

    def best_wrap(self, token_len: int) -> int | None:
        if not self.records:
            return None
        best_token, best_wrap = min(self.records, key=lambda item: abs(item[0] - token_len))
        return best_wrap


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render dataset blocks into near-square PDFs/JPGs.")
    parser.add_argument("--dataset", type=Path, default=Path("build_dataset_from_ebook/src_book/pride_and_prejudice_dataset.json"), help="Path to the JSON dataset produced by build_json_from_txt.py.")
    parser.add_argument("--output_dir", type=Path, default=Path("build_dataset_from_ebook/pride_dataset_square/"), help="Directory where PDFs, JPGs, and metadata JSON are stored.")
    parser.add_argument("--font_path", type=Path, default=None, help="Optional path to a TTF/OTF font file. Defaults to DejaVuSans or PIL font.")
    parser.add_argument("--font_size", type=int, default=28, help="Font size for rendered text.")
    parser.add_argument("--padding", type=int, default=20, help="Pixel padding around rendered text.")
    parser.add_argument("--wrap_width", type=int, default=60, help="Fallback character count per line when no preset applies.")
    parser.add_argument("--dpi", type=int, default=300, help="DPI used when converting PDFs to JPGs.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducible sampling.")
    parser.add_argument("--single_count", type=int, default=100, help="Number of single-block samples.")
    parser.add_argument("--double_count", type=int, default=200, help="Number of double-block samples.")
    parser.add_argument("--triple_count", type=int, default=200, help="Number of triple-block samples.")
    parser.add_argument("--quadruple_count", type=int, default=300, help="Number of quadruple-block samples.")
    parser.add_argument("--quintuple_count", type=int, default=0, help="Number of five-block samples.")
    parser.add_argument("--sextuple_count", type=int, default=0, help="Number of six-block samples.")
    parser.add_argument("--septuple_count", type=int, default=0, help="Number of seven-block samples.")
    parser.add_argument("--octuple_count", type=int, default=0, help="Number of eight-block samples.")
    parser.add_argument("--nonuple_count", type=int, default=0, help="Number of nine-block samples.")
    parser.add_argument("--decuple_count", type=int, default=0, help="Number of ten-block samples.")
    parser.add_argument("--ratio_min", type=float, default=0.9, help="Minimum acceptable width/height ratio.")
    parser.add_argument("--ratio_max", type=float, default=1.1, help="Maximum acceptable width/height ratio.")
    parser.add_argument("--tokenizer_name", type=str, default="Qwen/Qwen2.5-0.5B-Instruct", help="Tokenizer identifier for computing text_token_len.")
    parser.add_argument("--metadata_name", type=str, default="pride_prejudice_square_gt.json", help="Filename for the output metadata JSON.")
    parser.add_argument("--output_image_prefix", type=str, default="pride_pejudice", help="Prefix for generated PDF/JPG filenames.")
    parser.add_argument("--line_spacing", type=int, default=6, help="Line spacing (pixels) used during rendering.")
    parser.add_argument("--letter_spacing", type=int, default=0, help="Letter spacing (pixels) used during rendering.")
    return parser.parse_args()


def load_blocks_from_dataset(path: Path) -> list[Block]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError("Dataset JSON must contain a list of records.")

    blocks: list[Block] = []
    for record in data:
        page_id = str(record.get("page_id", "")).strip()
        if not page_id:
            continue
        block_keys = sorted(
            (key for key in record.keys() if key.startswith("block")),
            key=lambda k: int(k[5:]) if k[5:].isdigit() else k,
        )
        for key in block_keys:
            text = str(record.get(key, "")).strip()
            if not text:
                continue
            blocks.append(Block(key=f"{page_id}_{key}", text=text))
    return blocks


def wrap_text(text: str, width: int) -> list[str]:
    text = text.replace("\n", " ").strip()
    paragraphs = [p.strip() for p in re.split(r"\s{2,}", text) if p.strip()]
    if not paragraphs:
        paragraphs = [text]

    lines: list[str] = []
    for paragraph in paragraphs:
        wrapped = textwrap.wrap(paragraph, width=width)
        lines.extend(wrapped or [""])
        lines.append("")
    if lines:
        lines.pop()
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


def ensure_output_dirs(base_dir: Path) -> dict[str, Path]:
    pdf_dir = base_dir / "pdf"
    jpg_dir = base_dir / "images"
    base_dir.mkdir(parents=True, exist_ok=True)
    pdf_dir.mkdir(parents=True, exist_ok=True)
    jpg_dir.mkdir(parents=True, exist_ok=True)
    return {"base": base_dir, "pdf": pdf_dir, "jpg": jpg_dir}


def remove_unused_files(directory: Path, keep: set[str]) -> int:
    removed = 0
    for path in directory.glob("*"):
        if path.is_file() and path.name not in keep:
            path.unlink()
            removed += 1
    return removed


def sample_unique_groups(
    indices: Sequence[int], group_size: int, count: int, rng: random.Random
) -> list[tuple[int, ...]]:
    if group_size > len(indices):
        raise ValueError("group_size cannot exceed number of available blocks.")
    seen: set[tuple[int, ...]] = set()
    groups: list[tuple[int, ...]] = []
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
        logger.warning(
            "Requested %d groups of size %d, but only %d unique combinations exist. Using %d.",
            requested,
            group_size,
            available,
            available,
        )
    return min(requested, available)


def block_code(block: Block) -> str:
    match = re.match(r"(?P<chapter>.+?)_block(?P<idx>\d+)$", block.key)
    if not match:
        raise ValueError(f"Unexpected block key format: {block.key}")
    chapter = re.sub(r"[^0-9A-Za-z]+", "", match.group("chapter")) or "X"
    block_idx = match.group("idx")
    return f"C{chapter}B{block_idx}"


def compute_text_token_len(text: str, tokenizer: AutoTokenizer, cache: dict[str, int]) -> int:
    if text in cache:
        return cache[text]
    tokens = tokenizer(text, add_special_tokens=True, truncation=False)
    length = len(tokens.get("input_ids", []))
    cache[text] = length
    return length


def render_squareish_image(
    text: str,
    font: ImageFont.ImageFont,
    font_path: Path | None,
    font_size: int,
    pdf_dpi: int,
    padding: int,
    line_spacing: int,
    letter_spacing: int,
    token_len: int,
    base_wrap: int,
    history: WrapHistory,
    ratio_min: float,
    ratio_max: float,
) -> tuple[Image.Image, bytes, int, float]:
    render_result = render_text_assets(
        text=text,
        font_size=font_size,
        line_spacing=line_spacing,
        letter_spacing=letter_spacing,
        font_path=font_path,
        pdf_dpi=pdf_dpi,
        padding=padding,
        ratio_min=ratio_min,
        ratio_max=ratio_max,
    )
    image = render_result["image"]
    pdf_bytes = render_result["pdf"]
    width, height = render_result["image_shape"]
    aspect_ratio = (width / height) if height else 0.0
    history.add(token_len, base_wrap)
    return image, pdf_bytes, base_wrap, aspect_ratio


def process_groups(
    label: str,
    groups: Sequence[tuple[int, ...]],
    blocks: Sequence[Block],
    font: ImageFont.ImageFont,
    font_path: Path | None,
    font_size: int,
    line_spacing: int,
    letter_spacing: int,
    dirs: dict[str, Path],
    wrap_width: int,
    padding: int,
    dpi: int,
    tokenizer: AutoTokenizer,
    token_cache: dict[str, int],
    history: WrapHistory,
    ratio_min: float,
    ratio_max: float,
    prefix: str,
    used_pdfs: set[str],
    used_jpgs: set[str],
) -> list[dict]:
    records: list[dict] = []
    for combo in groups:
        combined_text = " ".join(blocks[idx].text for idx in combo)
        codes = [block_code(blocks[idx]) for idx in combo]
        base_name = f"{prefix}_{label}_{'_'.join(codes)}"
        pdf_path = dirs["pdf"] / f"{base_name}.pdf"
        jpg_path = dirs["jpg"] / f"{base_name}.jpg"
        pdf_name = pdf_path.name
        jpg_name = jpg_path.name
        used_pdfs.add(pdf_name)
        used_jpgs.add(jpg_name)
        block_count = len(combo)
        dynamic_wrap = WRAP_WIDTH_BY_BLOCKS.get(block_count, wrap_width)
        token_len = compute_text_token_len(combined_text, tokenizer, token_cache)
        image, pdf_bytes, wrap_used, aspect_ratio = render_squareish_image(
            text=combined_text,
            font=font,
            font_path=font_path,
            font_size=font_size,
            pdf_dpi=dpi,
            padding=padding,
            line_spacing=line_spacing,
            letter_spacing=letter_spacing,
            token_len=token_len,
            base_wrap=dynamic_wrap,
            history=history,
            ratio_min=ratio_min,
            ratio_max=ratio_max,
        )
        if jpg_path.exists():
            logger.info("Skipping existing image/pdf pair for %s", base_name)
        else:
            save_image_and_pdf(image=image, pdf=pdf_bytes, image_path=jpg_path, pdf_path=pdf_path, dpi=dpi)
        records.append(
            {
                "image": jpg_name,
                "pdf": pdf_name,
                "gt": combined_text,
                "text_token_len": token_len,
                "wrap_width": wrap_used,
                "aspect_ratio": aspect_ratio,
                "group_label": label,
                "blocks": codes,
            }
        )
    return records


def ratio_within_range(aspect_ratio: float | None, ratio_min: float, ratio_max: float) -> bool:
    if aspect_ratio is None:
        return False
    return ratio_min <= aspect_ratio <= ratio_max

def generate_metadata(
    args: argparse.Namespace,
    blocks: list[Block],
    font: ImageFont.ImageFont,
    tokenizer: AutoTokenizer,
    token_cache: dict[str, int],
    history: WrapHistory,
    rng: random.Random,
    dirs: dict[str, Path],
) -> tuple[list[dict], set[str], set[str]]:
    all_indices = list(range(len(blocks)))
    sampling_plan = [
        ("single", 1, args.single_count),
        ("double", 2, args.double_count),
        ("triple", 3, args.triple_count),
        ("quadruple", 4, args.quadruple_count),
        ("quintuple", 5, args.quintuple_count),
        ("sextuple", 6, args.sextuple_count),
        ("septuple", 7, args.septuple_count),
        ("octuple", 8, args.octuple_count),
        ("nonuple", 9, args.nonuple_count),
        ("decuple", 10, args.decuple_count),
    ]

    metadata: list[dict] = []
    used_pdfs: set[str] = set()
    used_jpgs: set[str] = set()
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
                font_path=args.font_path,
                font_size=args.font_size,
                line_spacing=args.line_spacing,
                letter_spacing=args.letter_spacing,
                dirs=dirs,
                wrap_width=args.wrap_width,
                padding=args.padding,
                dpi=args.dpi,
                tokenizer=tokenizer,
                token_cache=token_cache,
                history=history,
                ratio_min=args.ratio_min,
                ratio_max=args.ratio_max,
                prefix=args.output_image_prefix,
                used_pdfs=used_pdfs,
                used_jpgs=used_jpgs,
            )
        )
    return metadata, used_pdfs, used_jpgs


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)

    blocks = load_blocks_from_dataset(args.dataset)
    if not blocks:
        raise RuntimeError("No non-empty blocks found in the dataset.")

    dirs = ensure_output_dirs(args.output_dir)
    existing_pdfs = {path.name for path in dirs["pdf"].glob("*.pdf")}
    existing_jpgs = {path.name for path in dirs["jpg"].glob("*.jpg")}
    font = load_font(args.font_path, args.font_size)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name)
    token_cache: dict[str, int] = {}
    history = WrapHistory()

    metadata, used_pdfs, used_jpgs = generate_metadata(args, blocks, font, tokenizer, token_cache, history, rng, dirs)

    output_json = dirs["base"] / args.metadata_name
    output_json.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    pdf_new = len(used_pdfs - existing_pdfs)
    pdf_reused = len(used_pdfs & existing_pdfs)
    jpg_new = len(used_jpgs - existing_jpgs)
    jpg_reused = len(used_jpgs & existing_jpgs)
    removed_pdfs = remove_unused_files(dirs["pdf"], used_pdfs)
    removed_jpgs = remove_unused_files(dirs["jpg"], used_jpgs)
    logger.info(
        "PDF files - new: %d, reused/skipped: %d, removed stale: %d",
        pdf_new,
        pdf_reused,
        removed_pdfs,
    )
    logger.info(
        "Image files - new: %d, reused/skipped: %d, removed stale: %d",
        jpg_new,
        jpg_reused,
        removed_jpgs,
    )

    logger.info(
        "Created %d square-ish JPG/PDF pairs under %s and wrote metadata to %s",
        len(metadata),
        dirs["base"],
        output_json,
    )


if __name__ == "__main__":
    main()
