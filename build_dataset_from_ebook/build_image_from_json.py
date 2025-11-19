#!/usr/bin/env python3
"""Generate randomized Pride & Prejudice block images via PDF conversion.

Workflow:
1. Load pre-chunked blocks from the JSON dataset produced by build_json_from_txt.py.
2. Randomly sample block groups of different sizes (1, 2, 3, 4 blocks).
3. Render each group into a tightly padded PDF, then convert to JPG.
4. Record each group's ground-truth text alongside the corresponding image.

"""

from __future__ import annotations

import argparse
import json
import math
import random
import re
import textwrap
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont
from transformers import AutoTokenizer

WRAP_WIDTH_BY_BLOCKS = {1: 70, 2: 120, 3: 200, 4: 220}


@dataclass(frozen=True)
class Block:
    key: str
    text: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sample dataset blocks, create PDFs, and convert them to JPGs.")
    parser.add_argument("--dataset", type=Path, default=Path("build_dataset_from_ebook/src_book/pride_and_prejudice_dataset.json"), help="Path to the JSON dataset produced by build_json_from_txt.py.")
    parser.add_argument("--output_dir", type=Path, default=Path("build_dataset_from_ebook/pride_dataset/"), help="Directory where PDFs, JPGs, and metadata JSON are stored.")
    parser.add_argument("--font_path", type=Path, default=None, help="Optional path to a TTF/OTF font file. Defaults to DejaVuSans or PIL font.")
    parser.add_argument("--font_size", type=int, default=28, help="Font size for rendered text.")
    parser.add_argument("--wrap_width", type=int, default=60, help="Fallback character count per line when no preset applies.")
    parser.add_argument("--padding", type=int, default=20, help="Pixel padding around rendered text.")
    parser.add_argument("--dpi", type=int, default=300, help="DPI used when converting PDFs to JPGs.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducible sampling.")
    parser.add_argument("--single_count", type=int, default=100, help="Number of single-block samples.")
    parser.add_argument("--double_count", type=int, default=200, help="Number of double-block samples.")
    parser.add_argument("--triple_count", type=int, default=200, help="Number of triple-block samples.")
    parser.add_argument("--quadruple_count", type=int, default=300, help="Number of quadruple-block samples.")
    parser.add_argument("--tokenizer_name", type=str, default="Qwen/Qwen2.5-0.5B-Instruct", help="Tokenizer identifier for computing text_token_len.")
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
    text = text.replace("\\n", " ").strip()
    paragraphs = [p.strip() for p in re.split(r"\\s{2,}", text) if p.strip()]
    if not paragraphs:
        paragraphs = [text]

    lines: list[str] = []
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


def ensure_output_dirs(base_dir: Path) -> dict[str, Path]:
    pdf_dir = base_dir / "pdf"
    jpg_dir = base_dir / "images"
    base_dir.mkdir(parents=True, exist_ok=True)
    pdf_dir.mkdir(parents=True, exist_ok=True)
    jpg_dir.mkdir(parents=True, exist_ok=True)
    return {"base": base_dir, "pdf": pdf_dir, "jpg": jpg_dir}


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


def block_code(block: Block) -> str:
    match = re.match(r"(?P<chapter>.+?)_block(?P<idx>\d+)$", block.key)
    if not match:
        raise ValueError(f"Unexpected block key format: {block.key}")
    chapter = re.sub(r"[^0-9A-Za-z]+", "", match.group("chapter")) or "X"
    block_idx = match.group("idx")
    return f"C{chapter}B{block_idx}"


def process_groups(
    label: str,
    groups: Sequence[tuple[int, ...]],
    blocks: Sequence[Block],
    font: ImageFont.ImageFont,
    dirs: dict[str, Path],
    wrap_width: int,
    padding: int,
    dpi: int,
    tokenizer: AutoTokenizer,
    token_cache: dict[str, int],
) -> list[dict]:
    records: list[dict] = []
    for combo in groups:
        combined_text = " ".join(blocks[idx].text for idx in combo)
        codes = [block_code(blocks[idx]) for idx in combo]
        base_name = f"pride_prejudice_{label}_{'_'.join(codes)}"
        block_count = len(combo)
        dynamic_wrap = WRAP_WIDTH_BY_BLOCKS.get(block_count, wrap_width)
        pdf_path = dirs["pdf"] / f"{base_name}.pdf"
        jpg_path = dirs["jpg"] / f"{base_name}.jpg"
        image = render_text_image(
            combined_text,
            font=font,
            wrap_width=dynamic_wrap,
            padding=padding,
        )
        save_pdf_and_jpg(image, pdf_path, jpg_path, dpi=dpi)
        text_token_len = token_cache.get(combined_text)
        if text_token_len is None:
            tokens = tokenizer(combined_text, add_special_tokens=True, truncation=False)
            text_token_len = len(tokens.get("input_ids", []))
            token_cache[combined_text] = text_token_len
        records.append({"image": jpg_path.name, "gt": combined_text, "text_token_len": text_token_len})
    return records


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)

    blocks = load_blocks_from_dataset(args.dataset)
    if not blocks:
        raise RuntimeError("No non-empty blocks found in the dataset.")

    dirs = ensure_output_dirs(args.output_dir)
    font = load_font(args.font_path, args.font_size)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name)
    token_cache: dict[str, int] = {}
    all_indices = list(range(len(blocks)))

    sampling_plan = [
        ("single", 1, args.single_count),
        ("double", 2, args.double_count),
        ("triple", 3, args.triple_count),
        ("quadruple", 4, args.quadruple_count),
    ]

    metadata: list[dict] = []
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
                tokenizer=tokenizer,
                token_cache=token_cache,
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
