#!/usr/bin/env python3
"""Render identical text blocks into multiple page widths for PDF/JPG pairs."""

from __future__ import annotations

import argparse
import json
import random
import re
import textwrap
from pathlib import Path
from typing import Iterable

from PIL import Image, ImageDraw, ImageFont
from transformers import AutoTokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render same block pairs into PDFs/JPGs with different widths.")
    parser.add_argument("--dataset", type=Path, default=Path("build_dataset_from_ebook/src_book/pride_and_prejudice_dataset.json"), help="JSON file produced by build_json_from_txt.py.")
    parser.add_argument("--output_dir", type=Path, default=Path("build_dataset_from_ebook/pride_dataset_same_content_diff_shape/"), help="Destination directory for generated PDFs, images, and metadata.")
    parser.add_argument("--font_path", type=Path, default=None, help="Optional path to a TTF/OTF font. Defaults to DejaVuSans or PIL fallback.")
    parser.add_argument("--font_size", type=int, default=28, help="Font size for rendering.")
    parser.add_argument("--padding", type=int, default=20, help="Inner padding (pixels).")
    parser.add_argument("--dpi", type=int, default=300, help="Resolution for PDF/JPG export.")
    parser.add_argument("--widths", type=int, nargs="+", required=True, help="One or more page widths (pixels). Each width produces its own PDF/JPG.")
    parser.add_argument("--metadata_name", type=str, default="pride_prejudice_same_content_diff_shape_gt.json", help="Filename for the output metadata JSON.")
    parser.add_argument("--max_chapters", type=int, default=None, help="Optionally limit how many chapters (records) to process.")
    parser.add_argument("--tokenizer_name", type=str, default="Qwen/Qwen2.5-0.5B-Instruct", help="Tokenizer used to compute text_token_len.")
    return parser.parse_args()


def load_dataset(path: Path) -> list[dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError("Dataset JSON must contain a list of records.")
    return data


def normalize_page_id(value: str | int) -> str:
    cleaned = re.sub(r"[^0-9A-Za-z]+", "", str(value))
    return cleaned or "X"


def extract_block_keys(record: dict) -> list[str]:
    block_keys = [
        key for key in record.keys()
        if key.startswith("block") and isinstance(record.get(key), str)
    ]
    return sorted(block_keys, key=lambda k: int(k[5:]) if k[5:].isdigit() else k)


def pick_blocks(record: dict, size: int) -> tuple[list[str], str] | None:
    block_keys = extract_block_keys(record)
    if len(block_keys) < size:
        return None
    selected = random.sample(block_keys, size)
    texts = [record.get(key, "").strip() for key in selected]
    if any(not text for text in texts):
        return None
    combined = "\n\n".join(texts)
    return selected, combined


def load_font(font_path: Path | None, font_size: int) -> ImageFont.ImageFont:
    if font_path is not None:
        return ImageFont.truetype(str(font_path), font_size)
    try:
        return ImageFont.truetype("DejaVuSans.ttf", font_size)
    except OSError:
        return ImageFont.load_default()


def wrap_text(text: str, width: int) -> list[str]:
    cleaned = text.replace("\\n", " ").strip()
    paragraphs = [p.strip() for p in re.split(r"\\s{2,}", cleaned) if p.strip()] or [cleaned]
    lines: list[str] = []
    for paragraph in paragraphs:
        wrapped = textwrap.wrap(paragraph, width=width) or [""]
        lines.extend(wrapped)
        lines.append("")
    if lines:
        lines.pop()
    return lines or [""]


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


def compute_wrap_width(target_width: int, padding: int) -> int:
    usable_width = max(target_width - padding * 2, 50)
    approx_char_width = 8
    return max(int(usable_width / approx_char_width), 10)


def ensure_dirs(base_dir: Path) -> dict[str, Path]:
    pdf_dir = base_dir / "pdf"
    img_dir = base_dir / "images"
    base_dir.mkdir(parents=True, exist_ok=True)
    pdf_dir.mkdir(parents=True, exist_ok=True)
    img_dir.mkdir(parents=True, exist_ok=True)
    return {"base": base_dir, "pdf": pdf_dir, "img": img_dir}


def save_pdf_and_jpg(image: Image.Image, pdf_path: Path, jpg_path: Path, dpi: int) -> None:
    rgb_image = image.convert("RGB")
    rgb_image.save(pdf_path, "PDF", resolution=dpi)
    rgb_image.save(jpg_path, "JPEG", quality=95, dpi=(dpi, dpi))


def block_code(page_id: str, block_key: str) -> str:
    chapter = normalize_page_id(page_id)
    match = re.search(r"(\d+)$", block_key)
    block_idx = match.group(1) if match else "X"
    return f"C{chapter}B{block_idx}"


def unique_widths(widths: Iterable[int]) -> list[int]:
    cleaned = sorted({int(w) for w in widths if int(w) > 0})
    if not cleaned:
        raise ValueError("At least one positive width must be provided.")
    return cleaned


def compute_text_token_len(text: str, tokenizer: AutoTokenizer, cache: dict[str, int]) -> int:
    if text in cache:
        return cache[text]
    tokens = tokenizer(text, add_special_tokens=True, truncation=False)
    length = len(tokens.get("input_ids", []))
    cache[text] = length
    return length


def main() -> None:
    args = parse_args()
    dataset = load_dataset(args.dataset)
    widths = unique_widths(args.widths)
    font = load_font(args.font_path, args.font_size)
    dirs = ensure_dirs(args.output_dir)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name)
    token_cache: dict[str, int] = {}

    records_out: list[dict] = []
    processed = 0
    size_plan = [("single", 1), ("double", 2), ("triple", 3)]

    for record in dataset:
        if args.max_chapters is not None and processed >= args.max_chapters:
            break
        page_id = record.get("page_id") or record.get("chapter_id") or record.get("id")
        if page_id is None:
            continue
        generated_any = False
        for label, size in size_plan:
            selection = pick_blocks(record, size)
            if selection is None:
                continue
            block_keys, combined_text = selection
            codes = [block_code(page_id, key) for key in block_keys]
            base_stub = f"pride_prejudice_{label}_{'_'.join(codes)}"
            token_len = compute_text_token_len(combined_text, tokenizer, token_cache)
            for width in widths:
                pdf_path = dirs["pdf"] / f"{base_stub}_{width}.pdf"
                jpg_path = dirs["img"] / f"{base_stub}_{width}.jpg"
                wrap_chars = compute_wrap_width(width, args.padding)
                image = render_text_image(
                    combined_text,
                    font=font,
                    wrap_width=wrap_chars,
                    padding=args.padding,
                )
                save_pdf_and_jpg(image, pdf_path, jpg_path, dpi=args.dpi)
                records_out.append(
                    {
                        "image": jpg_path.name,
                        "pdf": pdf_path.name,
                        "gt": combined_text,
                        "width": width,
                        "chapter_id": normalize_page_id(page_id),
                        "blocks": codes,
                        "text_token_len": token_len,
                        "group_label": label,
                    }
                )
            generated_any = True
        if generated_any:
            processed += 1

    if not records_out:
        raise RuntimeError("No valid chapters were processed; check the dataset content.")

    metadata_path = dirs["base"] / args.metadata_name
    metadata_path.write_text(json.dumps(records_out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {len(records_out)} records to {metadata_path} with PDFs in {dirs['pdf']} and images in {dirs['img']}.")


if __name__ == "__main__":
    main()
