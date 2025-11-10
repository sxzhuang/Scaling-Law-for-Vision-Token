#!/usr/bin/env python3
"""Convert Pride and Prejudice dataset entries into per-page images."""

from __future__ import annotations

import argparse
import json
import textwrap
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

from PIL import Image, ImageDraw, ImageFont

DEFAULT_FONT_SIZE = 20
DEFAULT_LINE_WIDTH = 80
DEFAULT_PADDING = 40


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render dataset blocks into text images grouped by page_id."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("pride_and_prejudice_dataset.json"),
        help="Path to the JSON dataset produced by create_dataset_from_ebook.py.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("pride_pages"),
        help="Directory where generated images are stored.",
    )
    parser.add_argument(
        "--font-path",
        type=Path,
        default=None,
        help="Optional path to a .ttf/.otf font. Falls back to PIL default font.",
    )
    parser.add_argument(
        "--font-size",
        type=int,
        default=DEFAULT_FONT_SIZE,
        help="Font size for rendered text.",
    )
    parser.add_argument(
        "--line-width",
        type=int,
        default=DEFAULT_LINE_WIDTH,
        help="Approximate number of characters per line before wrapping.",
    )
    parser.add_argument(
        "--padding",
        type=int,
        default=DEFAULT_PADDING,
        help="Padding (in pixels) around the text area.",
    )
    parser.add_argument(
        "--book-name",
        default=None,
        help="Optional book_name filter. Only records matching this value are rendered.",
    )
    return parser.parse_args()


def load_dataset(path: Path) -> Sequence[dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError("Dataset JSON must be a list of records.")
    return data


def group_blocks_by_page(
    records: Iterable[dict], book_filter: str | None = None
) -> Dict[str, List[str]]:
    grouped: Dict[str, List[str]] = defaultdict(list)
    for record in records:
        if book_filter and record.get("book_name") != book_filter:
            continue
        page_id = str(record.get("page_id"))
        if not page_id:
            continue
        blocks = [
            record.get(f"block{idx}", "")
            for idx in range(1, 9)
            if record.get(f"block{idx}")
        ]
        if blocks:
            grouped[page_id].extend(blocks)
    return grouped


def load_font(font_path: Path | None, font_size: int) -> ImageFont.FreeTypeFont:
    if font_path is None:
        return ImageFont.load_default()
    return ImageFont.truetype(str(font_path), font_size)


def wrap_text_to_lines(text: str, width: int) -> List[str]:
    lines: List[str] = []
    for paragraph in text.split("\n"):
        paragraph = paragraph.strip()
        if not paragraph:
            lines.append("")
            continue
        wrapped = textwrap.wrap(paragraph, width=width)
        lines.extend(wrapped or [""])
    return lines


def _line_width(line: str, font: ImageFont.ImageFont) -> float:
    if hasattr(font, "getlength"):
        return float(font.getlength(line))
    bbox = font.getbbox(line or " ")
    return float(bbox[2] - bbox[0])


def measure_canvas(
    lines: Sequence[str], font: ImageFont.ImageFont, padding: int
) -> tuple[int, int, int]:
    bbox = font.getbbox("Hg")
    line_height = (bbox[3] - bbox[1]) + 6
    max_width = max((_line_width(line, font) for line in lines), default=0)
    width = int(max_width) + padding * 2
    height = line_height * max(1, len(lines)) + padding * 2
    return width, height, line_height


def draw_text_image(
    page_id: str,
    book_name: str | None,
    blocks: Sequence[str],
    font: ImageFont.ImageFont,
    line_width: int,
    padding: int,
    output_dir: Path,
) -> Path:
    text = "\n\n".join(blocks)
    lines = wrap_text_to_lines(text, width=line_width)
    width, height, line_height = measure_canvas(lines, font, padding)

    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    y = padding
    for line in lines:
        draw.text((padding, y), line, font=font, fill="black")
        y += line_height

    output_dir.mkdir(parents=True, exist_ok=True)
    safe_page_id = page_id.replace(" ", "_")
    filename = f"{(book_name or 'book').replace(' ', '_')}_{safe_page_id}.jpg"
    out_path = output_dir / filename
    image.save(out_path, format="JPEG")
    return out_path


def main() -> None:
    args = parse_args()
    records = load_dataset(args.input)
    grouped = group_blocks_by_page(records, args.book_name)
    if not grouped:
        raise RuntimeError("No matching records found to render.")

    font = load_font(args.font_path, args.font_size)

    for page_id in sorted(grouped):
        blocks = grouped[page_id]
        out_path = draw_text_image(
            page_id=page_id,
            book_name=args.book_name,
            blocks=blocks,
            font=font,
            line_width=args.line_width,
            padding=args.padding,
            output_dir=args.output_dir,
        )
        print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
