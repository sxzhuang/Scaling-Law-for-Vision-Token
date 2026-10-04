#!/usr/bin/env python3
"""Build JSON records from the "Jing Shi Tong Yan" text."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from build_dataset_from_ebook.build_json_from_txt import chunk_sentences, load_text, normalize_paragraph

VOLUME_PATTERN = re.compile(r"^\s*第([一二三四五六七八九十百千万零〇两\d]+)章[^\r\n]*", re.MULTILINE)
SENTENCE_PATTERN = re.compile(r".+?(?:[。！？；][”’\"'」』】）]*)|.+$", re.DOTALL)
END_MARKER = "End of Project Gutenberg's Jing Shi Tong Yan, by Feng Meng Long Feng"
INTERNAL_SPACE_PATTERN = re.compile(r"[ \u3000]+")
SEPARATOR_PATTERN = re.compile(r"\s*-{8,}\s*")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Split jingshitongyan.txt into volume blocks.")
    parser.add_argument("--input", type=Path, default=Path("build_dataset_from_ebook/src_book/jingshitongyan.txt"), help="Path to the source text file.")
    parser.add_argument("--output", type=Path, default=Path("build_dataset_from_ebook/src_book/jingshi_dataset.json"), help="Destination JSON file for the generated dataset.")
    parser.add_argument("--book_name", default="jingshi", help='Book identifier stored under the "book_name" field.')
    parser.add_argument("--blocks", type=int, default=4, help="Number of sentence-aligned blocks per volume.")
    return parser.parse_args()


def chinese_numeral_to_int(value: str) -> str:
    if value.isdigit():
        return value

    digit_map = {
        "零": 0,
        "〇": 0,
        "一": 1,
        "二": 2,
        "三": 3,
        "四": 4,
        "五": 5,
        "六": 6,
        "七": 7,
        "八": 8,
        "九": 9,
        "两": 2,
    }
    unit_map = {"十": 10, "百": 100, "千": 1000, "万": 10000}

    total = 0
    current = 0
    num = 0
    for ch in value:
        if ch in digit_map:
            num = digit_map[ch]
            continue
        if ch not in unit_map:
            return value
        unit = unit_map[ch]
        if num == 0:
            num = 1
        current += num * unit
        num = 0
        if unit == 10000:
            total += current
            current = 0

    total += current + num
    return str(total) if total > 0 else value


def split_sentences_zh(paragraph: str) -> list[str]:
    if not paragraph:
        return []
    return [sentence.strip() for sentence in SENTENCE_PATTERN.findall(paragraph) if sentence.strip()]


def remove_internal_spaces(text: str) -> str:
    return INTERNAL_SPACE_PATTERN.sub("", text).strip()


def remove_chapter_separator(text: str) -> str:
    return SEPARATOR_PATTERN.sub(" ", text)


def truncate_at_end_marker(text: str) -> str:
    marker_index = text.find(END_MARKER)
    if marker_index == -1:
        return text
    return text[:marker_index].rstrip()


def extract_volumes(text: str) -> list[tuple[str, str]]:
    matches = list(VOLUME_PATTERN.finditer(text))
    volumes: list[tuple[str, str]] = []
    for idx, match in enumerate(matches):
        start = match.end()
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(text)
        body = text[start:end].strip()
        if not body:
            continue
        raw_volume_id = match.group(1).strip()
        volume_id = chinese_numeral_to_int(raw_volume_id)
        volumes.append((volume_id, body))
    return volumes


def build_dataset(volumes: list[tuple[str, str]], block_count: int, book_name: str) -> list[dict]:
    dataset: list[dict] = []
    for page_id, raw_body in volumes:
        normalized = normalize_paragraph(remove_chapter_separator(raw_body))
        sentences = split_sentences_zh(normalized)
        blocks = [remove_internal_spaces(block) for block in chunk_sentences(sentences, block_count)]
        record = {"book_name": book_name, "page_id": page_id}
        for idx, block_text in enumerate(blocks, start=1):
            record[f"block{idx}"] = block_text
        dataset.append(record)
    return dataset


def main() -> None:
    args = parse_args()
    text = truncate_at_end_marker(load_text(args.input))
    volumes = extract_volumes(text)
    if not volumes:
        raise RuntimeError(
            f"No volume markers found in {args.input}. "
            "Ensure lines start with '第一卷' style headings."
        )
    dataset = build_dataset(volumes, args.blocks, args.book_name)
    args.output.write_text(json.dumps(dataset, indent=2, ensure_ascii=False), encoding="utf-8")
    print(
        f"Wrote {len(dataset)} records with {args.blocks} blocks each to {args.output}"
    )


if __name__ == "__main__":
    main()

# python -m build_dataset_from_ebook.build_json_from_txt_jingshi --blocks 8
# python -m build_dataset_from_ebook.build_json_from_txt_jingshi --input build_dataset_from_ebook/src_book/jianai.txt --output build_dataset_from_ebook/src_book/jianai.json --book_name "jianai" --blocks 8
