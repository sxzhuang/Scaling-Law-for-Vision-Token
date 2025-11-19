#!/usr/bin/env python3
"""Build JSON records from a newspaper text using separator lines."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from build_dataset_from_ebook.build_json_from_txt import (
    chunk_sentences,
    load_text,
    normalize_paragraph,
    split_sentences,
)

SEPARATOR_PATTERN = re.compile(r"^\s*-{5,}\s*$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Split newspaper.txt into separator-defined blocks.")
    parser.add_argument("--input", type=Path, default=Path("build_dataset_from_ebook/src_book/newspaper.txt"), help="Path to the source newspaper text file.")
    parser.add_argument("--output", type=Path, default=Path("build_dataset_from_ebook/src_book/newspaper_dataset.json"), help="Destination JSON file for the generated dataset.")
    parser.add_argument("--book_name", default="newspaper", help='Identifier stored under the "book_name" field.')
    parser.add_argument("--blocks", type=int, default=4, help="Number of sentence-aligned blocks per article segment.")
    return parser.parse_args()


def extract_sections(text: str) -> list[tuple[str, str]]:
    lines = text.splitlines()
    sections: list[str] = []
    current: list[str] = []

    for line in lines:
        if SEPARATOR_PATTERN.match(line):
            if current:
                section = "\n".join(current).strip()
                if section:
                    sections.append(section)
                current = []
            continue
        current.append(line)

    if current:
        section = "\n".join(current).strip()
        if section:
            sections.append(section)

    return [(str(idx + 1), section) for idx, section in enumerate(sections)]


def build_dataset(sections: list[tuple[str, str]], blocks_per_section: int, book_name: str) -> list[dict]:
    dataset: list[dict] = []
    for page_id, raw_text in sections:
        normalized = normalize_paragraph(raw_text)
        sentences = split_sentences(normalized)
        if not sentences:
            continue
        blocks = chunk_sentences(sentences, blocks_per_section)
        record = {"book_name": book_name, "page_id": page_id}
        for block_idx, block_text in enumerate(blocks, start=1):
            record[f"block{block_idx}"] = block_text
        dataset.append(record)
    return dataset


def main() -> None:
    args = parse_args()
    text = load_text(args.input)
    sections = extract_sections(text)
    if len(sections) > 3:
        sections = [(str(idx + 1), section) for idx, (_, section) in enumerate(sections[3:])]
    else:
        sections = []
    if not sections:
        raise RuntimeError(
            f"No sections were detected in {args.input}. Ensure separator lines consist of hyphens."
        )
    dataset = build_dataset(sections, args.blocks, args.book_name)
    args.output.write_text(json.dumps(dataset, indent=2, ensure_ascii=False), encoding="utf-8")
    print(
        f"Wrote {len(dataset)} records with {args.blocks} blocks each to {args.output}"
    )


if __name__ == "__main__":
    main()
