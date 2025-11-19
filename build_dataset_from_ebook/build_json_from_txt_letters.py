#!/usr/bin/env python3
"""Build JSON records from 'Letters' text using letter boundaries."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from build_dataset_from_ebook.build_json_from_txt import build_dataset, load_text

LETTER_START_PATTERN = re.compile(r"^\s*_Dear\s+Pierrepont:_", re.IGNORECASE)
LETTER_END_LINE = "your affectionate father,"
SIGNATURE_LINE = "john graham."


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Split letters.txt into Pierrepont letters.")
    parser.add_argument("--input", type=Path, default=Path("build_dataset_from_ebook/src_book/letters.txt"), help="Path to the source letters text file.")
    parser.add_argument("--output", type=Path, default=Path("build_dataset_from_ebook/src_book/letters_dataset.json"), help="Destination JSON file for the generated dataset.")
    parser.add_argument("--book_name", default="letters", help='Book identifier stored under the "book_name" field.')
    parser.add_argument("--blocks", type=int, default=4, help="Number of sentence-aligned blocks per letter.")
    return parser.parse_args()


def extract_letters(text: str) -> list[tuple[str, str]]:
    lines = text.splitlines()
    total = len(lines)
    letters: list[tuple[str, str]] = []
    idx = 0

    while idx < total:
        if not LETTER_START_PATTERN.match(lines[idx]):
            idx += 1
            continue

        start_idx = idx
        idx += 1
        end_idx = None

        while idx < total:
            current = lines[idx].strip().lower()
            if current == LETTER_END_LINE:
                if idx + 1 < total and lines[idx + 1].strip().lower() == SIGNATURE_LINE:
                    end_idx = idx + 2
                    idx = end_idx
                    break
            idx += 1

        if end_idx is None:
            end_idx = total
            idx = total

        body = "\n".join(lines[start_idx:end_idx]).strip()
        if not body:
            continue

        page_id = str(len(letters) + 1)
        letters.append((page_id, body))

    return letters


def main() -> None:
    args = parse_args()
    text = load_text(args.input)
    letters = extract_letters(text)
    if not letters:
        raise RuntimeError(
            f"No letters were found in {args.input}. Ensure the start/end markers are present."
        )
    dataset = build_dataset(letters, args.blocks, args.book_name)
    args.output.write_text(json.dumps(dataset, indent=2, ensure_ascii=False), encoding="utf-8")
    print(
        f"Wrote {len(dataset)} records with {args.blocks} blocks each to {args.output}"
    )


if __name__ == "__main__":
    main()
