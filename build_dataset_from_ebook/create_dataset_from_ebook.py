#!/usr/bin/env python3
"""Create a structured dataset from the Pride and Prejudice ebook.

The script extracts each chapter (identified by lines beginning with
``CHAPTER ...``), normalizes whitespace into single-paragraph blocks,
and splits the content of every chapter into eight sentence-aligned
segments. The resulting records are written to JSON so they can be fed
into downstream pipelines.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import List, Sequence

CHAPTER_PATTERN = re.compile(r"^\s*CHAPTER\s+([A-Z0-9]+)[^\r\n]*", re.MULTILINE)
SENTENCE_SPLIT_PATTERN = re.compile(r"(?<=[.!?])\s+")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Split pride-and-prejudice.txt into chapter blocks."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("pride-and-prejudice.txt"),
        help="Path to the source ebook text file.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("pride_and_prejudice_dataset.json"),
        help="Destination JSON file for the generated dataset.",
    )
    parser.add_argument(
        "--book-name",
        default="pride_and_prejudice",
        help='Book identifier stored under the "book_name" field.',
    )
    parser.add_argument(
        "--blocks",
        type=int,
        default=4,
        help="Number of sentence-aligned blocks per chapter.",
    )
    return parser.parse_args()


def load_text(path: Path) -> str:
    text = path.read_text(encoding="utf-8")
    return text.replace("\ufeff", "")


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


def build_dataset(
    chapters: Sequence[tuple[str, str]], block_count: int, book_name: str
) -> List[dict]:
    dataset: List[dict] = []
    for page_id, raw_body in chapters:
        normalized = normalize_paragraph(raw_body)
        sentences = split_sentences(normalized)
        blocks = chunk_sentences(sentences, block_count)
        record = {"book_name": book_name, "page_id": page_id}
        for idx, block_text in enumerate(blocks, start=1):
            record[f"block{idx}"] = block_text
        dataset.append(record)
    return dataset


def main() -> None:
    args = parse_args()
    text = load_text(args.input)
    chapters = extract_chapters(text)
    if not chapters:
        raise RuntimeError(
            f"No chapter markers found in {args.input}. "
            "Ensure lines start with 'CHAPTER'."
        )
    dataset = build_dataset(chapters, args.blocks, args.book_name)
    args.output.write_text(
        json.dumps(dataset, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(
        f"Wrote {len(dataset)} records with {args.blocks} blocks each to {args.output}"
    )


if __name__ == "__main__":
    main()
