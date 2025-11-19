#!/usr/bin/env python3
"""Create dataset slices from the medicine textbook."""

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

CHAPTER_PATTERN = re.compile(r"^\s*CHAPTER\s+([IVXLCDM]+|\d+)\.\s*$", re.IGNORECASE)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Split medicine.txt into chapter blocks.")
    parser.add_argument("--input", type=Path, default=Path("build_dataset_from_ebook/src_book/medicine.txt"), help="Path to the medicine text file.")
    parser.add_argument("--output", type=Path, default=Path("build_dataset_from_ebook/src_book/medicine_dataset.json"), help="Destination JSON file for the generated dataset.")
    parser.add_argument("--book_name", default="medicine", help='Book identifier stored under the "book_name" field.')
    parser.add_argument("--chunk_num", type=int, default=6, help="Number of pages per subchapter.")
    parser.add_argument("--blocks", type=int, default=4, help="Number of blocks per page.")
    return parser.parse_args()


def extract_chapters(text: str) -> list[tuple[str, str]]:
    lines = text.splitlines()
    total = len(lines)
    chapters: list[tuple[str, str]] = []
    idx = 0
    while idx < total:
        match = CHAPTER_PATTERN.match(lines[idx])
        if not match:
            idx += 1
            continue
        raw_id = match.group(1).strip()
        idx += 1

        while idx < total:
            stripped = lines[idx].strip()
            if not stripped:
                idx += 1
                continue
            if stripped.upper() == stripped:
                idx += 1
                continue
            break

        start_idx = idx
        while idx < total and not CHAPTER_PATTERN.match(lines[idx]):
            idx += 1
        body = "\n".join(lines[start_idx:idx]).strip()
        if not body:
            continue
        chapters.append((str(len(chapters) + 1), body))
    return chapters


def chunk_sentences_by_word_count(sentences: list[str], chunk_count: int) -> list[str]:
    if chunk_count <= 0:
        raise ValueError("chunk_count must be greater than zero.")
    if not sentences:
        return [""] * chunk_count

    word_counts = [len(sentence.split()) for sentence in sentences]
    total_words = sum(word_counts)
    target_words = total_words / chunk_count if chunk_count else total_words

    chunks: list[str] = []
    current_sentences: list[str] = []
    current_words = 0

    for idx, (sentence, words) in enumerate(zip(sentences, word_counts)):
        current_sentences.append(sentence)
        current_words += words
        remaining_sentences = len(sentences) - idx - 1
        remaining_chunks = chunk_count - len(chunks) - 1
        must_split = remaining_chunks > 0 and remaining_sentences <= remaining_chunks
        should_split = (
            remaining_chunks > 0
            and remaining_sentences > remaining_chunks
            and current_words >= target_words
        )
        if must_split or should_split:
            chunks.append(" ".join(current_sentences).strip())
            current_sentences = []
            current_words = 0

    chunks.append(" ".join(current_sentences).strip())

    if len(chunks) < chunk_count:
        chunks.extend([""] * (chunk_count - len(chunks)))
    elif len(chunks) > chunk_count:
        tail = " ".join(chunks[chunk_count - 1 :]).strip()
        chunks = chunks[: chunk_count - 1]
        chunks.append(tail)
    return chunks


def count_words(sentences: list[str]) -> int:
    return sum(len(sentence.split()) for sentence in sentences)


def build_dataset(chapters: list[tuple[str, str]], book_name: str, chunk_num: int, blocks_per_page: int) -> list[dict]:
    dataset: list[dict] = []
    page_counter = 1
    if not chapters:
        return dataset

    reference_index = 2 if len(chapters) > 2 else 0
    reference_sentences = split_sentences(normalize_paragraph(chapters[reference_index][1]))
    reference_words = count_words(reference_sentences) or 1

    for _, raw_body in chapters:
        normalized = normalize_paragraph(raw_body)
        sentences = split_sentences(normalized)
        if not sentences:
            continue
        current_words = count_words(sentences)
        ratio = current_words / reference_words if reference_words else 1.0
        subchapter_count = max(1, int(ratio)) if ratio >= 1.0 else 1
        subchapters = chunk_sentences_by_word_count(sentences, subchapter_count)

        for sub_text in subchapters:
            if not sub_text:
                continue
            sub_sentences = split_sentences(sub_text)
            if not sub_sentences:
                continue
            pages = chunk_sentences_by_word_count(sub_sentences, chunk_num)
            for page_text in pages:
                if not page_text:
                    continue
                page_sentences = split_sentences(page_text)
                blocks = chunk_sentences(page_sentences, blocks_per_page)
                record = {
                    "book_name": book_name,
                    "page_id": str(page_counter),
                }
                for block_idx, text in enumerate(blocks, start=1):
                    record[f"block{block_idx}"] = text
                dataset.append(record)
                page_counter += 1
    return dataset


def main() -> None:
    args = parse_args()
    text = load_text(args.input)
    chapters = extract_chapters(text)
    if not chapters:
        raise RuntimeError(
            f"No chapter markers found in {args.input}. Ensure lines follow the expected 'CHAPTER <num>.' format."
        )
    dataset = build_dataset(chapters, args.book_name, args.chunk_num, args.blocks)
    args.output.write_text(json.dumps(dataset, indent=2, ensure_ascii=False), encoding="utf-8")
    print(
        f"Wrote {len(dataset)} records with {args.blocks} blocks per subchapter to {args.output}"
    )


if __name__ == "__main__":
    main()
