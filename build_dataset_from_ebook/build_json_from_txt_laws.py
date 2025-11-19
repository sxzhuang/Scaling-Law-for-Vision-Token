#!/usr/bin/env python3
"""Build JSON records from the Postconviction Remedies text."""

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

CHAPTER_PATTERN = re.compile(r"^\s*Chapter\s+(\d+)\s*:[^\r\n]*", re.MULTILINE | re.IGNORECASE)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Split Laws_PostconvictionRemedies.txt into chapter blocks.")
    parser.add_argument("--input", type=Path, default=Path("build_dataset_from_ebook/src_book/Laws_PostconvictionRemedies.txt"), help="Path to the source text file.")
    parser.add_argument("--output", type=Path, default=Path("build_dataset_from_ebook/src_book/Laws_PostconvictionRemedies_dataset.json"), help="Destination JSON file for the generated dataset.")
    parser.add_argument("--book_name", default="laws_postconviction_remedies", help='Book identifier stored under the "book_name" field.')
    parser.add_argument("--chunk_num", type=int, default=6, help="Number of pages per chapter.")
    parser.add_argument("--blocks", type=int, default=4, help="Number of blocks per page.")
    return parser.parse_args()


def extract_chapters(text: str) -> list[tuple[str, str]]:
    matches = list(CHAPTER_PATTERN.finditer(text))
    chapters: list[tuple[str, str]] = []
    for idx, match in enumerate(matches):
        start = match.end()
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(text)
        body = text[start:end].strip()
        if not body:
            continue
        chapters.append((match.group(1), body))
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


def build_dataset(
    chapters: list[tuple[str, str]], book_name: str, chunk_num: int, blocks_per_page: int
) -> list[dict]:
    dataset: list[dict] = []
    page_counter = 1
    if not chapters:
        return dataset

    standard_sentences = split_sentences(normalize_paragraph(chapters[0][1]))
    standard_words = count_words(standard_sentences) or 1

    for chapter_id, raw_body in chapters:
        normalized = normalize_paragraph(raw_body)
        sentences = split_sentences(normalized)
        if not sentences:
            continue
        chapter_words = count_words(sentences)
        ratio = chapter_words / standard_words if standard_words else 1.0
        virtual_chapter_count = max(1, int(ratio)) if ratio > 0 else 1
        virtual_chunks = chunk_sentences_by_word_count(sentences, virtual_chapter_count)
        for virtual_text in virtual_chunks:
            if not virtual_text:
                continue
            virtual_sentences = split_sentences(virtual_text)
            if not virtual_sentences:
                continue
            pages = chunk_sentences_by_word_count(virtual_sentences, chunk_num)
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
            f"No chapter markers found in {args.input}. Ensure lines follow 'Chapter <num>:' format."
        )
    dataset = build_dataset(chapters, args.book_name, args.chunk_num, args.blocks)
    args.output.write_text(json.dumps(dataset, indent=2, ensure_ascii=False), encoding="utf-8")
    print(
        f"Wrote {len(dataset)} records with {args.blocks} blocks per page across {args.chunk_num} pages per chapter to {args.output}"
    )


if __name__ == "__main__":
    main()
