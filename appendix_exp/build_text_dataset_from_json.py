#!/usr/bin/env python3
"""Build sampled text-only OCR datasets from chapter/block JSON files.

This appendix script keeps only the text-sampling part of the original
image-rendering pipeline. It reads a dataset JSON with chapter-level blocks,
randomly samples block combinations, writes a text-only metadata JSON, and
saves a text-length histogram. It does not render JPG/PDF files.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import random
import re
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.pyplot as plt

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Block:
    key: str
    text: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sample text-only OCR records from a chapter/block JSON dataset.")
    parser.add_argument("--dataset", type=Path, default=Path("build_dataset_from_ebook/src_book/jianai.json"), help="Path to the JSON dataset containing page_id and block* fields.")
    parser.add_argument("--output_dir", type=Path, default=Path("appendix_exp/text_samples"), help="Directory where sampled text JSON and plots are stored.")
    parser.add_argument("--metadata_name", type=str, default="sampled_text_dataset.json", help="Filename for the sampled text dataset JSON.")
    parser.add_argument("--summary_name", type=str, default="sampled_text_summary.json", help="Filename for the summary JSON.")
    parser.add_argument("--output_prefix", type=str, default="jianai", help="Prefix for generated sample IDs.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducible sampling.")
    parser.add_argument("--single_count", type=int, default=200, help="Number of single-block samples.")
    parser.add_argument("--double_count", type=int, default=150, help="Number of double-block samples.")
    parser.add_argument("--triple_count", type=int, default=150, help="Number of triple-block samples.")
    parser.add_argument("--quadruple_count", type=int, default=175, help="Number of quadruple-block samples.")
    parser.add_argument("--quintuple_count", type=int, default=200, help="Number of five-block samples.")
    parser.add_argument("--sextuple_count", type=int, default=0, help="Number of six-block samples.")
    parser.add_argument("--septuple_count", type=int, default=0, help="Number of seven-block samples.")
    parser.add_argument("--octuple_count", type=int, default=0, help="Number of eight-block samples.")
    parser.add_argument("--nonuple_count", type=int, default=0, help="Number of nine-block samples.")
    parser.add_argument("--decuple_count", type=int, default=0, help="Number of ten-block samples.")
    parser.add_argument("--joiner", type=str, default="", help="String used to join sampled blocks. Use a space for English text.")
    parser.add_argument("--histogram_name", type=str, default="sample_text_length_histogram.png", help="Filename for the text_len character-count histogram.")
    parser.add_argument("--hist_bin_width", type=int, default=1000, help="Histogram bin width for text_len, where text_len is len(gt).")
    parser.add_argument("--tokenizer_name", type=str, default="", help="Optional tokenizer name/path for adding text_token_len. Empty disables tokenization.")
    return parser.parse_args()


def load_blocks_from_dataset(path: Path) -> list[Block]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError("Dataset JSON must contain a list of records.")

    blocks: list[Block] = []
    for record in data:
        if not isinstance(record, dict):
            continue
        page_id = str(record.get("page_id", "")).strip()
        if not page_id:
            continue
        block_keys = sorted((key for key in record.keys() if key.startswith("block")), key=lambda k: int(k[5:]) if k[5:].isdigit() else k)
        for key in block_keys:
            text = str(record.get(key, "")).strip()
            if text:
                blocks.append(Block(key=f"{page_id}_{key}", text=text))
    return blocks


def block_code(block: Block) -> str:
    match = re.match(r"(?P<chapter>.+?)_block(?P<idx>\d+)$", block.key)
    if not match:
        raise ValueError(f"Unexpected block key format: {block.key}")
    chapter = re.sub(r"[^0-9A-Za-z]+", "", match.group("chapter")) or "X"
    block_idx = match.group("idx")
    return f"C{chapter}B{block_idx}"


def effective_sample_count(total_blocks: int, group_size: int, requested: int) -> int:
    if requested <= 0:
        return 0
    if group_size <= 0:
        raise ValueError("group_size must be positive.")
    if total_blocks < group_size:
        raise ValueError(f"Not enough blocks ({total_blocks}) to sample group size {group_size}.")
    available = total_blocks if group_size == 1 else math.comb(total_blocks, group_size)
    if requested > available:
        logger.warning("Requested %d groups of size %d, but only %d unique combinations exist. Using %d.", requested, group_size, available, available)
    return min(requested, available)


def sample_unique_groups(indices: Sequence[int], group_size: int, count: int, rng: random.Random) -> list[tuple[int, ...]]:
    if group_size > len(indices):
        raise ValueError("group_size cannot exceed number of available blocks.")
    seen: set[tuple[int, ...]] = set()
    groups: list[tuple[int, ...]] = []
    available = list(indices)
    max_attempts = max(100, count * 20)
    attempts = 0

    while len(groups) < count and attempts < max_attempts:
        selection = tuple(sorted(rng.sample(available, group_size)))
        if selection not in seen:
            seen.add(selection)
            groups.append(selection)
        attempts += 1

    if len(groups) < count:
        raise RuntimeError(f"Unable to sample {count} unique groups of size {group_size}. Reduce the requested count or ensure more blocks exist.")
    return groups


def load_tokenizer(tokenizer_name: str):
    if not tokenizer_name:
        return None
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(tokenizer_name)


def compute_text_token_len(text: str, tokenizer, cache: dict[str, int]) -> int | None:
    if tokenizer is None:
        return None
    if text in cache:
        return cache[text]
    tokens = tokenizer(text, add_special_tokens=True, truncation=False)
    length = len(tokens.get("input_ids", []))
    cache[text] = length
    return length


def build_records(args: argparse.Namespace, blocks: Sequence[Block], rng: random.Random) -> list[dict]:
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
    tokenizer = load_tokenizer(args.tokenizer_name)
    token_cache: dict[str, int] = {}
    records: list[dict] = []

    for label, group_size, requested_count in sampling_plan:
        count = effective_sample_count(len(all_indices), group_size, requested_count)
        if count == 0:
            continue
        if group_size == 1:
            sampled = rng.sample(all_indices, count)
            groups = [(idx,) for idx in sampled]
        else:
            groups = sample_unique_groups(all_indices, group_size, count, rng)

        for combo in groups:
            selected_blocks = [blocks[idx] for idx in combo]
            combined_text = args.joiner.join(block.text for block in selected_blocks)
            codes = [block_code(block) for block in selected_blocks]
            sample_id = f"{args.output_prefix}_{label}_{'_'.join(codes)}"
            text_len = len(combined_text)
            record = {
                "sample_id": sample_id,
                "gt": combined_text,
                "text_len": text_len,
                "group_label": label,
                "blocks": codes,
            }
            token_len = compute_text_token_len(combined_text, tokenizer, token_cache)
            if token_len is not None:
                record["text_token_len"] = token_len
            records.append(record)

    return records


def save_text_length_histogram(records: Sequence[dict], output_path: Path, bin_width: int) -> None:
    lengths = [int(record["text_len"]) for record in records if "text_len" in record]
    if not lengths:
        logger.warning("No text_len values found; skipping histogram.")
        return

    safe_bin_width = max(1, int(bin_width))
    max_len = max(lengths)
    upper = ((max_len // safe_bin_width) + 1) * safe_bin_width
    bins = list(range(0, upper + safe_bin_width, safe_bin_width))
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(lengths, bins=bins, color="#4C78A8", edgecolor="white")
    ax.set_xlabel("text_len = len(gt)")
    ax.set_ylabel("Sample count")
    ax.set_title(f"Sample text_len Distribution (bin={safe_bin_width})")
    ax.set_xticks(bins)
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    logger.info("Saved text-length histogram to %s", output_path)


def summarize_records(records: Sequence[dict], blocks: Sequence[Block], args: argparse.Namespace) -> dict:
    lengths = [int(record["text_len"]) for record in records]
    counts_by_group: dict[str, int] = {}
    for record in records:
        label = str(record["group_label"])
        counts_by_group[label] = counts_by_group.get(label, 0) + 1
    return {
        "dataset": str(args.dataset),
        "total_blocks": len(blocks),
        "total_samples": len(records),
        "counts_by_group": counts_by_group,
        "text_len": {
            "min": min(lengths) if lengths else None,
            "max": max(lengths) if lengths else None,
            "mean": sum(lengths) / len(lengths) if lengths else None,
        },
        "joiner": args.joiner,
        "seed": args.seed,
    }


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    blocks = load_blocks_from_dataset(args.dataset)
    if not blocks:
        raise RuntimeError("No non-empty blocks found in the dataset.")

    records = build_records(args, blocks, rng)
    if not records:
        raise RuntimeError("No samples were generated. Check sample count arguments.")

    metadata_path = args.output_dir / args.metadata_name
    metadata_path.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("Saved %d text samples to %s", len(records), metadata_path)

    summary = summarize_records(records, blocks, args)
    summary_path = args.output_dir / args.summary_name
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("Saved summary to %s", summary_path)

    histogram_path = args.output_dir / args.histogram_name
    save_text_length_histogram(records, histogram_path, args.hist_bin_width)


if __name__ == "__main__":
    main()
