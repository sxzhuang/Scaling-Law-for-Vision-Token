#!/usr/bin/env python3
"""Add text_len based on gt character counts."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any


logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
LOGGER = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Add text_len to JSON records based on gt character counts.")
    parser.add_argument("--json_path", type=Path, default=Path("build_dataset_from_ebook/pride_dataset_square/pride_square_gt.json"), help="Path to the JSON file to update.")
    parser.add_argument("--output_json", type=Path, default=None, help="Optional output path (defaults to overwrite input).")
    return parser.parse_args()


def compute_text_len(text: Any) -> int:
    if not isinstance(text, str):
        return 0
    return len(text)


def add_text_len_to_records(records: list[dict[str, Any]]) -> int:
    updated = 0
    for record in records:
        if not isinstance(record, dict):
            continue
        record["text_len"] = compute_text_len(record.get("gt", ""))
        updated += 1
    return updated


def add_text_len(data: Any) -> int:
    if isinstance(data, list):
        return add_text_len_to_records(data)
    if isinstance(data, dict):
        return add_text_len_to_records(list(data.values()))
    raise ValueError("JSON root must be a list or dict of records.")


def main() -> None:
    args = parse_args()
    if not args.json_path.exists():
        raise FileNotFoundError(f"JSON file not found: {args.json_path}")
    data = json.loads(args.json_path.read_text(encoding="utf-8"))
    updated = add_text_len(data)
    output_path = args.output_json or args.json_path
    output_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    LOGGER.info("Updated %d records and wrote %s", updated, output_path)


if __name__ == "__main__":
    main()
