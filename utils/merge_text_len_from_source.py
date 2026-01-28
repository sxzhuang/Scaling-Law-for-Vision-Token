#!/usr/bin/env python3
"""Fill text_len in a JSON by matching ids to a source metadata file."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any


logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
LOGGER = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Copy text_len values from a source JSON into a target JSON.")
    parser.add_argument("--source_json", type=Path, default=Path("build_dataset_from_ebook/pride_dataset_square/pride_square_gt.json"), help="Source JSON with image/text_len fields.")
    parser.add_argument("--modify_json", type=Path, default=Path("eval_results/pride_square/512/dpsk_eval_metric.json"), help="Target JSON to update based on id matching.")
    parser.add_argument("--output_json", type=Path, default=None, help="Optional output path (defaults to overwrite modify_json).")
    return parser.parse_args()


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def iter_records(data: Any) -> list[dict[str, Any]]:
    if isinstance(data, list):
        return [item for item in data if isinstance(item, dict)]
    if isinstance(data, dict):
        return [item for item in data.values() if isinstance(item, dict)]
    raise ValueError("JSON root must be a list or dict of records.")


def build_text_len_map(source_data: Any) -> dict[str, int]:
    mapping: dict[str, int] = {}
    for record in iter_records(source_data):
        image_name = record.get("image")
        if not isinstance(image_name, str) or not image_name.strip():
            continue
        text_len = record.get("text_len")
        if text_len is None:
            continue
        try:
            text_len_value = int(text_len)
        except (TypeError, ValueError):
            continue
        mapping[Path(image_name).stem] = text_len_value
    return mapping


def update_records(target_data: Any, text_len_map: dict[str, int]) -> tuple[int, int]:
    updated = 0
    missing = 0
    for record in iter_records(target_data):
        record_id = record.get("id")
        if not isinstance(record_id, str) or not record_id.strip():
            continue
        text_len = text_len_map.get(record_id)
        if text_len is None:
            missing += 1
            continue
        record["text_len"] = text_len
        updated += 1
    return updated, missing


def main() -> None:
    args = parse_args()
    if not args.source_json.exists():
        raise FileNotFoundError(f"Source JSON not found: {args.source_json}")
    if not args.modify_json.exists():
        raise FileNotFoundError(f"Target JSON not found: {args.modify_json}")

    source_data = load_json(args.source_json)
    target_data = load_json(args.modify_json)
    text_len_map = build_text_len_map(source_data)
    if not text_len_map:
        raise RuntimeError("No text_len entries found in source_json.")

    updated, missing = update_records(target_data, text_len_map)
    output_path = args.output_json or args.modify_json
    output_path.write_text(json.dumps(target_data, ensure_ascii=False, indent=2), encoding="utf-8")
    LOGGER.info("Updated %d records; %d ids missing text_len", updated, missing)
    LOGGER.info("Wrote %s", output_path)


if __name__ == "__main__":
    main()
