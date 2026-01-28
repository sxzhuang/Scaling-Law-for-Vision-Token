#!/usr/bin/env python3
"""Aggregate *_square eval results into a CSV."""

from __future__ import annotations

import argparse
import csv
import json
import logging
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate *_square eval results into a CSV.")
    parser.add_argument("--eval_root", type=Path, default=Path("eval_results"), help="Root directory containing *_square folders.")
    parser.add_argument("--output_csv", type=Path, default=Path("data_analysis/data4analysis_alltype_default.csv"), help="Output CSV path.")
    parser.add_argument("--font_size", type=int, default=28, help="Font size to record in output.")
    parser.add_argument("--line_spacing", type=int, default=6, help="Line spacing to record in output.")
    parser.add_argument("--letter_spacing", type=int, default=0, help="Letter spacing to record in output.")
    return parser.parse_args()


def load_eval_records(eval_path: Path) -> list[dict[str, Any]]:
    try:
        payload: Any = json.loads(eval_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        logging.warning("Failed to parse %s: %s", eval_path, exc)
        return []
    if not isinstance(payload, list):
        logging.warning("Expected list in %s, got %s", eval_path, type(payload).__name__)
        return []
    return [item for item in payload if isinstance(item, dict)]


def parse_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def parse_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = parse_args()
    eval_root = args.eval_root
    if not eval_root.exists():
        raise FileNotFoundError(f"Eval root not found: {eval_root}")

    rows: list[dict[str, Any]] = []
    dataset_dirs = sorted([path for path in eval_root.glob("*_square") if path.is_dir()])
    for dataset_dir in dataset_dirs:
        for res_dir in sorted([path for path in dataset_dir.iterdir() if path.is_dir()]):
            if not res_dir.name.isdigit():
                continue
            resolution = int(res_dir.name)
            eval_path = res_dir / "dpsk_eval_metric.json"
            if not eval_path.exists():
                logging.warning("Missing %s", eval_path)
                continue
            for record in load_eval_records(eval_path):
                sample_id = str(record.get("id", "")).strip()
                if not sample_id:
                    continue
                edit_distance = parse_float(record.get("edit_dist"))
                if edit_distance is None:
                    continue
                text_len = parse_int(record.get("text_len"))
                rows.append(
                    {
                        "image_id": sample_id,
                        "resolution": resolution,
                        "font_size": args.font_size,
                        "line_spacing": args.line_spacing,
                        "letter_spacing": args.letter_spacing,
                        "edit_distance": edit_distance,
                        "text_len": text_len,
                    }
                )

    output_csv = args.output_csv
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["image_id", "resolution", "font_size", "line_spacing", "letter_spacing", "edit_distance", "text_len"]
    with output_csv.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    logging.info("Aggregated %s rows into %s", len(rows), output_csv)


if __name__ == "__main__":
    main()
