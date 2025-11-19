#!/usr/bin/env python3
"""Remove oversized text_token_len entries and corresponding artifacts."""

from __future__ import annotations

import json
from pathlib import Path

THRESHOLD = 5627


def find_square_dirs(root: Path) -> list[Path]:
    return sorted([path for path in root.iterdir() if path.is_dir() and path.name.endswith("_square")])


def process_json(json_path: Path) -> tuple[list[dict], list[dict]]:
    data = json.loads(json_path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"{json_path} must contain a list of records.")
    kept: list[dict] = []
    removed: list[dict] = []
    for record in data:
        if not isinstance(record, dict):
            continue
        token_len = record.get("text_token_len", 0)
        if isinstance(token_len, (int, float)) and token_len >= THRESHOLD:
            removed.append(record)
        else:
            kept.append(record)
    return kept, removed


def delete_artifacts(square_dir: Path, records: list[dict]) -> None:
    images_dir = square_dir / "images"
    pdf_dir = square_dir / "pdf"
    for record in records:
        image_name = record.get("image")
        pdf_name = record.get("pdf")
        if image_name:
            image_path = images_dir / image_name
            if image_path.exists():
                image_path.unlink()
        if pdf_name:
            pdf_path = pdf_dir / pdf_name
            if pdf_path.exists():
                pdf_path.unlink()


def main() -> None:
    root = Path("build_dataset_from_ebook")
    square_dirs = find_square_dirs(root)
    if not square_dirs:
        raise RuntimeError(f"No *_square directories found under {root}")

    for square_dir in square_dirs:
        json_files = sorted(square_dir.glob("*.json"))
        if len(json_files) != 1:
            continue
        json_path = json_files[0]
        kept, removed = process_json(json_path)
        if not removed:
            continue
        delete_artifacts(square_dir, removed)
        json_path.write_text(json.dumps(kept, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"{square_dir.name}: removed {len(removed)} records ≥ {THRESHOLD}")


if __name__ == "__main__":
    main()
