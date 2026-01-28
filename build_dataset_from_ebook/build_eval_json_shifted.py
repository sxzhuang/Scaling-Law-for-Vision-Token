#!/usr/bin/env python3
"""Assemble eval JSON for images with padded-shift suffixes in their filenames."""

from __future__ import annotations

import argparse
import json
import logging
import math
import re
from pathlib import Path
from typing import Dict, List

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
LOGGER = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build eval JSON for padded/shifted images.")
    parser.add_argument("--images_dir", type=Path, default=Path("build_dataset_from_ebook/pride_dataset_square_large_variance/selected_from1024/images_paddling_shift"), help="Directory containing padded-shift JPG images.")
    parser.add_argument("--metadata_path", type=Path, default=Path("build_dataset_from_ebook/pride_dataset_square/pride_square_gt.json"), help="JSON file with ground-truth entries.")
    parser.add_argument("--predictions_dir", type=Path, default=Path("ocr_predictions/pride_dataset_square_large_variance/selected_from1024/1024"), help="Directory containing prediction .md files.")
    parser.add_argument("--output_path", type=Path, default=Path("eval_results/pride_square_large_variance/selected_from1024/prepare_json_for_eval.json"), help="Destination JSON file.")
    return parser.parse_args()


def load_metadata(path: Path) -> Dict[str, Dict[str, str]]:
    entries = json.loads(path.read_text(encoding="utf-8"))
    mapping: Dict[str, Dict[str, str]] = {}
    for record in entries:
        image_name = record.get("image")
        gt_text = record.get("gt")
        token_len = record.get("text_token_len")
        text_len = record.get("text_len")
        if not image_name or gt_text is None:
            continue
        mapping[str(image_name)] = {
            "gt": str(gt_text),
            "text_token_len": token_len,
            "text_len": text_len,
        }
    return mapping


def strip_shift_suffix(stem: str) -> str:
    parts = stem.rsplit("_", 2)
    if len(parts) >= 3 and parts[-1].isdigit() and parts[-2].isdigit():
        return "_".join(parts[:-2])
    return stem


def md_to_text(md_path: Path) -> str:
    raw = md_path.read_text(encoding="utf-8")
    text = re.sub(r"!\[[^\]]*\]\([^)]+\)", " ", raw)
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)
    lines: List[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        stripped = re.sub(r"^[#>*\-\d\.\)\(]+\s*", "", stripped)
        lines.append(stripped)
    return " ".join(lines).strip()


def main() -> None:
    args = parse_args()
    metadata = load_metadata(args.metadata_path)
    if not metadata:
        raise RuntimeError("Metadata file contained no valid entries.")

    image_paths = sorted(args.images_dir.glob("*.jpg"))
    if not image_paths:
        raise RuntimeError(f"No JPG images found under {args.images_dir}.")

    records: List[Dict[str, str]] = []
    missing_gt: List[str] = []
    missing_pred: List[str] = []
    checkpoints = sorted({min(len(image_paths), math.ceil(len(image_paths) * frac / 4)) for frac in range(1, 5)})

    for idx, image_path in enumerate(image_paths, start=1):
        image_id = image_path.stem
        base_id = strip_shift_suffix(image_id)
        possible_names = [image_path.name, f"{base_id}{image_path.suffix}"]

        meta_entry = None
        for candidate in possible_names:
            meta_entry = metadata.get(candidate)
            if meta_entry:
                break
        if meta_entry is None:
            missing_gt.append(image_path.name)
            continue

        pred_candidates = [args.predictions_dir / f"{image_id}.md", args.predictions_dir / f"{base_id}.md"]
        pred_path = None
        for candidate in pred_candidates:
            if candidate.exists():
                pred_path = candidate
                break
        if pred_path is None:
            missing_pred.append(image_id)
            continue

        pred_text = md_to_text(pred_path)
        records.append(
            {
                "id": image_id,
                "label": meta_entry["gt"],
                "answer": pred_text,
                "text_token_len": meta_entry.get("text_token_len"),
                "text_len": meta_entry.get("text_len"),
            }
        )

        if idx in checkpoints:
            percent = (idx / len(image_paths)) * 100
            LOGGER.info("Processed %d/%d images (%.1f%%)", idx, len(image_paths), percent)

    if missing_gt:
        LOGGER.warning("Missing GT for %d images (e.g., %s)", len(missing_gt), missing_gt[:5])
    if missing_pred:
        LOGGER.warning("Missing predictions for %d image ids (e.g., %s)", len(missing_pred), missing_pred[:5])
    if not records:
        raise RuntimeError("No eval records were generated. Check inputs.")

    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    args.output_path.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
    LOGGER.info("Wrote %d eval entries to %s", len(records), args.output_path)


if __name__ == "__main__":
    main()
