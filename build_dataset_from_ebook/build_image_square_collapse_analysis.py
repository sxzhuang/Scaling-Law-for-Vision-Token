#!/usr/bin/env python3
"""Analyze metric JSON for collapse-related text length thresholds."""

from __future__ import annotations

import argparse
import json
import logging
import random
from pathlib import Path

from PIL import Image

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

TEXT_LEN_LOW_MAX = 5000
EDIT_DIST_MAX = 0.1
TEXT_LEN_HIGH_MIN = 15000
CANVAS_SIZE = (3584, 3584)
ALLOWED_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze metric JSON for text length/edit distance thresholds.")
    parser.add_argument("--metric_json", type=Path, default=Path("eval_results/pride_square/640/dpsk_eval_metric.json"), help="Path to metric JSON.")
    parser.add_argument("--image_dir", type=Path, default=Path("build_dataset_from_ebook/pride_dataset_square/images"), help="Directory containing source images.")
    parser.add_argument("--output_dir", type=Path, default=Path("build_dataset_from_ebook/pride_dataset_square_collapse_analysis/images"), help="Directory to save pasted images.")
    parser.add_argument("--json_dir", type=Path, default=Path("build_dataset_from_ebook/pride_dataset_square_collapse_analysis"), help="Directory to save ID lists.")
    parser.add_argument("--sample_size", type=int, default=150, help="Number of ids to sample per category.")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for sampling.")
    return parser.parse_args()


def load_metric_records(path: Path) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(f"Metric JSON not found: {path}")
    records = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(records, list):
        raise ValueError(f"Expected a list of records in {path}")
    return records


def collect_ids(records: list[dict]) -> tuple[list[str], list[str]]:
    low_text_low_edit: list[str] = []
    long_text: list[str] = []
    for record in records:
        if not isinstance(record, dict):
            continue
        record_id = record.get("id")
        if not record_id:
            continue
        text_len = record.get("text_len")
        edit_dist = record.get("edit_dist")
        if isinstance(text_len, (int, float)):
            if text_len <= TEXT_LEN_LOW_MAX and isinstance(edit_dist, (int, float)) and edit_dist < EDIT_DIST_MAX:
                low_text_low_edit.append(str(record_id))
            if text_len >= TEXT_LEN_HIGH_MIN:
                long_text.append(str(record_id))
    return low_text_low_edit, long_text


def sample_ids(ids: list[str], sample_size: int, rng: random.Random) -> list[str]:
    if sample_size <= 0:
        return []
    if len(ids) <= sample_size:
        return list(ids)
    return rng.sample(ids, sample_size)


def resolve_image_path(image_dir: Path, image_id: str) -> Path | None:
    for suffix in ALLOWED_SUFFIXES:
        candidate = image_dir / f"{image_id}{suffix}"
        if candidate.exists():
            return candidate
    for candidate in image_dir.glob(f"{image_id}.*"):
        if candidate.suffix.lower() in ALLOWED_SUFFIXES:
            return candidate
    return None


def paste_to_canvas(image_path: Path, output_path: Path, canvas_size: tuple[int, int]) -> bool:
    canvas_width, canvas_height = canvas_size
    with Image.open(image_path) as img:
        base_image = img.convert("RGB")
    base_width, base_height = base_image.size
    if base_width > canvas_width or base_height > canvas_height:
        logger.warning("Skip %s because %dx%d exceeds canvas %dx%d.", image_path.name, base_width, base_height, canvas_width, canvas_height)
        return False
    offset_x = (canvas_width - base_width) // 2
    offset_y = (canvas_height - base_height) // 2
    canvas = Image.new("RGB", canvas_size, "white")
    canvas.paste(base_image, (offset_x, offset_y))
    canvas.save(output_path)
    return True


def write_ids(path: Path, ids: list[str]) -> None:
    path.write_text(json.dumps(ids, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    args = parse_args()
    records = load_metric_records(args.metric_json)
    low_text_low_edit, long_text = collect_ids(records)
    logger.info("text_len <= %d and edit_dist < %.3f: %d ids", TEXT_LEN_LOW_MAX, EDIT_DIST_MAX, len(low_text_low_edit))
    logger.info("text_len >= %d: %d ids", TEXT_LEN_HIGH_MIN, len(long_text))

    rng = random.Random(args.seed)
    sampled_low = sample_ids(low_text_low_edit, args.sample_size, rng)
    sampled_long = sample_ids(long_text, args.sample_size, rng)
    logger.info("Sampled %d ids for low_text_low_edit.", len(sampled_low))
    logger.info("Sampled %d ids for long_text.", len(sampled_long))

    if not args.image_dir.exists():
        raise FileNotFoundError(f"Image directory not found: {args.image_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.json_dir.mkdir(parents=True, exist_ok=True)

    def process_ids(ids: list[str], label: str) -> list[str]:
        saved = 0
        missing = 0
        saved_ids: list[str] = []
        for image_id in ids:
            image_path = resolve_image_path(args.image_dir, image_id)
            if image_path is None:
                missing += 1
                logger.warning("Missing image for id %s in %s.", image_id, args.image_dir)
                continue
            output_path = args.output_dir / image_path.name
            if paste_to_canvas(image_path, output_path, CANVAS_SIZE):
                saved += 1
                saved_ids.append(image_id)
        logger.info("Saved %d images for %s (missing %d).", saved, label, missing)
        return saved_ids

    saved_low = process_ids(sampled_low, "low_text_low_edit")
    saved_long = process_ids(sampled_long, "long_text")
    write_ids(args.json_dir / "low_text_low_edit.json", saved_low)
    write_ids(args.json_dir / "long_text.json", saved_long)


if __name__ == "__main__":
    main()
