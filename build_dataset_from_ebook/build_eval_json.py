#!/usr/bin/env python3
"""Assemble evaluation JSON for focus_benchmark_test/eval_tools/eval_ocr_test.py.

The script scans rendered Pride & Prejudice block images, aligns each image
with its ground-truth text from ``pride_blocks_experiment/pride_blocks_gt.json``
and the corresponding prediction stored as a Markdown file under
``pride_blocks_experiment_pred/640``. The resulting list of
``{"id", "label", "answer"}`` dictionaries can be fed directly into
``eval_ocr_test.py``.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import re
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build OCR eval JSON input.")
    parser.add_argument("--images_dir", type=Path, default=Path("focus_benchmark_test/demo_test"), help="Directory containing JPG images to evaluate.")
    parser.add_argument("--metadata_path", type=Path, default=Path("build_dataset_from_ebook/pride_dataset/pride_prejudice_blocks_gt.json"), help="JSON file with image->ground truth mappings.")
    parser.add_argument("--predictions_dir", type=Path, default=Path("focus_benchmark_test/demo_pred"), help="Directory containing prediction .md files.")
    parser.add_argument("--output_path", type=Path, default=Path("tmp_eval.json"), help="Destination JSON for eval_ocr_test.py.")
    return parser.parse_args()


def load_metadata(path: Path) -> dict[str, dict]:
    if not path.exists():
        fallback = path.parent / "pride_prejudice_blocks_gt.json"
        if fallback.exists():
            path = fallback
        else:
            raise FileNotFoundError(f"Metadata not found at {path}")
    records = json.loads(path.read_text(encoding="utf-8"))
    mapping = {}
    for record in records:
        image_name = record.get("image")
        text = record.get("gt")
        token_len = record.get("text_token_len")
        if not image_name or text is None:
            continue
        mapping[str(image_name)] = {
            "gt": str(text),
            "text_token_len": token_len,
        }
    return mapping


def md_to_text(md_path: Path) -> str:
    raw = md_path.read_text(encoding="utf-8")
    # Remove image/link syntax.
    text = re.sub(r"!\[[^\]]*\]\([^)]+\)", " ", raw)
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)
    lines: list[str] = []
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

    records: list[dict] = []
    missing_gt = []
    missing_pred = []

    image_paths = sorted(args.images_dir.glob("*.jpg"))
    total = len(image_paths)
    if total == 0:
        raise RuntimeError("No JPG images found under the specified images_dir.")
    checkpoints = sorted({min(total, math.ceil(total * i / 4)) for i in range(1, 5)})

    for idx, image_path in enumerate(image_paths, start=1):
        image_name = image_path.name
        image_id = image_path.stem

        meta_entry = metadata.get(image_name)
        if meta_entry is None:
            missing_gt.append(image_name)
            continue
        gt_text = meta_entry.get("gt")
        token_len = meta_entry.get("text_token_len")

        pred_path = args.predictions_dir / f"{image_id}.md"
        if not pred_path.exists():
            missing_pred.append(pred_path)
            continue

        pred_text = md_to_text(pred_path)
        records.append(
            {
                "id": image_id,
                "label": gt_text,
                "answer": pred_text,
                "text_token_len": token_len,
            }
        )

        if idx in checkpoints:
            percent = (idx / total) * 100
            logger.info("Processed %d/%d images (%.1f%%)", idx, total, percent)

    if missing_gt:
        logger.warning("Missing GT for %d images: %s ...", len(missing_gt), missing_gt[:5])
    if missing_pred:
        logger.warning(
            "Missing predictions for %d images: %s ...",
            len(missing_pred),
            [p.name for p in missing_pred[:5]],
        )
    if not records:
        raise RuntimeError("No eval records were generated. Check inputs.")

    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    args.output_path.write_text(
        json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    logger.info("Wrote %d eval entries to %s", len(records), args.output_path)


if __name__ == "__main__":
    main()
