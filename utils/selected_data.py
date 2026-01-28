#!/usr/bin/env python3
"""Utilities for selecting and exporting dataset slices based on eval metrics."""

from __future__ import annotations

import argparse
import json
import logging
import shutil
from pathlib import Path
from typing import Iterable, List

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
LOGGER = logging.getLogger(__name__)


def select_ids(
    json_path: Path | str,
    edit_dist_threshold: float,
    min_token_len: int,
    max_token_len: int,
) -> List[str]:
    """Return IDs whose token length and edit distance exceed the thresholds."""

    if min_token_len > max_token_len:
        raise ValueError("min_token_len must be less than or equal to max_token_len.")

    path = Path(json_path)
    with path.open("r", encoding="utf-8") as file:
        records = json.load(file)

    if not isinstance(records, list):
        raise ValueError(f"{path} must contain a list of metric entries.")

    selected = []
    for entry in records:
        try:
            token_len = int(entry["text_token_len"])
            edit_dist = float(entry["edit_dist"])
            sample_id = str(entry["id"])
        except KeyError as exc:
            raise KeyError(f"Missing required field in entry: {exc}") from exc

        if min_token_len <= token_len <= max_token_len and edit_dist >= edit_dist_threshold:
            selected.append(sample_id)

    LOGGER.info(
        "Selected %d ids from %s with edit_dist >= %.4f and token_len in [%d, %d]",
        len(selected),
        path,
        edit_dist_threshold,
        min_token_len,
        max_token_len,
    )
    return selected


def copy_selected_images(
    image_folder: Path | str,
    ids: Iterable[str],
    output_folder: Path | str,
    ids_json_path: Path | str | None = None,
) -> list[Path]:
    """Copy images whose basename matches the provided ids to *output_folder*."""

    image_dir = Path(image_folder)
    destination = Path(output_folder)
    destination.mkdir(parents=True, exist_ok=True)

    ids_list = list(dict.fromkeys(ids))
    if not ids_list:
        LOGGER.warning("No ids provided; nothing to copy.")
        return []

    copied_paths: list[Path] = []
    for sample_id in ids_list:
        source_path = _find_image(image_dir, sample_id)
        target_path = destination / source_path.name
        shutil.copy2(source_path, target_path)
        copied_paths.append(target_path)
        LOGGER.debug("Copied %s -> %s", source_path, target_path)

    ids_output = Path(ids_json_path) if ids_json_path else destination.parent / "selected_ids.json"
    with ids_output.open("w", encoding="utf-8") as file:
        json.dump(ids_list, file, indent=2, ensure_ascii=False)
    LOGGER.info("Saved %d ids to %s", len(ids_list), ids_output)

    return copied_paths


def _find_image(image_dir: Path, sample_id: str) -> Path:
    """Locate the image file matching *sample_id* (without extension)."""

    candidates = [
        image_dir / f"{sample_id}{suffix}"
        for suffix in (".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG")
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Could not find image for id '{sample_id}' under {image_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Select IDs from metric JSON and optionally copy images.")
    parser.add_argument("--metrics_json", type=Path, default=Path("eval_results/pride_square/1024/dpsk_eval_metric.json"), help="Path to dpsk_eval_metric.json file.")
    parser.add_argument("--edit_dist_threshold", type=float, default=0.19, help="Minimum edit distance to keep.")
    parser.add_argument("--min_token_len", type=int, default=1500, help="Minimum text_token_len to keep.")
    parser.add_argument("--max_token_len", type=int, default=4050, help="Maximum text_token_len to keep.")
    parser.add_argument("--image_folder", type=Path, default=Path("build_dataset_from_ebook/pride_dataset_square/images"), help="Directory containing source images.")
    parser.add_argument("--output_folder", type=Path, default=Path("build_dataset_from_ebook/pride_dataset_square_large_variance/selected_from1024/images"), help="Directory to store copied images.")
    parser.add_argument("--ids_output", type=Path, default=None, help="Optional explicit path for saved id list.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ids = select_ids(
        json_path=args.metrics_json,
        edit_dist_threshold=args.edit_dist_threshold,
        min_token_len=args.min_token_len,
        max_token_len=args.max_token_len,
    )
    if args.image_folder and args.output_folder:
        copy_selected_images(
            image_folder=args.image_folder,
            ids=ids,
            output_folder=args.output_folder,
            ids_json_path=args.ids_output,
        )
    else:
        ids_output = args.ids_output or Path("selected_ids.json")
        with ids_output.open("w", encoding="utf-8") as file:
            json.dump(ids, file, indent=2, ensure_ascii=False)
        LOGGER.info("Saved %d ids to %s (no images copied)", len(ids), ids_output)


if __name__ == "__main__":
    main()


__all__ = ["select_ids", "copy_selected_images"]
