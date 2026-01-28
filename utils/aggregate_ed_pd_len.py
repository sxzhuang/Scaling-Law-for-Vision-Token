#!/usr/bin/env python3
"""Estimate per-patch character density for a dataset."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

from PIL import Image

from utils.generate_square_images import load_font


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Estimate per-patch character density for a dataset.")
    parser.add_argument("--image_dir", type=Path, default=Path("build_dataset_from_ebook/pride_dataset_square/images"), help="Directory containing rendered images.")
    parser.add_argument("--metadata", type=Path, default=Path("build_dataset_from_ebook/pride_dataset_square/pride_square_gt.json"), help="Metadata JSON with gt and text_token_len.")
    parser.add_argument("--eval_json", type=Path, default=Path("eval_results/pride_square/1024/dpsk_eval_metric.json"), help="Evaluation JSON containing edit_dist per id.")
    parser.add_argument("--data_id", type=str, default="baseline", help="Dataset identifier for output naming.")
    parser.add_argument("--font_path", type=Path, default=None, help="Optional font path.")
    parser.add_argument("--font_size", type=int, default=28, help="Font size used for rendering.")
    parser.add_argument("--line_spacing", type=int, default=6, help="Line spacing used for rendering.")
    parser.add_argument("--letter_spacing", type=int, default=0, help="Letter spacing used for rendering.")
    parser.add_argument("--resolution", type=int, default=1024, help="Target resolution for resizing.")
    parser.add_argument("--patch_size", type=int, default=16, help="Patch size for density calculation.")
    parser.add_argument("--sample_size", type=int, default=5, help="Number of samples to estimate character size.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for sampling.")
    parser.add_argument("--output_dir", type=Path, default=Path("eval_results/density_len_editdist_results"), help="Directory to save the output JSON.")
    return parser.parse_args()


def load_metadata(metadata_path: Path) -> list[dict[str, Any]]:
    data: Any = json.loads(metadata_path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"{metadata_path} must be a list of records.")
    return [r for r in data if isinstance(r, dict)]


def find_record_by_image(stem: str, records: list[dict[str, Any]]) -> dict[str, Any] | None:
    for record in records:
        image_name = str(record.get("image", "")).strip()
        if not image_name:
            continue
        if Path(image_name).stem == stem:
            return record
    return None


def estimate_char_size(text: str, font_path: Path | None, font_size: int) -> tuple[float, float]:
    font = load_font(font_size, font_path)
    char_count = len(text)
    total_width = font.getlength(text) if char_count else 0.0
    avg_width = total_width / char_count if char_count else 0.0
    bbox = font.getbbox("Hg")
    avg_height = float(bbox[3] - bbox[1])
    return avg_width, avg_height


def sample_average_char_size(
    images: list[Path],
    records: list[dict[str, Any]],
    font_path: Path | None,
    font_size: int,
    line_spacing: int,
    letter_spacing: int,
    sample_size: int,
) -> tuple[float, float]:
    if not images:
        raise ValueError("No images found under image_dir.")
    chosen = images if len(images) <= sample_size else random.sample(images, sample_size)
    widths: list[float] = []
    heights: list[float] = []
    for img_path in chosen:
        record = find_record_by_image(img_path.stem, records)
        if record is None:
            continue
        gt_text = str(record.get("gt", "")).strip()
        if not gt_text:
            continue
        base_w, base_h = estimate_char_size(gt_text, font_path, font_size)
        widths.append(base_w + letter_spacing)
        heights.append(base_h + line_spacing)
    if not widths or not heights:
        raise ValueError("Failed to estimate character size from samples; check metadata or samples.")
    return sum(widths) / len(widths), sum(heights) / len(heights)


def compute_density(
    image_size: tuple[int, int],
    char_size: tuple[float, float],
    resolution: int,
    patch_size: int,
) -> float:
    width, height = image_size
    if width <= 0 or height <= 0 or resolution <= 0 or patch_size <= 0:
        return 0.0
    char_w, char_h = char_size
    scale_w = resolution / float(width)
    scale_h = resolution / float(height)
    resized_char_w = char_w * scale_w
    resized_char_h = char_h * scale_h
    if resized_char_w <= 0 or resized_char_h <= 0:
        return 0.0
    patch_area = float(patch_size * patch_size)
    char_area = resized_char_w * resized_char_h
    return patch_area / char_area


def load_eval_edit_dist(eval_path: Path) -> dict[str, float]:
    if not eval_path.exists():
        return {}
    data: Any = json.loads(eval_path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        return {}
    result: dict[str, float] = {}
    for item in data:
        if not isinstance(item, dict):
            continue
        sample_id = str(item.get("id", "")).strip()
        if not sample_id:
            continue
        try:
            edit_dist = float(item.get("edit_dist"))
        except (TypeError, ValueError):
            continue
        result[sample_id] = edit_dist
    return result


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    records = load_metadata(args.metadata)
    eval_edit_dist = load_eval_edit_dist(args.eval_json)
    image_dir = args.image_dir
    images = sorted([p for p in image_dir.glob("*.jpg") if p.is_file()])
    avg_char_w, avg_char_h = sample_average_char_size(
        images=images,
        records=records,
        font_path=args.font_path,
        font_size=args.font_size,
        line_spacing=args.line_spacing,
        letter_spacing=args.letter_spacing,
        sample_size=args.sample_size,
    )
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    data_id = args.data_id
    if data_id is None or str(data_id).strip().lower() in {"", "none"}:
        output_path = output_dir / "ed_pd_len.json"
    else:
        output_path = output_dir / f"{args.resolution}_{args.font_size}_{args.line_spacing}_{args.letter_spacing}_{data_id}.json"
    results: list[dict[str, Any]] = []
    for img_path in images:
        record = find_record_by_image(img_path.stem, records)
        if record is None:
            continue
        token_len = record.get("text_token_len", None)
        try:
            token_len = int(token_len) if token_len is not None else None
        except (TypeError, ValueError):
            token_len = None
        text_len = record.get("text_len", None)
        try:
            text_len = int(text_len) if text_len is not None else None
        except (TypeError, ValueError):
            text_len = None
        with Image.open(img_path) as im:
            img_w, img_h = im.size
        density = compute_density(
            image_size=(img_w, img_h),
            char_size=(avg_char_w, avg_char_h),
            resolution=args.resolution,
            patch_size=args.patch_size,
        )
        edit_dist = eval_edit_dist.get(img_path.stem)
        results.append(
            {
                "id": img_path.stem,
                "patch_density": density,
                "text_token_len": token_len,
                "text_len": text_len,
                "edit_dist": edit_dist,
            }
        )
    output_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Estimated character area (with spacing): {avg_char_w:.4f}px × {avg_char_h:.4f}px")
    print(f"Processed {len(results)} images, output saved to: {output_path}")


if __name__ == "__main__":
    main()
