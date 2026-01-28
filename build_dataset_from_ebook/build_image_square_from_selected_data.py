#!/usr/bin/env python3
"""Render selected GT entries into JPG/PDF pairs with multiple density variants."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import logging
from dataclasses import dataclass
from multiprocessing import cpu_count
from pathlib import Path

from PIL import Image
from utils.generate_square_images import (
    estimate_information_density,
    render_text_assets,
    save_image_and_pdf,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RenderGroup:
    name: str
    font_size: int
    line_spacing: int
    letter_spacing: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render selected GT text into images and PDFs with preset density groups.")
    parser.add_argument("--selected_ids", type=Path, default=Path("build_dataset_from_ebook/pride_dataset_square_density_experiment/selected_ids.json"), help="Path to JSON list of selected image ids (without extensions).")
    parser.add_argument("--metadata", type=Path, default=Path("build_dataset_from_ebook/pride_dataset_square/pride_square_gt.json"), help="Path to metadata JSON containing gt text keyed by image name.")
    parser.add_argument("--output_root", type=Path, default=Path("build_dataset_from_ebook/pride_dataset_square_density_experiment"), help="Directory where GroupA/B/C outputs are written.")
    parser.add_argument("--font_path", type=Path, default=None, help="Optional path to a TTF/OTF font file.")
    parser.add_argument("--padding", type=int, default=20, help="Pixel padding around rendered text.")
    parser.add_argument("--dpi", type=int, default=300, help="DPI for saved PDFs and JPGs.")
    parser.add_argument("--ratio_min", type=float, default=0.9, help="Minimum acceptable width/height ratio.")
    parser.add_argument("--ratio_max", type=float, default=1.1, help="Maximum acceptable width/height ratio.")
    parser.add_argument("--group_a_font_size", type=int, default=28, help="Font size for GroupA.")
    parser.add_argument("--group_a_line_spacing", type=int, default=6, help="Line spacing for GroupA.")
    parser.add_argument("--group_a_letter_spacing", type=int, default=0, help="Letter spacing for GroupA.")
    parser.add_argument("--group_b_font_size", type=int, default=28, help="Font size for GroupB.")
    parser.add_argument("--group_b_line_spacing", type=int, default=2, help="Line spacing for GroupB.")
    parser.add_argument("--group_b_letter_spacing", type=int, default=-1, help="Letter spacing for GroupB.")
    parser.add_argument("--group_c_font_size", type=int, default=28, help="Font size for GroupC.")
    parser.add_argument("--group_c_line_spacing", type=int, default=12, help="Line spacing for GroupC.")
    parser.add_argument("--group_c_letter_spacing", type=int, default=6, help="Letter spacing for GroupC.")
    parser.add_argument("--num_workers", type=int, default=6, help="Number of threads for parallel rendering.")
    return parser.parse_args()


def load_selected_ids(path: Path) -> list[str]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError("selected_ids JSON must be a list of strings.")
    ids: list[str] = []
    for item in data:
        if not isinstance(item, str):
            raise ValueError("selected_ids JSON must contain only strings.")
        normalized = Path(item.strip()).stem
        if normalized:
            ids.append(normalized)
    return ids


def load_gt_map(metadata_path: Path) -> dict[str, str]:
    records = json.loads(metadata_path.read_text(encoding="utf-8"))
    if not isinstance(records, list):
        raise ValueError("Metadata JSON must contain a list of records.")
    gt_map: dict[str, str] = {}
    for record in records:
        if not isinstance(record, dict):
            continue
        image_name = str(record.get("image", "")).strip()
        gt_text = str(record.get("gt", "")).strip()
        if not image_name or not gt_text:
            continue
        key = Path(image_name).stem
        if key in gt_map and gt_map[key] != gt_text:
            logger.warning("Duplicate entry for %s with differing gt text; keeping the first instance.", key)
            continue
        gt_map.setdefault(key, gt_text)
    return gt_map


def ensure_group_dirs(output_root: Path, configs: list[RenderGroup]) -> dict[str, dict[str, Path]]:
    dirs: dict[str, dict[str, Path]] = {}
    for config in configs:
        base = output_root / config.name
        image_dir = base / "image"
        pdf_dir = base / "pdf"
        image_dir.mkdir(parents=True, exist_ok=True)
        pdf_dir.mkdir(parents=True, exist_ok=True)
        dirs[config.name] = {"image": image_dir, "pdf": pdf_dir}
    return dirs


def load_existing_capacities(output_root: Path, configs: list[RenderGroup]) -> dict[str, dict[str, float | None]]:
    existing: dict[str, dict[str, float | None]] = {}
    for config in configs:
        record_path = output_root / config.name / "density_record.json"
        group_map: dict[str, float | None] = {}
        if record_path.exists():
            try:
                records = json.loads(record_path.read_text(encoding="utf-8"))
                if isinstance(records, list):
                    for item in records:
                        if not isinstance(item, dict):
                            continue
                        sample_id = item.get("id")
                        if not isinstance(sample_id, str):
                            continue
                        group_map[sample_id] = item.get("capacity")
            except Exception as exc:  # pragma: no cover
                logger.warning("Failed to read existing density record at %s: %s", record_path, exc)
        existing[config.name] = group_map
    return existing


def main() -> None:
    args = parse_args()
    selected_ids = load_selected_ids(args.selected_ids)
    gt_map = load_gt_map(args.metadata)
    group_configs = [
        RenderGroup(name="GroupA", font_size=args.group_a_font_size, line_spacing=args.group_a_line_spacing, letter_spacing=args.group_a_letter_spacing),
        RenderGroup(name="GroupB", font_size=args.group_b_font_size, line_spacing=args.group_b_line_spacing, letter_spacing=args.group_b_letter_spacing),
        RenderGroup(name="GroupC", font_size=args.group_c_font_size, line_spacing=args.group_c_line_spacing, letter_spacing=args.group_c_letter_spacing),
    ]
    group_dirs = ensure_group_dirs(args.output_root, group_configs)
    existing_capacities = load_existing_capacities(args.output_root, group_configs)

    if not selected_ids:
        raise RuntimeError("No ids found in selected_ids.json.")

    rendered = 0
    missing: list[str] = []
    density_records: dict[str, list[dict[str, float | str]]] = {config.name: [] for config in group_configs}
    tasks = []
    max_workers = max(1, args.num_workers)
    logger.info("Using %d worker threads.", max_workers)

    for sample_id in selected_ids:
        text = gt_map.get(sample_id)
        if text is None:
            logger.warning("Missing gt text for id %s in %s", sample_id, args.metadata)
            missing.append(sample_id)
            continue
        for config in group_configs:
            tasks.append(
                (
                    sample_id,
                    config,
                    text,
                )
            )

    def render_task(sample_id: str, text: str, config: RenderGroup) -> tuple[str, str, float | None, int, int]:
        image_path = group_dirs[config.name]["image"] / f"{sample_id}.jpg"
        pdf_path = group_dirs[config.name]["pdf"] / f"{sample_id}.pdf"
        if image_path.exists():
            with Image.open(image_path) as existing_img:
                width, height = existing_img.size
            capacity = existing_capacities.get(config.name, {}).get(sample_id)
        else:
            render_result = render_text_assets(
                text=text,
                font_size=config.font_size,
                line_spacing=config.line_spacing,
                letter_spacing=config.letter_spacing,
                font_path=args.font_path,
                pdf_dpi=args.dpi,
                padding=args.padding,
                ratio_min=args.ratio_min,
                ratio_max=args.ratio_max,
            )
            save_image_and_pdf(
                image=render_result["image"],
                pdf=render_result["pdf"],
                image_path=image_path,
                pdf_path=pdf_path,
                dpi=args.dpi,
            )
            width, height = render_result["image_shape"]
            capacity = estimate_information_density(
                image_size=render_result["image_shape"],
                character_rect=render_result["character_rect"],
                patch_size=16,
                resolution=1024,
            )
        return config.name, sample_id, capacity, int(width), int(height)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_map = {
            executor.submit(render_task, sample_id, text, config): (sample_id, config.name)
            for sample_id, config, text in tasks
        }
        for future in as_completed(future_map):
            group_name, sample_id, capacity, width, height = future.result()
            density_records[group_name].append(
                {"id": sample_id, "capacity": capacity, "width": width, "height": height}
            )
            rendered += 1

    logger.info("Rendered %d assets (%d ids x %d groups).", rendered, len(selected_ids) - len(missing), len(group_configs))
    if missing:
        logger.warning("Skipped %d ids without gt text: %s", len(missing), ", ".join(missing))

    for config in group_configs:
        records = density_records.get(config.name, [])
        records.sort(key=lambda item: item["id"])
        record_path = args.output_root / config.name / "density_record.json"
        record_path.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
        logger.info("Saved density record for %s to %s (%d entries).", config.name, record_path, len(records))


if __name__ == "__main__":
    main()
