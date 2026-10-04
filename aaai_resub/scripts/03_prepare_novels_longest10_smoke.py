#!/usr/bin/env python3
"""Prepare a 40-image long-text novels smoke test from rendered intervention data."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
AAAI_ROOT = REPO_ROOT / "aaai_resub"
RESOLUTIONS = (512, 640, 1024, 1280)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare the ten longest novels samples at each resolution for OCR smoke testing.")
    parser.add_argument("--layout", type=str, default="16_60_-1", help="Rendered intervention-layout directory name.")
    parser.add_argument("--samples-per-resolution", type=int, default=10, help="Number of longest samples to retain per resolution.")
    parser.add_argument("--seed", type=int, default=20260721, help="Seed used only to break equal-length ties.")
    parser.add_argument("--output-root", type=Path, default=AAAI_ROOT / "smoke_tests/novels_longest10_16_60_-1", help="Destination root for selection metadata and image links.")
    return parser.parse_args()


def select_records(records: list[dict], sample_count: int, seed: int) -> list[dict]:
    if sample_count <= 0:
        raise ValueError("samples-per-resolution must be positive.")
    if len(records) < sample_count:
        raise ValueError(f"Only {len(records)} records are available, fewer than {sample_count} requested.")
    tie_breakers = {str(record["image"]): value for record, value in zip(records, random.Random(seed).sample(range(len(records)), len(records)))}
    ranked = sorted(records, key=lambda record: (-int(record["text_len"]), tie_breakers[str(record["image"])]))
    return ranked[:sample_count]


def link_image(source: Path, destination: Path) -> None:
    if destination.is_symlink():
        if destination.resolve() == source.resolve():
            return
        raise FileExistsError(f"Existing image link targets a different file: {destination}")
    if destination.exists():
        raise FileExistsError(f"Destination already exists and is not a link: {destination}")
    destination.symlink_to(source.resolve())


def main() -> None:
    args = parse_args()
    manifest: dict[str, object] = {
        "layout": args.layout,
        "samples_per_resolution": args.samples_per_resolution,
        "seed": args.seed,
        "resolutions": {},
    }
    for resolution in RESOLUTIONS:
        source_metadata = AAAI_ROOT / "selected_samples/novels" / str(resolution) / "selected_metadata.json"
        source_image_dir = AAAI_ROOT / "rendered_datasets/novels" / str(resolution) / args.layout / "images"
        records = json.loads(source_metadata.read_text(encoding="utf-8"))
        selected = select_records(records, args.samples_per_resolution, args.seed + resolution)
        selected_dir = args.output_root / "selected_samples" / str(resolution)
        image_dir = args.output_root / "images" / str(resolution)
        selected_dir.mkdir(parents=True, exist_ok=True)
        image_dir.mkdir(parents=True, exist_ok=True)
        for record in selected:
            image_name = str(record["image"])
            source_image = source_image_dir / image_name
            if not source_image.exists():
                raise FileNotFoundError(f"Rendered image is missing: {source_image}")
            link_image(source_image, image_dir / image_name)
        metadata_path = selected_dir / "selected_metadata.json"
        metadata_path.write_text(json.dumps(selected, ensure_ascii=False, indent=2), encoding="utf-8")
        manifest["resolutions"][str(resolution)] = [
            {"id": Path(str(record["image"])).stem, "text_len": int(record["text_len"])}
            for record in selected
        ]
        print(f"R={resolution}: selected {len(selected)} images; character-count range {selected[-1]['text_len']}–{selected[0]['text_len']}")
    manifest_path = args.output_root / "selection_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote smoke-test selection manifest to {manifest_path}")


if __name__ == "__main__":
    main()
