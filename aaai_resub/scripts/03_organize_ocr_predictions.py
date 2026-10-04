#!/usr/bin/env python3
"""Move OCR files created with a missing output-path separator into directories."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
AAAI_ROOT = REPO_ROOT / "aaai_resub"
DOMAINS = ("novels", "laws", "economics", "medicine", "newspapers", "letters")
RESOLUTIONS = (512, 640, 1024, 1280)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Organize flat OCR predictions into per-resolution directories.")
    parser.add_argument("--execute", action="store_true", help="Perform the validated file moves. Without this flag, only validate and report the plan.")
    parser.add_argument("--manifest", type=Path, default=AAAI_ROOT / "analysis/ocr_prediction_migration.json", help="Path for the migration manifest.")
    return parser.parse_args()


def expected_prediction_names(domain: str, resolution: int) -> set[str]:
    metadata_path = AAAI_ROOT / "selected_samples" / domain / str(resolution) / "selected_metadata.json"
    records = json.loads(metadata_path.read_text(encoding="utf-8"))
    if not isinstance(records, list):
        raise ValueError(f"Expected a metadata list: {metadata_path}")
    names: set[str] = set()
    for record in records:
        image_name = str(record.get("image", "")).strip()
        if not image_name:
            raise ValueError(f"Missing image name in {metadata_path}")
        stem = Path(image_name).stem
        names.add(f"{stem}.md")
        names.add(f"{stem}_det.md")
    if len(names) != 2 * len(records):
        raise ValueError(f"Duplicate image IDs in {metadata_path}")
    return names


def plan_moves(domain: str, resolution: int) -> tuple[list[tuple[Path, Path]], set[str], set[str]]:
    domain_root = AAAI_ROOT / "ocr_predictions" / domain
    target_dir = domain_root / str(resolution)
    prefix = str(resolution)
    source_paths = sorted(
        path for path in domain_root.glob(f"{prefix}*.md")
        if path.is_file() and not path.name.startswith(f"{prefix}/")
    )
    existing_target_names = {path.name for path in target_dir.glob("*.md")} if target_dir.exists() else set()
    source_target_names: set[str] = set()
    moves: list[tuple[Path, Path]] = []
    for source_path in source_paths:
        target_name = source_path.name.removeprefix(prefix)
        if not target_name:
            raise ValueError(f"Cannot remove resolution prefix from {source_path}")
        if target_name in source_target_names:
            raise ValueError(f"Duplicate flat prediction target name: {target_name}")
        source_target_names.add(target_name)
        target_path = target_dir / target_name
        if target_name in existing_target_names:
            raise FileExistsError(f"Prediction appears in both flat and target locations: {target_path}")
        moves.append((source_path, target_path))
    return moves, existing_target_names, source_target_names


def validate_group(domain: str, resolution: int, moves: list[tuple[Path, Path]], existing_target_names: set[str], source_target_names: set[str]) -> dict:
    expected_names = expected_prediction_names(domain, resolution)
    combined_names = existing_target_names | source_target_names
    if combined_names != expected_names:
        missing = sorted(expected_names - combined_names)
        unexpected = sorted(combined_names - expected_names)
        raise RuntimeError(
            f"{domain}, R={resolution}: incomplete or unexpected predictions "
            f"(missing={len(missing)}, unexpected={len(unexpected)})."
        )
    expected = len(expected_names) // 2
    plain_count = sum(not name.endswith("_det.md") for name in combined_names)
    det_count = sum(name.endswith("_det.md") for name in combined_names)
    return {
        "domain": domain,
        "resolution": resolution,
        "expected_samples": expected,
        "plain_predictions": plain_count,
        "det_predictions": det_count,
        "already_organized_files": len(existing_target_names),
        "files_to_move": len(moves),
        "target_directory": str((AAAI_ROOT / "ocr_predictions" / domain / str(resolution)).relative_to(REPO_ROOT)),
    }


def main() -> None:
    args = parse_args()
    all_moves: list[tuple[Path, Path]] = []
    groups: list[dict] = []
    for domain in DOMAINS:
        for resolution in RESOLUTIONS:
            moves, existing_target_names, source_target_names = plan_moves(domain, resolution)
            groups.append(validate_group(domain, resolution, moves, existing_target_names, source_target_names))
            all_moves.extend(moves)

    if args.execute:
        for _, target_path in all_moves:
            target_path.parent.mkdir(parents=True, exist_ok=True)
        for source_path, target_path in all_moves:
            source_path.rename(target_path)

    manifest = {
        "executed": args.execute,
        "groups": groups,
        "total_files": sum(group["plain_predictions"] + group["det_predictions"] for group in groups),
        "files_moved_this_run": len(all_moves),
        "total_samples": sum(group["expected_samples"] for group in groups),
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    action = "Moved" if args.execute else "Validated"
    print(
        f"{action} {len(all_moves)} OCR Markdown files; "
        f"validated {manifest['total_files']} files for {manifest['total_samples']} samples."
    )


if __name__ == "__main__":
    main()
