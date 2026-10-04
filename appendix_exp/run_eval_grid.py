#!/usr/bin/env python3
"""Generate prepare/eval metric JSON files for the appendix Jianai grid."""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from build_dataset_from_ebook.build_eval_json_shifted import load_metadata, md_to_text, strip_shift_suffix
from evaluation_dpsk_ocr_predictions import evaluate_predictions

LOGGER = logging.getLogger(__name__)

DEFAULT_FONT_SIZES = "20,28,36"
DEFAULT_LINE_SPACINGS = "0,6,24,42"
DEFAULT_LETTER_SPACINGS = "-1,0,7"
DEFAULT_RESOLUTIONS = "512,640,1024,1280"
DEFAULT_DATASET_ROOT = Path("appendix_exp/jianai_dataset_square")
DEFAULT_PREDICTIONS_ROOT = Path("appendix_exp/ocr_predictions/jianai_dataset_square")
DEFAULT_EVAL_ROOT = Path("appendix_exp/eval_results/jianai_dataset_square")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate prepare_json_for_eval.json and dpsk_eval_metric.json for the appendix grid.")
    parser.add_argument("--dataset_root", type=Path, default=DEFAULT_DATASET_ROOT, help="Root with <font>_<line>_<letter>/images and metadata.")
    parser.add_argument("--predictions_root", type=Path, default=DEFAULT_PREDICTIONS_ROOT, help="Root with <font>_<line>_<letter>/<resolution> OCR predictions.")
    parser.add_argument("--eval_root", type=Path, default=DEFAULT_EVAL_ROOT, help="Root where eval JSON files are written.")
    parser.add_argument("--font_sizes", type=str, default=DEFAULT_FONT_SIZES, help="Comma-separated font sizes.")
    parser.add_argument("--line_spacings", type=str, default=DEFAULT_LINE_SPACINGS, help="Comma-separated line spacings.")
    parser.add_argument("--letter_spacings", type=str, default=DEFAULT_LETTER_SPACINGS, help="Comma-separated letter spacings.")
    parser.add_argument("--resolutions", type=str, default=DEFAULT_RESOLUTIONS, help="Comma-separated resolutions.")
    parser.add_argument("--metadata_name", type=str, default="jianai_square_gt.json", help="Metadata filename under each layout folder.")
    parser.add_argument("--overwrite", action="store_true", help="Regenerate files even when complete outputs already exist.")
    parser.add_argument("--jobs", type=int, default=1, help="Number of layout/resolution groups to process in parallel.")
    return parser.parse_args()


def parse_int_list(values_arg: str, name: str) -> list[int]:
    values: list[int] = []
    for token in values_arg.split(","):
        token = token.strip()
        if not token:
            continue
        try:
            values.append(int(token))
        except ValueError as exc:
            raise ValueError(f"Invalid {name}: {token}") from exc
    if not values:
        raise ValueError(f"No valid {name} provided.")
    return values


def load_json_list(path: Path) -> list:
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return []
    return data if isinstance(data, list) else []


def is_complete_json(path: Path, expected_count: int) -> bool:
    data = load_json_list(path)
    return len(data) == expected_count


def build_prepare_json(images_dir: Path, metadata_path: Path, predictions_dir: Path, output_path: Path) -> int:
    metadata = load_metadata(metadata_path)
    if not metadata:
        raise RuntimeError(f"Metadata file contained no valid entries: {metadata_path}")

    image_paths = sorted(images_dir.glob("*.jpg"))
    if not image_paths:
        raise RuntimeError(f"No JPG images found under {images_dir}")

    records: list[dict[str, object]] = []
    missing_gt: list[str] = []
    missing_pred: list[str] = []
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

        pred_candidates = [predictions_dir / f"{image_id}.md", predictions_dir / f"{base_id}.md"]
        pred_path = None
        for candidate in pred_candidates:
            if candidate.exists():
                pred_path = candidate
                break
        if pred_path is None:
            missing_pred.append(image_id)
            continue

        records.append(
            {
                "id": image_id,
                "label": meta_entry["gt"],
                "answer": md_to_text(pred_path),
                "text_token_len": meta_entry.get("text_token_len"),
                "text_len": meta_entry.get("text_len"),
            }
        )

        if idx in checkpoints:
            LOGGER.info("Prepared %d/%d images for %s", idx, len(image_paths), output_path.parent)

    if missing_gt:
        LOGGER.warning("%s missing GT for %d images, examples=%s", output_path.parent, len(missing_gt), missing_gt[:5])
    if missing_pred:
        LOGGER.warning("%s missing predictions for %d image ids, examples=%s", output_path.parent, len(missing_pred), missing_pred[:5])
    if not records:
        raise RuntimeError(f"No eval records were generated for {output_path.parent}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
    return len(records)


def process_group(config: dict[str, object]) -> dict[str, object]:
    layout_name = str(config["layout_name"])
    resolution = int(config["resolution"])
    images_dir = Path(config["images_dir"])
    metadata_path = Path(config["metadata_path"])
    predictions_dir = Path(config["predictions_dir"])
    prepare_path = Path(config["prepare_path"])
    metric_path = Path(config["metric_path"])
    overwrite = bool(config["overwrite"])

    expected_count = len(list(images_dir.glob("*.jpg"))) if images_dir.exists() else 0
    if expected_count <= 0:
        raise RuntimeError(f"No JPG images found under {images_dir}")

    if overwrite or not is_complete_json(prepare_path, expected_count):
        prepare_count = build_prepare_json(images_dir, metadata_path, predictions_dir, prepare_path)
        prepare_status = "written"
    else:
        prepare_count = expected_count
        prepare_status = "skipped"

    if overwrite or not is_complete_json(metric_path, prepare_count):
        evaluate_predictions(str(prepare_path), str(metric_path))
        metric_status = "written"
    else:
        metric_status = "skipped"

    metric_count = len(load_json_list(metric_path))
    return {
        "layout_name": layout_name,
        "resolution": resolution,
        "prepare_status": prepare_status,
        "metric_status": metric_status,
        "prepare_count": prepare_count,
        "metric_count": metric_count,
    }


def build_configs(args: argparse.Namespace) -> list[dict[str, object]]:
    font_sizes = parse_int_list(args.font_sizes, "font_size")
    line_spacings = parse_int_list(args.line_spacings, "line_spacing")
    letter_spacings = parse_int_list(args.letter_spacings, "letter_spacing")
    resolutions = parse_int_list(args.resolutions, "resolution")
    configs: list[dict[str, object]] = []
    for font_size in font_sizes:
        for line_spacing in line_spacings:
            for letter_spacing in letter_spacings:
                layout_name = f"{font_size}_{line_spacing}_{letter_spacing}"
                layout_dir = args.dataset_root / layout_name
                for resolution in resolutions:
                    eval_dir = args.eval_root / layout_name / str(resolution)
                    configs.append(
                        {
                            "layout_name": layout_name,
                            "resolution": resolution,
                            "images_dir": layout_dir / "images",
                            "metadata_path": layout_dir / args.metadata_name,
                            "predictions_dir": args.predictions_root / layout_name / str(resolution),
                            "prepare_path": eval_dir / "prepare_json_for_eval.json",
                            "metric_path": eval_dir / "dpsk_eval_metric.json",
                            "overwrite": args.overwrite,
                        }
                    )
    return configs


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    configs = build_configs(args)
    total = len(configs)
    jobs = max(1, int(args.jobs))

    if jobs == 1:
        for idx, config in enumerate(configs, start=1):
            LOGGER.info("Processing %s @ %s (%d/%d)", config["layout_name"], config["resolution"], idx, total)
            result = process_group(config)
            LOGGER.info(
                "Done %s @ %s: prepare=%s(%d), metric=%s(%d)",
                result["layout_name"],
                result["resolution"],
                result["prepare_status"],
                result["prepare_count"],
                result["metric_status"],
                result["metric_count"],
            )
    else:
        with ProcessPoolExecutor(max_workers=jobs) as executor:
            future_to_config = {executor.submit(process_group, config): config for config in configs}
            for idx, future in enumerate(as_completed(future_to_config), start=1):
                config = future_to_config[future]
                result = future.result()
                LOGGER.info(
                    "Done %s @ %s (%d/%d): prepare=%s(%d), metric=%s(%d)",
                    result["layout_name"],
                    result["resolution"],
                    idx,
                    total,
                    result["prepare_status"],
                    result["prepare_count"],
                    result["metric_status"],
                    result["metric_count"],
                )

    LOGGER.info("Finished grid eval for %d layout/resolution groups.", total)


if __name__ == "__main__":
    main()
