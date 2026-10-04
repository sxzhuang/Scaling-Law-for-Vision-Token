import argparse
import csv
import json
import logging
import math
import random
import shutil
from pathlib import Path


LOGGER = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Select stratified Qwen 1288-instability samples for a uniform 2380 resize test.")
    parser.add_argument("--scaling_csv", type=Path, default=Path("VQAExp/scaling_law/qwen_ocr_scaling_points.csv"), help="Qwen OCR scaling-point CSV.")
    parser.add_argument("--fit_summary", type=Path, default=Path("VQAExp/scaling_law/Fitting/ocr/1280/em_fit_summary.json"), help="Qwen 1288 effective-resolution fit summary.")
    parser.add_argument("--gt_json", type=Path, default=Path("VQAExp/data/Retrieval/retrieval_cropped_0_25k_600.json"), help="Ground-truth metadata JSON.")
    parser.add_argument("--data_root", type=Path, default=Path("VQAExp/data/Retrieval"), help="Root containing each layout's rendered images.")
    parser.add_argument("--output_root", type=Path, default=Path("aaai_resub/qwen_1288_to_2380_test/data"), help="Test data output root.")
    parser.add_argument("--samples_per_stratum", type=int, default=10, help="Number sampled from each of three equal-width Z strata.")
    parser.add_argument("--seed", type=int, default=20260722, help="Deterministic sampling seed.")
    parser.add_argument("--stable_ed", type=float, default=0.1, help="Expected edit-distance threshold defining Stable.")
    parser.add_argument("--collapse_ed", type=float, default=0.7, help="Expected edit-distance threshold defining Collapse.")
    return parser.parse_args()


def boundary_for_expected_ed(edit_distance: float, low_mean: float, high_mean: float) -> float:
    probability = (edit_distance - low_mean) / (high_mean - low_mean)
    if not 0.0 < probability < 1.0:
        raise ValueError(f"Expected-ED threshold {edit_distance} is outside mixture means ({low_mean}, {high_mean}).")
    return math.log(probability / (1.0 - probability))


def load_gt_by_id(path: Path) -> dict[str, dict]:
    with path.open("r", encoding="utf-8") as file_handle:
        records = json.load(file_handle)
    return {str(record["id"]): record for record in records}


def read_instability_rows(args: argparse.Namespace, stable_boundary: float, collapse_boundary: float, gating: dict) -> list[dict]:
    selected = []
    with args.scaling_csv.open("r", newline="", encoding="utf-8") as file_handle:
        for row in csv.DictReader(file_handle):
            if int(row["resolution"]) != 1280:
                continue
            z_value = float(gating["w0"]) + float(gating["a"]) * float(row["qwen_logV"]) + float(gating["alpha"]) * float(row["qwen_logG"])
            if stable_boundary <= z_value < collapse_boundary:
                row["z_1288"] = z_value
                row["baseline_edit_distance"] = float(row["edit_distance"])
                selected.append(row)
    return selected


def stratified_sample(rows: list[dict], stable_boundary: float, collapse_boundary: float, samples_per_stratum: int, seed: int) -> list[dict]:
    rng = random.Random(seed)
    width = (collapse_boundary - stable_boundary) / 3.0
    sampled = []
    for stratum_index in range(3):
        lower = stable_boundary + stratum_index * width
        upper = collapse_boundary if stratum_index == 2 else lower + width
        candidates = [row for row in rows if lower <= float(row["z_1288"]) < upper]
        if len(candidates) < samples_per_stratum:
            raise ValueError(f"Z stratum {stratum_index} has only {len(candidates)} candidates; requested {samples_per_stratum}.")
        chosen = rng.sample(candidates, samples_per_stratum)
        for row in chosen:
            row["z_stratum"] = stratum_index
        sampled.extend(chosen)
    return sorted(sampled, key=lambda row: (int(row["z_stratum"]), float(row["z_1288"])))


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    with args.fit_summary.open("r", encoding="utf-8") as file_handle:
        fit_summary = json.load(file_handle)
    gating = fit_summary["gating_params"]
    low_mean = float(fit_summary["beta_params"]["low_ed"]["mean"])
    high_mean = float(fit_summary["beta_params"]["high_ed"]["mean"])
    stable_boundary = boundary_for_expected_ed(args.stable_ed, low_mean, high_mean)
    collapse_boundary = boundary_for_expected_ed(args.collapse_ed, low_mean, high_mean)
    instability_rows = read_instability_rows(args, stable_boundary, collapse_boundary, gating)
    sampled_rows = stratified_sample(instability_rows, stable_boundary, collapse_boundary, args.samples_per_stratum, args.seed)
    gt_by_id = load_gt_by_id(args.gt_json)

    raw_images_dir = args.output_root / "raw_images"
    raw_images_dir.mkdir(parents=True, exist_ok=True)
    for old_image in raw_images_dir.glob("*.jpg"):
        old_image.unlink()

    metadata = []
    manifest = []
    for row in sampled_rows:
        original_id = str(row["image_id"])
        layout = str(row["layout"])
        test_id = f"{layout}__{original_id}"
        source_image = args.data_root / layout / "images" / f"{original_id}.jpg"
        if not source_image.is_file():
            raise FileNotFoundError(f"Rendered source image not found: {source_image}")
        shutil.copy2(source_image, raw_images_dir / f"{test_id}.jpg")
        gt_record = dict(gt_by_id[original_id])
        gt_record["id"] = test_id
        gt_record["original_id"] = original_id
        gt_record["layout"] = layout
        metadata.append(gt_record)
        manifest.append({
            "id": test_id,
            "original_id": original_id,
            "layout": layout,
            "z_1288": float(row["z_1288"]),
            "z_stratum": int(row["z_stratum"]),
            "baseline_edit_distance": float(row["baseline_edit_distance"]),
            "baseline_actual_stable": float(row["baseline_edit_distance"]) < args.stable_ed,
            "source_image": str(source_image),
        })

    args.output_root.mkdir(parents=True, exist_ok=True)
    (args.output_root / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    (args.output_root / "manifest.json").write_text(json.dumps({
        "seed": args.seed,
        "stable_boundary": stable_boundary,
        "collapse_boundary": collapse_boundary,
        "stable_ed": args.stable_ed,
        "collapse_ed": args.collapse_ed,
        "num_instability_candidates": len(instability_rows),
        "num_selected": len(manifest),
        "samples": manifest,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    LOGGER.info("Selected %d of %d 1288-instability rows into %s", len(manifest), len(instability_rows), args.output_root)


if __name__ == "__main__":
    main()
