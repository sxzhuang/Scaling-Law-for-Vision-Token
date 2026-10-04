import argparse
import json
import logging
import shutil
from pathlib import Path


LOGGER = logging.getLogger(__name__)
TARGETS = {504: 728, 644: 840, 1036: 1260, 1288: 1652}
QWEN_FACTOR = 28


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare the 0.8R negative-control dataset from the same Stable and Instability samples.")
    parser.add_argument("--instability_root", type=Path, default=Path("aaai_resub/qwen_instability_intervention/stage1"), help="Completed Instability Stage 1 root.")
    parser.add_argument("--stable_root", type=Path, default=Path("aaai_resub/qwen_instability_intervention/stage1_stable"), help="Completed Stable Stage 1 root.")
    parser.add_argument("--source_images", type=Path, default=Path("VQAExp/data/Retrieval/28_6_0/images"), help="High-resolution mother-image directory.")
    parser.add_argument("--output_root", type=Path, default=Path("aaai_resub/qwen_instability_intervention/negative_control/stage1"), help="Negative-control Stage 1 output root.")
    return parser.parse_args()


def load_group_map(root: Path) -> dict[int, dict]:
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    return {int(group["baseline_effective_resolution"]): group for group in manifest["groups"]}


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    instability_groups = load_group_map(args.instability_root)
    stable_groups = load_group_map(args.stable_root)
    if set(instability_groups) != set(TARGETS) or set(stable_groups) != set(TARGETS):
        raise ValueError("Stable and Instability manifests must both contain the four expected baseline resolutions.")

    output_groups = []
    for baseline_resolution, target_resolution in TARGETS.items():
        if target_resolution % QWEN_FACTOR != 0:
            raise ValueError(f"Target resolution is not divisible by 28: {target_resolution}")
        phase_records = []
        phase_counts = {}
        for phase, root, group_map in (
            ("stable", args.stable_root, stable_groups),
            ("instability", args.instability_root, instability_groups),
        ):
            source_group = group_map[baseline_resolution]
            metadata_path = root / "data" / str(source_group["group"]) / "metadata.json"
            records = json.loads(metadata_path.read_text(encoding="utf-8"))
            phase_counts[phase] = len(records)
            phase_records.extend(records)

        ids = [str(record["id"]) for record in phase_records]
        if len(ids) != len(set(ids)):
            raise ValueError(f"Stable and Instability IDs overlap for R={baseline_resolution}")
        group_name = f"{baseline_resolution}_to_{target_resolution}"
        group_root = args.output_root / "data" / group_name
        raw_images_dir = group_root / "raw_images"
        raw_images_dir.mkdir(parents=True, exist_ok=True)
        for old_image in raw_images_dir.glob("*.jpg"):
            old_image.unlink()

        output_metadata = []
        for record in sorted(phase_records, key=lambda item: (int(item["text_len"]), str(item["id"]))):
            sample_id = str(record["id"])
            source_image = args.source_images / f"{sample_id}.jpg"
            if not source_image.is_file():
                raise FileNotFoundError(f"Missing mother image: {source_image}")
            shutil.copy2(source_image, raw_images_dir / source_image.name)
            output_record = dict(record)
            output_record["target_effective_resolution"] = target_resolution
            output_record["intervention"] = "negative_control_0.8R_continuous_budget_floor28"
            output_metadata.append(output_record)

        metadata_path = group_root / "metadata.json"
        metadata_path.write_text(json.dumps(output_metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        output_group = {
            "group": group_name,
            "baseline_nominal_resolution": int(instability_groups[baseline_resolution]["baseline_nominal_resolution"]),
            "baseline_effective_resolution": baseline_resolution,
            "target_effective_resolution": target_resolution,
            "baseline_vision_tokens": (baseline_resolution // QWEN_FACTOR) ** 2,
            "target_vision_tokens": (target_resolution // QWEN_FACTOR) ** 2,
            "stable_samples": phase_counts["stable"],
            "instability_samples": phase_counts["instability"],
            "selected_samples": len(output_metadata),
            "metadata": str(metadata_path),
            "raw_images": str(raw_images_dir),
        }
        output_groups.append(output_group)
        LOGGER.info("Prepared %s with %d Stable + %d Instability samples", group_name, phase_counts["stable"], phase_counts["instability"])

    manifest = {
        "experiment": "negative_control_0.8R_continuous_budget_floor28",
        "sample_set": "same Stable + Instability samples as the positive intervention",
        "source_images": str(args.source_images),
        "groups": output_groups,
        "total_samples": sum(group["selected_samples"] for group in output_groups),
    }
    args.output_root.mkdir(parents=True, exist_ok=True)
    output_manifest = args.output_root / "manifest.json"
    output_manifest.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    LOGGER.info("Saved negative-control manifest to %s", output_manifest)


if __name__ == "__main__":
    main()
