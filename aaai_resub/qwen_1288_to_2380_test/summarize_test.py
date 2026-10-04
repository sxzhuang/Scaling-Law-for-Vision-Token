import argparse
import json
import logging
from pathlib import Path


LOGGER = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize the Qwen 1288-instability to 2380 OCR test.")
    parser.add_argument("--manifest", type=Path, default=Path("aaai_resub/qwen_1288_to_2380_test/data/manifest.json"), help="Selection manifest.")
    parser.add_argument("--evaluation_jsonl", type=Path, default=Path("aaai_resub/qwen_1288_to_2380_test/evaluation/uniform/ocr_edit_distance_res_2380.jsonl"), help="2380 edit-distance JSONL.")
    parser.add_argument("--output", type=Path, default=Path("aaai_resub/qwen_1288_to_2380_test/evaluation/summary.json"), help="Summary JSON path.")
    parser.add_argument("--stable_ed", type=float, default=0.1, help="Observed edit-distance threshold defining Stable.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    manifest_payload = json.loads(args.manifest.read_text(encoding="utf-8"))
    baseline_by_id = {sample["id"]: sample for sample in manifest_payload["samples"]}
    evaluated_by_id = {}
    with args.evaluation_jsonl.open("r", encoding="utf-8") as file_handle:
        for line in file_handle:
            if line.strip():
                record = json.loads(line)
                evaluated_by_id[str(record["id"])] = record

    paired = []
    for sample_id, baseline in baseline_by_id.items():
        if sample_id not in evaluated_by_id:
            continue
        target_ed = float(evaluated_by_id[sample_id]["edit_dist"])
        baseline_ed = float(baseline["baseline_edit_distance"])
        paired.append({**baseline, "target_edit_distance": target_ed, "target_actual_stable": target_ed < args.stable_ed, "improved": target_ed < baseline_ed})

    baseline_nonstable = [sample for sample in paired if not sample["baseline_actual_stable"]]
    converted = [sample for sample in baseline_nonstable if sample["target_actual_stable"]]
    summary = {
        "num_selected": len(baseline_by_id),
        "num_evaluated": len(paired),
        "stable_ed": args.stable_ed,
        "baseline_mean_edit_distance": sum(sample["baseline_edit_distance"] for sample in paired) / len(paired),
        "target_mean_edit_distance": sum(sample["target_edit_distance"] for sample in paired) / len(paired),
        "baseline_actual_stable_rate": sum(sample["baseline_actual_stable"] for sample in paired) / len(paired),
        "target_actual_stable_rate": sum(sample["target_actual_stable"] for sample in paired) / len(paired),
        "improvement_rate": sum(sample["improved"] for sample in paired) / len(paired),
        "num_baseline_actual_nonstable": len(baseline_nonstable),
        "nonstable_to_stable_conversion_rate": None if not baseline_nonstable else len(converted) / len(baseline_nonstable),
        "samples": paired,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    LOGGER.info("Evaluated %d/%d samples; mean ED %.4f -> %.4f; target stable %.2f%%; nonstable conversion %.2f%%", len(paired), len(baseline_by_id), summary["baseline_mean_edit_distance"], summary["target_mean_edit_distance"], 100.0 * summary["target_actual_stable_rate"], 100.0 * (summary["nonstable_to_stable_conversion_rate"] or 0.0))


if __name__ == "__main__":
    main()
