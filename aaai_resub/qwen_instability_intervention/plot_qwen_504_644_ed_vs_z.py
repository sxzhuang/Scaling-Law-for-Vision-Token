import argparse
import csv
import json
import logging
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image


LOGGER = logging.getLogger(__name__)
LAYOUT = "28_6_0"
FONT_SIZE = 28
LINE_SPACING = 6
LETTER_SPACING = 0
CHARACTER_WIDTH = 14.0533
CHARACTER_HEIGHT = 26.0
QWEN_FACTOR = 28
RESOLUTIONS = {
    512: {"effective": 504, "color": "#1F77B4"},
    640: {"effective": 644, "color": "#D95F02"},
    1024: {"effective": 1036, "color": "#2CA02C"},
    1280: {"effective": 1288, "color": "#9467BD"},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot Qwen OCR edit distance against fitted Z for all four effective resolutions.")
    parser.add_argument("--scaling_csv", type=Path, default=Path("VQAExp/scaling_law/qwen_ocr_scaling_points.csv"), help="Qwen OCR scaling-point CSV.")
    parser.add_argument("--fit_root", type=Path, default=Path("VQAExp/scaling_law/Fitting/ocr"), help="Root containing per-resolution fitted summaries.")
    parser.add_argument("--additional_eval_dir", type=Path, default=Path("aaai_resub/qwen_instability_intervention/pride_long_context/evaluation"), help="Edit-distance JSONL directory for added long-context points.")
    parser.add_argument("--additional_raw_image_dir", type=Path, default=Path("aaai_resub/qwen_instability_intervention/pride_long_context/rendered/mother_images/28_6_0/images"), help="Raw rendered images used to derive V for added points.")
    parser.add_argument("--output", type=Path, default=Path("aaai_resub/qwen_instability_intervention/figures/qwen_ocr_four_resolutions_edit_distance_vs_z.jpg"), help="Output JPG path.")
    parser.add_argument("--points_csv", type=Path, default=Path("aaai_resub/qwen_instability_intervention/figures/qwen_ocr_four_resolutions_edit_distance_vs_z.csv"), help="Output CSV containing transformed points.")
    parser.add_argument("--summary", type=Path, default=Path("aaai_resub/qwen_instability_intervention/figures/qwen_ocr_four_resolutions_edit_distance_vs_z.json"), help="Output transformation summary.")
    parser.add_argument("--dpi", type=int, default=300, help="Output resolution in dots per inch.")
    return parser.parse_args()


def load_fit_params(fit_root: Path) -> dict[int, dict]:
    parameters = {}
    shared_a = None
    shared_alpha = None
    for nominal_resolution in RESOLUTIONS:
        summary_path = fit_root / str(nominal_resolution) / "em_fit_summary.json"
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        gating = summary["gating_params"]
        a = float(gating["a"])
        alpha = float(gating["alpha"])
        if shared_a is None:
            shared_a = a
            shared_alpha = alpha
        elif not math.isclose(a, shared_a, rel_tol=0.0, abs_tol=1e-12) or not math.isclose(alpha, shared_alpha, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError("The fitted a and alpha parameters are not shared across resolutions.")
        parameters[nominal_resolution] = {"w0": float(gating["w0"]), "a": a, "alpha": alpha, "fit_summary": str(summary_path)}
    return parameters


def load_points(scaling_csv: Path, parameters: dict[int, dict]) -> dict[int, list[dict]]:
    points = {resolution: [] for resolution in RESOLUTIONS}
    with scaling_csv.open("r", newline="", encoding="utf-8") as file_handle:
        for row in csv.DictReader(file_handle):
            nominal_resolution = int(row["resolution"])
            if row["layout"] != LAYOUT or nominal_resolution not in points:
                continue
            params = parameters[nominal_resolution]
            log_v = float(row["qwen_logV"])
            log_g = float(row["qwen_logG"])
            z_value = params["w0"] + params["a"] * log_v + params["alpha"] * log_g
            points[nominal_resolution].append({
                "image_id": str(row["image_id"]),
                "nominal_resolution": nominal_resolution,
                "effective_resolution": RESOLUTIONS[nominal_resolution]["effective"],
                "text_len": int(float(row["text_len"])),
                "edit_distance": float(row["edit_distance"]),
                "qwen_logV": log_v,
                "qwen_logG": log_g,
                "z": z_value,
                "source": "original",
            })
    for nominal_resolution, records in points.items():
        if not records:
            raise ValueError(f"No points found for layout={LAYOUT}, resolution={nominal_resolution}")
        ids = [record["image_id"] for record in records]
        if len(ids) != len(set(ids)):
            raise ValueError(f"Duplicate image IDs for resolution={nominal_resolution}")
    reference_ids = {record["image_id"] for record in points[min(points)]}
    for nominal_resolution, records in points.items():
        if {record["image_id"] for record in records} != reference_ids:
            raise ValueError(f"Sample IDs differ for resolution={nominal_resolution}")
    return points


def append_additional_points(points: dict[int, list[dict]], parameters: dict[int, dict], eval_dir: Path, raw_image_dir: Path) -> None:
    for nominal_resolution in (1024, 1280):
        eval_path = eval_dir / f"ocr_edit_distance_res_{nominal_resolution}.jsonl"
        if not eval_path.is_file():
            raise FileNotFoundError(f"Additional edit-distance file not found: {eval_path}")
        effective_resolution = int(RESOLUTIONS[nominal_resolution]["effective"])
        vision_tokens = (effective_resolution // QWEN_FACTOR) ** 2
        params = parameters[nominal_resolution]
        existing_ids = {record["image_id"] for record in points[nominal_resolution]}
        added = 0
        with eval_path.open("r", encoding="utf-8") as file_handle:
            for line_number, line in enumerate(file_handle, start=1):
                if not line.strip():
                    continue
                record = json.loads(line)
                image_id = str(record["id"])
                if image_id in existing_ids:
                    raise ValueError(f"Duplicate additional ID for resolution={nominal_resolution}: {image_id}")
                raw_image_path = raw_image_dir / f"{image_id}.jpg"
                if not raw_image_path.is_file():
                    raise FileNotFoundError(f"Raw image not found for {eval_path}:{line_number}: {raw_image_path}")
                with Image.open(raw_image_path) as image:
                    raw_width, raw_height = image.size
                text_len = int(record["text_len"])
                qwen_character_width = CHARACTER_WIDTH * effective_resolution / raw_width
                qwen_character_height = CHARACTER_HEIGHT * effective_resolution / raw_height
                qwen_letter_spacing = LETTER_SPACING * effective_resolution / raw_width
                qwen_line_spacing = LINE_SPACING * effective_resolution / raw_height
                qwen_v = ((qwen_character_width + qwen_letter_spacing) * text_len) / (qwen_character_height + qwen_line_spacing)
                qwen_g = text_len / vision_tokens
                log_v = math.log(max(qwen_v, 1e-12))
                log_g = math.log(max(qwen_g, 1e-12))
                z_value = params["w0"] + params["a"] * log_v + params["alpha"] * log_g
                points[nominal_resolution].append({
                    "image_id": image_id,
                    "nominal_resolution": nominal_resolution,
                    "effective_resolution": effective_resolution,
                    "text_len": text_len,
                    "edit_distance": float(record["edit_dist"]),
                    "qwen_logV": log_v,
                    "qwen_logG": log_g,
                    "z": z_value,
                    "source": "pride_long_context",
                })
                existing_ids.add(image_id)
                added += 1
        LOGGER.info("Added %d long-context points for nominal resolution %d", added, nominal_resolution)


def save_points(points: dict[int, list[dict]], path: Path) -> None:
    fieldnames = ["image_id", "nominal_resolution", "effective_resolution", "text_len", "edit_distance", "qwen_logV", "qwen_logG", "z", "source"]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file_handle:
        writer = csv.DictWriter(file_handle, fieldnames=fieldnames)
        writer.writeheader()
        for nominal_resolution in RESOLUTIONS:
            writer.writerows(points[nominal_resolution])


def plot(points: dict[int, list[dict]], output: Path, dpi: int) -> None:
    plt.rcParams.update({"font.size": 10, "axes.titlesize": 11, "axes.labelsize": 11, "legend.fontsize": 9})
    fig, axis = plt.subplots(figsize=(6.4, 4.0), dpi=dpi)
    for nominal_resolution, style in RESOLUTIONS.items():
        records = points[nominal_resolution]
        axis.scatter(
            [record["z"] for record in records],
            [record["edit_distance"] for record in records],
            s=12,
            alpha=0.48,
            edgecolors="none",
            color=style["color"],
            label=f"R={style['effective']} (n={len(records)})",
            rasterized=True,
        )
    axis.set_xlabel(r"$Z = w_{0,r} + a\log V + \alpha\log G$")
    axis.set_ylabel("Normalized edit distance")
    axis.set_ylim(-0.03, 1.03)
    axis.set_title(f"Qwen2.5-VL OCR, layout {LAYOUT}")
    axis.grid(True, linestyle=":", linewidth=0.6, alpha=0.65)
    axis.legend(loc="upper left", frameon=False)
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, format="jpg", dpi=dpi)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    parameters = load_fit_params(args.fit_root)
    points = load_points(args.scaling_csv, parameters)
    append_additional_points(points, parameters, args.additional_eval_dir, args.additional_raw_image_dir)
    save_points(points, args.points_csv)
    plot(points, args.output, args.dpi)
    summary = {
        "layout": LAYOUT,
        "z_formula": "w0_resolution + a * qwen_logV + alpha * qwen_logG",
        "shared_parameters": {"a": parameters[512]["a"], "alpha": parameters[512]["alpha"]},
        "resolutions": {},
    }
    for nominal_resolution, records in points.items():
        z_values = [record["z"] for record in records]
        summary["resolutions"][str(nominal_resolution)] = {
            "effective_resolution": RESOLUTIONS[nominal_resolution]["effective"],
            "n_samples": len(records),
            "w0": parameters[nominal_resolution]["w0"],
            "z_min": min(z_values),
            "z_max": max(z_values),
            "mean_edit_distance": sum(record["edit_distance"] for record in records) / len(records),
        }
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    LOGGER.info("Saved %d transformed points and plot to %s", sum(len(records) for records in points.values()), args.output)


if __name__ == "__main__":
    main()
