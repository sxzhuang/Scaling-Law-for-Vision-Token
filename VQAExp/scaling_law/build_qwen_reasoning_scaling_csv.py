import argparse
import csv
import json
import logging
import math
from pathlib import Path

from PIL import Image

from build_qwen_ocr_scaling_csv import CHARACTER_SIZE_BY_FONT, QWEN_FACTOR, qwen_smart_resize


LOGGER = logging.getLogger(__name__)
FONT_SIZES = [20, 28, 36]
LINE_SPACINGS = [0, 6, 24, 42]
LETTER_SPACINGS = [-1, 0, 7]
RESOLUTIONS = [512, 640, 1024, 1280]


def iter_layouts() -> list[str]:
    return [f"{font_size}_{line_spacing}_{letter_spacing}" for font_size in FONT_SIZES for line_spacing in LINE_SPACINGS for letter_spacing in LETTER_SPACINGS]


def parse_layout(layout: str) -> tuple[int, int, int]:
    parts = layout.split("_")
    if len(parts) != 3:
        raise ValueError(f"Invalid layout name: {layout}")
    return int(parts[0]), int(parts[1]), int(parts[2])


def load_json_list(json_path: Path) -> list[dict]:
    with json_path.open("r", encoding="utf-8") as file_handle:
        payload = json.load(file_handle)
    if not isinstance(payload, list):
        raise ValueError(f"Expected list JSON: {json_path}")
    return payload


def load_eval_details(eval_json: Path) -> list[dict]:
    with eval_json.open("r", encoding="utf-8") as file_handle:
        payload = json.load(file_handle)
    if isinstance(payload, dict) and isinstance(payload.get("details"), list):
        return payload["details"]
    if isinstance(payload, list):
        return payload
    raise ValueError(f"Unsupported eval JSON format: {eval_json}")


def parse_source(source) -> dict:
    if isinstance(source, str):
        return json.loads(source)
    if isinstance(source, dict):
        return source
    return {}


def infer_hop_from_source(source: dict) -> tuple[int, str]:
    question_id = str(source.get("question_id", "")).lower()
    source_text = json.dumps(source, ensure_ascii=False).lower()
    if "twohop2" in question_id or "two_hop2" in question_id or "two-hop2" in question_id:
        return 2, "twohop2"
    if "twohop" in question_id or "two_hop" in question_id or "two-hop" in question_id:
        return 2, "twohop"
    if "onehop" in question_id or "one_hop" in question_id or "one-hop" in question_id:
        return 1, "onehop"
    if "twohop2" in source_text or "two_hop2" in source_text or "two-hop2" in source_text:
        return 2, "twohop2"
    if "twohop" in source_text or "two_hop" in source_text or "two-hop" in source_text:
        return 2, "twohop"
    if "onehop" in source_text or "one_hop" in source_text or "one-hop" in source_text:
        return 1, "onehop"
    raise ValueError(f"Could not infer reasoning hop from source metadata: {source}")


def find_default_arrow_cache_dir() -> Path | None:
    cache_root = Path.home() / ".cache" / "huggingface" / "datasets" / "MLLM-CL___vtc_bench" / "default" / "0.0.0"
    if not cache_root.is_dir():
        return None
    candidates = sorted(path for path in cache_root.iterdir() if path.is_dir() and list(path.glob("vtc_bench-Reasoning-*.arrow")))
    return candidates[-1] if candidates else None


def load_reasoning_hops_from_arrow_cache(arrow_cache_dir: Path) -> dict[int, tuple[int, str]]:
    try:
        import pyarrow.ipc as ipc
    except ImportError as exc:
        raise ImportError("pyarrow is required to read cached VTCBench Arrow files directly.") from exc
    arrow_files = sorted(arrow_cache_dir.glob("vtc_bench-Reasoning-*.arrow"))
    if not arrow_files:
        raise FileNotFoundError(f"No cached Reasoning Arrow shards found in {arrow_cache_dir}")
    hops = {}
    source_index = 0
    for arrow_file in arrow_files:
        with arrow_file.open("rb") as file_handle:
            reader = ipc.open_stream(file_handle)
            source_field_index = reader.schema.get_field_index("_source")
            if source_field_index < 0:
                raise ValueError(f"Arrow shard is missing _source column: {arrow_file}")
            for batch in reader:
                source_column = batch.column(source_field_index)
                for source_value in source_column:
                    source = parse_source(source_value.as_py())
                    hops[source_index] = infer_hop_from_source(source)
                    source_index += 1
    LOGGER.info("Loaded hop metadata for %d source Reasoning rows from cached Arrow files in %s.", len(hops), arrow_cache_dir)
    return hops


def load_reasoning_hops(repo_id: str, split: str, cache_dir: Path | None, arrow_cache_dir: Path | None, disable_arrow_cache: bool) -> dict[int, tuple[int, str]]:
    resolved_arrow_cache_dir = None if disable_arrow_cache else (arrow_cache_dir if arrow_cache_dir and arrow_cache_dir.is_dir() else find_default_arrow_cache_dir())
    if resolved_arrow_cache_dir is not None:
        return load_reasoning_hops_from_arrow_cache(resolved_arrow_cache_dir)
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise ImportError("The datasets package is required to recover Reasoning hop metadata. Run this script in the environment used to load VTCBench, for example the deepseek-ocr conda environment.") from exc
    dataset = load_dataset(repo_id, split=split, cache_dir=str(cache_dir) if cache_dir else None)
    hops = {}
    for source_index, row in enumerate(dataset):
        source = parse_source(row.get("_source", {}))
        hops[source_index] = infer_hop_from_source(source)
    LOGGER.info("Loaded hop metadata for %d source Reasoning rows.", len(hops))
    return hops


def get_image_size(image_path: Path, cache: dict[Path, tuple[int, int]]) -> tuple[int, int]:
    if image_path in cache:
        return cache[image_path]
    with Image.open(image_path) as image:
        size = image.size
    cache[image_path] = size
    return size


def build_geometry(sample_id: str, text_len: int, layout: str, resolution: int, data_root: Path, factor: int, min_pixels: int | None, max_pixels: int | None, image_size_cache: dict[Path, tuple[int, int]]) -> dict:
    font_size, line_spacing, letter_spacing = parse_layout(layout)
    base_character_width, base_character_height = CHARACTER_SIZE_BY_FONT[font_size]
    raw_image_path = data_root / layout / "images" / f"{sample_id}.jpg"
    input_image_path = data_root / layout / f"images_{resolution}" / f"{sample_id}.jpg"
    if not raw_image_path.is_file():
        raise FileNotFoundError(f"Raw rendered image not found: {raw_image_path}")
    if not input_image_path.is_file():
        raise FileNotFoundError(f"Qwen input image not found: {input_image_path}")
    raw_image_w, raw_image_h = get_image_size(raw_image_path, image_size_cache)
    input_w, input_h = get_image_size(input_image_path, image_size_cache)
    qwen_h, qwen_w = qwen_smart_resize(input_h, input_w, factor, min_pixels, max_pixels)
    qwen_vision_token_num = (qwen_w // factor) * (qwen_h // factor)
    qwen_character_width = base_character_width * qwen_w / raw_image_w
    qwen_character_height = base_character_height * qwen_h / raw_image_h
    qwen_letter_spacing = letter_spacing * qwen_w / raw_image_w
    qwen_line_spacing = line_spacing * qwen_h / raw_image_h
    qwen_v_denominator = qwen_character_height + qwen_line_spacing
    qwen_v = ((qwen_character_width + qwen_letter_spacing) * float(text_len)) / qwen_v_denominator if qwen_v_denominator != 0 else float("nan")
    qwen_g = float(text_len) / float(qwen_vision_token_num)
    return {
        "image_id": sample_id,
        "layout": layout,
        "font_size": font_size,
        "line_spacing": line_spacing,
        "letter_spacing": letter_spacing,
        "resolution": resolution,
        "input_W": input_w,
        "input_H": input_h,
        "qwen_W": qwen_w,
        "qwen_H": qwen_h,
        "qwen_factor": factor,
        "qwen_vision_token_num": qwen_vision_token_num,
        "raw_image_W": raw_image_w,
        "raw_image_H": raw_image_h,
        "character_width": base_character_width,
        "character_height": base_character_height,
        "qwen_character_width": qwen_character_width,
        "qwen_character_height": qwen_character_height,
        "qwen_letter_spacing": qwen_letter_spacing,
        "qwen_line_spacing": qwen_line_spacing,
        "qwen_V": qwen_v,
        "qwen_G": qwen_g,
        "qwen_logV": math.log(max(qwen_v, 1e-12)),
        "qwen_logG": math.log(max(qwen_g, 1e-12)),
        "text_len": int(text_len),
    }


def build_rows(args: argparse.Namespace) -> list[dict]:
    gt_samples = load_json_list(args.gt_json)
    gt_by_id = {str(sample["id"]): sample for sample in gt_samples}
    hops_by_source_index = load_reasoning_hops(args.repo_id, args.split, args.cache_dir, args.arrow_cache_dir, args.disable_arrow_cache)
    rows = []
    image_size_cache: dict[Path, tuple[int, int]] = {}
    for layout in iter_layouts():
        for resolution in args.resolutions:
            eval_json = args.eval_root / layout / f"reasoning_acc_res_{resolution}.jsonl"
            if not eval_json.is_file():
                LOGGER.warning("Skip missing Reasoning eval file: %s", eval_json)
                continue
            details = load_eval_details(eval_json)
            for detail in details:
                sample_id = str(detail["id"])
                gt_sample = gt_by_id.get(sample_id)
                if gt_sample is None:
                    raise ValueError(f"Missing GT sample for id={sample_id}")
                source_index = int(gt_sample["source_index"])
                hop, hop_type = hops_by_source_index[source_index]
                contains_all = float(detail.get("contains_all", 0.0))
                correct = 1 if bool(detail.get("correct", contains_all == 1.0)) else 0
                row = build_geometry(sample_id, int(gt_sample["text_len"]), layout, int(resolution), args.data_root, args.qwen_factor, args.min_pixels, args.max_pixels, image_size_cache)
                row.update(
                    {
                        "task": "reasoning",
                        "source_index": source_index,
                        "hop": hop,
                        "hop_type": hop_type,
                        "task_complexity": hop,
                        "log_complexity": math.log(hop),
                        "contains_all": contains_all,
                        "correct": correct,
                    }
                )
                rows.append(row)
            LOGGER.info("Loaded %d Reasoning eval rows from %s", len(details), eval_json)
    return rows


def write_csv(rows: list[dict], output_csv: Path) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "image_id",
        "layout",
        "font_size",
        "line_spacing",
        "letter_spacing",
        "resolution",
        "input_W",
        "input_H",
        "qwen_W",
        "qwen_H",
        "qwen_factor",
        "qwen_vision_token_num",
        "raw_image_W",
        "raw_image_H",
        "character_width",
        "character_height",
        "qwen_character_width",
        "qwen_character_height",
        "qwen_letter_spacing",
        "qwen_line_spacing",
        "qwen_V",
        "qwen_G",
        "qwen_logV",
        "qwen_logG",
        "text_len",
        "task",
        "source_index",
        "hop",
        "hop_type",
        "task_complexity",
        "log_complexity",
        "contains_all",
        "correct",
    ]
    with output_csv.open("w", newline="", encoding="utf-8") as file_handle:
        writer = csv.DictWriter(file_handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    LOGGER.info("Saved %d rows to %s", len(rows), output_csv)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build Qwen Reasoning scaling-law CSV with Qwen image geometry and hop metadata.")
    parser.add_argument("--eval_root", type=Path, default=Path("VQAExp/eval_results/reasoning/qwen25vl7b"), help="Root directory containing Reasoning accuracy JSON files.")
    parser.add_argument("--data_root", type=Path, default=Path("VQAExp/data/Reasoning"), help="Root directory containing rendered and resized Reasoning images.")
    parser.add_argument("--gt_json", type=Path, default=Path("VQAExp/data/Reasoning/reasoning_cropped_0_25k_600.json"), help="Cropped Reasoning GT JSON with source_index.")
    parser.add_argument("--output_csv", type=Path, default=Path("VQAExp/scaling_law/qwen_reasoning_scaling_points.csv"), help="Output CSV path.")
    parser.add_argument("--repo_id", type=str, default="MLLM-CL/VTCBench", help="Hugging Face dataset repo id.")
    parser.add_argument("--split", type=str, default="Reasoning", help="Hugging Face Reasoning split name.")
    parser.add_argument("--cache_dir", type=Path, default=None, help="Optional Hugging Face dataset cache directory.")
    parser.add_argument("--arrow_cache_dir", type=Path, default=None, help="Optional local cached VTCBench Arrow directory. If unset, the default Hugging Face cache is scanned first.")
    parser.add_argument("--disable_arrow_cache", action="store_true", help="Disable direct Arrow-cache reading and use datasets.load_dataset instead.")
    parser.add_argument("--resolutions", type=int, nargs="+", default=RESOLUTIONS, help="Resolutions to include.")
    parser.add_argument("--qwen_factor", type=int, default=QWEN_FACTOR, help="Qwen image resize factor.")
    parser.add_argument("--min_pixels", type=int, default=None, help="Optional Qwen min_pixels constraint. Leave unset to only round to qwen_factor.")
    parser.add_argument("--max_pixels", type=int, default=None, help="Optional Qwen max_pixels constraint. Leave unset to only round to qwen_factor.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    rows = build_rows(args)
    if not rows:
        raise ValueError("No Reasoning scaling rows were built. Check eval_root and data_root.")
    write_csv(rows, args.output_csv)


if __name__ == "__main__":
    main()
