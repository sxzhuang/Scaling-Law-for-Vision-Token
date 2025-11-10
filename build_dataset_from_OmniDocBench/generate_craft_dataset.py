import argparse
import copy
import json
import math
from pathlib import Path
from typing import Dict, Iterable, List, Set, Tuple

from PIL import Image, ImageDraw


def load_records(dataset_path: Path) -> Dict[str, dict]:
    with dataset_path.open("r", encoding="utf-8") as fp:
        records = json.load(fp)
    mapping: Dict[str, dict] = {}
    for record in records:
        image_name = record.get("page_info", {}).get("image_path")
        if image_name:
            mapping[image_name] = record
    return mapping


def collect_truncated_pairs(record: dict) -> List[Tuple[int, int]]:
    extra = record.get("extra") or {}
    relations = extra.get("relation") or []
    pairs: List[Tuple[int, int]] = []
    seen: Set[Tuple[int, int]] = set()
    for relation in relations:
        if relation.get("relation_type") != "truncated":
            continue
        src = relation.get("source_anno_id")
        tgt = relation.get("target_anno_id")
        if src is None or tgt is None or src == tgt:
            continue
        key = tuple(sorted((src, tgt)))
        if key in seen:
            continue
        seen.add(key)
        pairs.append((src, tgt))
    return pairs


def merge_truncated_blocks(record: dict) -> None:
    layout = record.get("layout_dets", [])
    if not layout:
        return

    id_to_index: Dict[int, int] = {}
    for idx, block in enumerate(layout):
        anno_id = block.get("anno_id")
        if anno_id is not None:
            id_to_index[anno_id] = idx

    pairs = collect_truncated_pairs(record)
    if not pairs:
        return

    representative_map: Dict[int, dict] = {}
    skip_ids: Set[int] = set()

    for src, tgt in pairs:
        relevant_ids = [aid for aid in (src, tgt) if aid in id_to_index]
        if len(relevant_ids) < 2:
            continue

        relevant_ids.sort(key=lambda aid: id_to_index.get(aid, math.inf))
        if relevant_ids[0] in skip_ids:
            continue
        if relevant_ids[1] in skip_ids:
            continue

        blocks = [layout[id_to_index[aid]] for aid in relevant_ids]
        representative_id = relevant_ids[0]

        merged_block = build_merged_block(blocks)
        representative_map[representative_id] = merged_block
        skip_ids.add(relevant_ids[1])

    if not representative_map:
        return

    new_layout: List[dict] = []
    for block in layout:
        anno_id = block.get("anno_id")
        if anno_id in representative_map:
            new_layout.append(representative_map[anno_id])
        elif anno_id in skip_ids:
            continue
        else:
            new_layout.append(block)

    record["layout_dets"] = new_layout


def build_merged_block(blocks: List[dict]) -> dict:
    if not blocks:
        raise ValueError("Cannot merge empty block list")

    blocks_sorted = sorted(blocks, key=lambda blk: blk.get("order", math.inf))
    base_block = copy.deepcopy(blocks_sorted[0])

    all_polys = [blk.get("poly") for blk in blocks_sorted if blk.get("poly")]
    xs: List[float] = []
    ys: List[float] = []
    for poly in all_polys:
        xs.extend(poly[::2])
        ys.extend(poly[1::2])
    if xs and ys:
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)
        base_block["poly"] = [min_x, min_y, max_x, min_y, max_x, max_y, min_x, max_y]

    text_values = [blk.get("text") for blk in blocks_sorted if blk.get("text")]
    if text_values:
        base_block["text"] = "\n".join(text_values)

    combined_spans: List[dict] = []
    for blk in blocks_sorted:
        for span in blk.get("line_with_spans", []) or []:
            combined_spans.append(copy.deepcopy(span))
    if combined_spans:
        base_block["line_with_spans"] = combined_spans
    elif "line_with_spans" in base_block:
        del base_block["line_with_spans"]

    base_block["order"] = min((blk.get("order", math.inf) for blk in blocks_sorted), default=base_block.get("order"))

    return base_block


def polygon_points(poly: List[float]) -> List[Tuple[float, float]]:
    return list(zip(poly[::2], poly[1::2]))


def translate_poly(poly: List[float], offset_x: float, offset_y: float) -> List[float]:
    adjusted: List[float] = []
    for idx, value in enumerate(poly):
        if idx % 2 == 0:
            adjusted.append(value - offset_x)
        else:
            adjusted.append(value - offset_y)
    return adjusted


def add_block_to_image(base_image: Image.Image, source: Image.Image, poly: List[float]) -> None:
    mask = Image.new("L", base_image.size, 0)
    draw = ImageDraw.Draw(mask)
    draw.polygon(polygon_points(poly), fill=255)
    base_image.paste(source, mask=mask)


def accumulate_bounds(bounds: Tuple[float, float, float, float], poly: List[float]) -> Tuple[float, float, float, float]:
    xs = poly[::2]
    ys = poly[1::2]
    min_x, min_y, max_x, max_y = bounds
    return (
        min(min_x, min(xs)),
        min(min_y, min(ys)),
        max(max_x, max(xs)),
        max(max_y, max(ys)),
    )


def ensure_output_dirs(image_dir: Path, json_path: Path) -> None:
    image_dir.mkdir(parents=True, exist_ok=True)
    json_path.parent.mkdir(parents=True, exist_ok=True)


def record_contains_table(blocks: Iterable[dict]) -> bool:
    for block in blocks:
        category = (block.get("category_type") or "").lower()
        if "table" in category or "figure" in category:
            return True
    return False


def gather_target_images(images_dir: Path, records: Dict[str, dict], prefix: str) -> List[Path]:
    targets: List[Path] = []
    for image_path in sorted(images_dir.glob(f"{prefix}*")):
        if not image_path.is_file():
            continue
        record = records.get(image_path.name)
        if not record:
            continue
        if record_contains_table(record.get("layout_dets", [])):
            continue
        targets.append(image_path)
    return targets


def generate_craft_records_for_image(
    image_path: Path,
    record: dict,
    output_image_dir: Path,
) -> List[dict]:
    working_record = copy.deepcopy(record)
    merge_truncated_blocks(working_record)

    original_image = Image.open(image_path).convert("RGB")
    width, height = original_image.size

    allowed_categories = {"title", "text_block"}
    non_table_blocks: List[dict] = []
    for block in working_record.get("layout_dets", []):
        category = (block.get("category_type") or "").lower()
        if category not in allowed_categories:
            continue
        if block.get("ignore", False):
            continue
        non_table_blocks.append(block)

    if not non_table_blocks:
        return []

    non_table_blocks.sort(key=lambda blk: blk.get("order") if blk.get("order") is not None else math.inf)

    cumulative_image = Image.new("RGB", (width, height), color=(255, 255, 255))
    cumulative_bounds = (width, height, 0.0, 0.0)
    accumulated_blocks: List[dict] = []
    craft_records: List[dict] = []
    base_name = image_path.stem

    for index, block in enumerate(non_table_blocks, start=1):
        poly = block.get("poly")
        if not poly:
            continue

        add_block_to_image(cumulative_image, original_image, poly)
        cumulative_bounds = accumulate_bounds(cumulative_bounds, poly)
        accumulated_blocks.append(block)

        min_x, min_y, max_x, max_y = cumulative_bounds
        min_x = max(0, math.floor(min_x))
        min_y = max(0, math.floor(min_y))
        max_x = min(width, math.ceil(max_x))
        max_y = min(height, math.ceil(max_y))
        if max_x <= min_x or max_y <= min_y:
            continue

        cropped = cumulative_image.crop((min_x, min_y, max_x, max_y))
        output_name = f"{base_name}_combined_image_{index}.jpg"
        cropped.save(output_image_dir / output_name)

        adjusted_blocks = []
        for blk in accumulated_blocks:
            blk_copy = copy.deepcopy(blk)
            if blk_copy.get("poly"):
                blk_copy["poly"] = translate_poly(blk_copy["poly"], min_x, min_y)
            if "line_with_spans" in blk_copy:
                for span in blk_copy["line_with_spans"]:
                    if span.get("poly"):
                        span["poly"] = translate_poly(span["poly"], min_x, min_y)
            adjusted_blocks.append(blk_copy)

        page_attribute = copy.deepcopy(record.get("page_info", {}).get("page_attribute", {}))
        craft_records.append(
            {
                "layout_dets": adjusted_blocks,
                "extra": {"relation": []},
                "page_info": {
                    "page_no": record.get("page_info", {}).get("page_no"),
                    "height": cropped.height,
                    "width": cropped.width,
                    "image_path": output_name,
                    "page_attribute": page_attribute,
                },
            }
        )

    return craft_records


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate combined images for filtered OmniDocBench pages.")
    parser.add_argument("--dataset", default="OmniDocBench/OmniDocBench.json", help="Path to OmniDocBench JSON file")
    parser.add_argument("--images-dir", default="OmniDocBench/images", help="Directory containing source images")
    parser.add_argument("--output-image-dir", default="craft_dataset/image", help="Directory to store generated images")
    parser.add_argument("--output-json", default="craft_dataset/CraftData.json", help="Output JSON path")
    parser.add_argument("--prefix", default="newspaper", help="Filename prefix for selecting images")
    args = parser.parse_args()

    dataset_path = Path(args.dataset)
    images_dir = Path(args.images_dir)
    output_image_dir = Path(args.output_image_dir)
    output_json_path = Path(args.output_json)

    ensure_output_dirs(output_image_dir, output_json_path)

    records = load_records(dataset_path)
    target_images = gather_target_images(images_dir, records, args.prefix)

    if not target_images:
        raise ValueError("No qualifying images found with the provided prefix and filters.")

    existing_records: List[dict] = []
    if output_json_path.exists():
        with output_json_path.open("r", encoding="utf-8") as fp:
            try:
                existing_records = json.load(fp)
            except json.JSONDecodeError:
                existing_records = []

    all_records: List[dict] = existing_records

    for image_path in target_images:
        record = records.get(image_path.name)
        if not record:
            continue
        new_records = generate_craft_records_for_image(image_path, record, output_image_dir)
        all_records.extend(new_records)

    with output_json_path.open("w", encoding="utf-8") as fp:
        json.dump(all_records, fp, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
