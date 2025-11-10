import argparse
import json
from pathlib import Path
from typing import Iterable, List


def load_records(json_path: Path) -> dict:
    with json_path.open("r", encoding="utf-8") as fp:
        data = json.load(fp)
    return {item.get("page_info", {}).get("image_path", ""): item for item in data}


def has_forbidden_category(blocks) -> bool:
    for block in blocks:
        category = block.get("category_type", "")
        lowered = category.lower()
        if "table" in lowered or "figure" in lowered:
            return True
    return False


def collect_target_images(input_path: Path) -> Iterable[Path]:
    if input_path.is_file():
        yield input_path
    elif input_path.is_dir():
        for image_path in sorted(input_path.glob("newspaper*")):
            if image_path.is_file():
                yield image_path
    else:
        raise FileNotFoundError(f"Provided path does not exist: {input_path}")

# OmniDocBench/images/newspaper_0edcd83c195557b43279fa73edaa59aa_1.jpg
def main() -> None:
    parser = argparse.ArgumentParser(description="Filter OmniDocBench images without table/figure categories.")
    parser.add_argument(
        "--path",
        nargs="?",
        default="OmniDocBench/images",
        help="Image file or directory (defaults to OmniDocBench/images)",
    )
    parser.add_argument(
        "--dataset",
        default="OmniDocBench/OmniDocBench.json",
        help="Path to OmniDocBench JSON annotations",
    )
    args = parser.parse_args()

    json_path = Path(args.dataset)
    if not json_path.exists():
        raise FileNotFoundError(f"JSON annotation file not found: {json_path}")

    input_path = Path(args.path)
    records = load_records(json_path)
    qualifying_images: List[str] = []

    for image_path in collect_target_images(input_path):
        image_name = image_path.name
        record = records.get(image_name)
        if not record:
            continue
        layout = record.get("layout_dets", [])
        if not has_forbidden_category(layout):
            qualifying_images.append(image_name)

    if not qualifying_images:
        print("No qualifying images found.")
        return

    print("Images without table/figure categories:")
    for name in qualifying_images:
        record = records.get(name, {})
        extra = record.get("extra") or {}
        relations = extra.get("relation") or []
        # print(name)
        layout = record.get("layout_dets", [])
        ordered_types = {blk["category_type"]
                        for blk in layout
                        if blk.get("order") is not None}
        print("  ordered categories:", ordered_types)     
        if relations:
            for relation in relations:
                print("  relation:", relation)
        else:
            print("  relation: []")


if __name__ == "__main__":
    main()