import json
import re
from collections import OrderedDict, defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt


def load_edit_distances(path: Path) -> OrderedDict:
    with path.open("r", encoding="utf-8") as fp:
        data = json.load(fp, object_pairs_hook=OrderedDict)
    return data


def load_craft_records(path: Path) -> List[dict]:
    with path.open("r", encoding="utf-8") as fp:
        return json.load(fp)


def build_image_index(records: List[dict]) -> Dict[str, dict]:
    mapping: Dict[str, dict] = {}
    for record in records:
        image_name = record.get("page_info", {}).get("image_path")
        if not image_name:
            continue
        mapping[image_name] = record
    return mapping


def parse_suffix(name: str) -> Tuple[str, int]:
    stem = Path(name).stem
    match = re.search(r"_(\d+)$", stem)
    if match:
        suffix = int(match.group(1))
        base = stem[: match.start()]
    else:
        suffix = -1
        base = stem
    return base, suffix


def plot_results(xs: List[int], ys: List[float], output_path: Path) -> None:
    plt.figure(figsize=(10, 6))
    plt.scatter(xs, ys, s=30, alpha=0.7)
    plt.xlabel("text_token_length")
    plt.ylabel("Edit Distance")
    plt.title("Match Edit Distance vs Text Token Length")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def main() -> None:
    edit_path = Path("OmniDocBench/result/_quick_match_text_block_per_page_edit.json")
    craft_path = Path("craft_dataset/CraftData.json")
    output_plot = Path("craft_match_edit_distance_scatter.png")
    output_json = Path("craft_match_edit_distance_scatter.json")

    edit_distances = load_edit_distances(edit_path)
    craft_records = load_craft_records(craft_path)
    image_index = build_image_index(craft_records)

    entries = []
    missing_records = []
    for key, value in edit_distances.items():
        record = image_index.get(key)
        if not record:
            missing_records.append(key)
            continue
        text_tokens = record.get("page_info", {}).get("text_token_length", 0)
        base, suffix = parse_suffix(key)
        entries.append(
            {
                "image_name": key,
                "base": base,
                "suffix": suffix,
                "text_token_length": text_tokens,
                "edit_distance": value,
            }
        )

    entries.sort(key=lambda item: (item["base"], item["suffix"]))

    xs = [entry["text_token_length"] for entry in entries]
    ys = [entry["edit_distance"] for entry in entries]
    plot_results(xs, ys, output_plot)

    with output_json.open("w", encoding="utf-8") as fp:
        json.dump(entries, fp, ensure_ascii=False, indent=2)

    # mismatched = []
    # grouped = defaultdict(list)
    # for entry in entries:
    #     grouped[entry["base"]].append(entry)

    # for base, items in grouped.items():
    #     numeric_order = [item["image_name"] for item in items]
    #     edit_order = [item["image_name"] for item in sorted(items, key=lambda x: x["edit_distance"])]
    #     if numeric_order != edit_order:
    #         mismatched.append({"base": base, "numeric_order": numeric_order, "edit_order": edit_order})

    # if mismatched:
    #     print("Groups where numeric order differs from edit-distance order:")
    #     for item in mismatched:
    #         print(json.dumps(item, ensure_ascii=False))
    # else:
    #     print("All groups consistent between numeric and edit-distance order.")

    # if missing_records:
    #     print("No CraftData entry for:", missing_records)


if __name__ == "__main__":
    main()
