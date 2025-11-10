import json
from pathlib import Path
from typing import Dict, List

from transformers import AutoTokenizer


def load_records(json_path: Path) -> List[dict]:
    with json_path.open("r", encoding="utf-8") as fp:
        return json.load(fp)


def build_image_index(records: List[dict]) -> Dict[str, List[int]]:
    index: Dict[str, List[int]] = {}
    for idx, record in enumerate(records):
        image_name = record.get("page_info", {}).get("image_path")
        if image_name:
            index.setdefault(image_name, []).append(idx)
    return index


def compute_token_length(blocks: List[dict], tokenizer: AutoTokenizer) -> int:
    total = 0
    for block in blocks:
        text = block.get("text")
        if not text:
            continue
        tokens = tokenizer(text, add_special_tokens=True, truncation=False)
        total += len(tokens.get("input_ids", []))
    return total


def main() -> None:
    images_dir = Path("craft_dataset/image")
    json_path = Path("craft_dataset/CraftData.json")

    if not images_dir.exists():
        raise FileNotFoundError(f"Images directory not found: {images_dir}")
    if not json_path.exists():
        raise FileNotFoundError(f"CraftData JSON not found: {json_path}")

    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")

    records = load_records(json_path)
    image_index = build_image_index(records)

    for image_path in sorted(images_dir.iterdir()):
        if not image_path.is_file():
            continue
        record_indices = image_index.get(image_path.name)
        if not record_indices:
            continue
        for idx in record_indices:
            record = records[idx]
            layout_dets = record.get("layout_dets", [])
            token_length = compute_token_length(layout_dets, tokenizer)
            page_info = record.setdefault("page_info", {})
            page_info["text_token_length"] = token_length

    with json_path.open("w", encoding="utf-8") as fp:
        json.dump(records, fp, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
