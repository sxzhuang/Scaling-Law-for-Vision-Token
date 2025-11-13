import argparse
import json
from pathlib import Path
from typing import List

from transformers import AutoTokenizer


def load_records(json_path: Path) -> List[dict]:
    with json_path.open("r", encoding="utf-8") as fp:
        return json.load(fp)


def compute_text_token_len(text: str, tokenizer: AutoTokenizer) -> int:
    if not text:
        return 0
    tokens = tokenizer(text, add_special_tokens=True, truncation=False)
    return len(tokens.get("input_ids", []))


def add_token_lengths(records: List[dict], tokenizer: AutoTokenizer) -> None:
    for record in records:
        text = record.get("gt", "")
        record["text_token_len"] = compute_text_token_len(text, tokenizer)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute token lengths for GT text fields.")
    parser.add_argument("--json_path", type=str, default=None, 
                        help="Path to the JSON file (e.g., pride_prejudice_blocks_gt.json).")
    parser.add_argument("--tokenizer_name", type=str, default="Qwen/Qwen2.5-0.5B-Instruct",
                        help="Tokenizer identifier to use for token counting.")
    args = parser.parse_args()

    json_path = Path(args.json_path)
    if not json_path.exists():
        raise FileNotFoundError(f"JSON file not found: {json_path}")

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name)

    records = load_records(json_path)
    add_token_lengths(records, tokenizer)

    with json_path.open("w", encoding="utf-8") as fp:
        json.dump(records, fp, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
