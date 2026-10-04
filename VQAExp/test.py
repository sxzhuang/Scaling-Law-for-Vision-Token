import argparse
from pathlib import Path

from PIL import Image
from datasets import load_dataset


DEFAULT_REPO_ID = "MLLM-CL/VTCBench"
DEFAULT_SPLIT = "Reasoning"


def summarize_value(value):
    if isinstance(value, Image.Image):
        return {
            "type": type(value).__name__,
            "size": value.size,
            "mode": value.mode,
        }
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, list):
        return {
            "type": "list",
            "length": len(value),
            "preview": value[:3],
        }
    if isinstance(value, dict):
        return {
            "type": "dict",
            "keys": list(value.keys()),
        }
    return {
        "type": type(value).__name__,
        "repr": repr(value),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect one example from a VTCBench split.")
    parser.add_argument("--repo_id", type=str, default=DEFAULT_REPO_ID, help="Hugging Face dataset repo id.")
    parser.add_argument("--split", type=str, default=DEFAULT_SPLIT, help="Dataset split to load.")
    parser.add_argument("--index", type=int, default=0, help="Example index to inspect.")
    parser.add_argument("--cache_dir", type=Path, default=None, help="Optional Hugging Face cache directory.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset = load_dataset(args.repo_id, split=args.split, cache_dir=str(args.cache_dir) if args.cache_dir else None)
    example = dataset[args.index]

    print(f"Dataset: {args.repo_id}")
    print(f"Split: {args.split}")
    print(f"Number of examples: {len(dataset)}")
    print(f"Example index: {args.index}")
    print(f"Fields: {list(example.keys())}")
    print()

    for key, value in example.items():
        print(f"[{key}]")
        print(summarize_value(value))
        print()


if __name__ == "__main__":
    main()
