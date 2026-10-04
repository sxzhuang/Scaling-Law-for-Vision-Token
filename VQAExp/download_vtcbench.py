import argparse
import json
import logging
from pathlib import Path

from datasets import load_dataset


LOGGER = logging.getLogger(__name__)
DEFAULT_REPO_ID = "MLLM-CL/VTCBench"
DEFAULT_SPLIT = "Retrieval"


def save_jsonl(dataset, output_file: Path) -> None:
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with output_file.open("w", encoding="utf-8") as f:
        for row in dataset:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download the Retrieval split of VTCBench.")
    parser.add_argument("--repo_id", type=str, default=DEFAULT_REPO_ID, help="Hugging Face dataset repo id.")
    parser.add_argument("--split", type=str, default=DEFAULT_SPLIT, help="Dataset split to download.")
    parser.add_argument("--output_file", type=Path, default=Path("VQAExp/data/vtcbench_retrieval.jsonl"), help="Path to save the downloaded split as JSONL.")
    parser.add_argument("--cache_dir", type=Path, default=None, help="Optional Hugging Face cache directory.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    LOGGER.info("Downloading %s split from %s", args.split, args.repo_id)
    dataset = load_dataset(args.repo_id, split=args.split, cache_dir=str(args.cache_dir) if args.cache_dir else None)
    LOGGER.info("Loaded %d examples", len(dataset))

    save_jsonl(dataset, args.output_file)
    LOGGER.info("Saved split to %s", args.output_file)


if __name__ == "__main__":
    main()
