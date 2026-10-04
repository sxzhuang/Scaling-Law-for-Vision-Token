import argparse
import json
import logging
import random
import statistics
import sys
from collections import Counter
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from build_dataset_from_ebook.build_image_square_from_json import block_code, load_blocks_from_dataset


LOGGER = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build length-binned long-context Pride samples without rendering images.")
    parser.add_argument("--source_json", type=Path, default=Path("build_dataset_from_ebook/src_book/pride_and_prejudice_dataset.json"), help="Existing sentence-aligned Pride block dataset.")
    parser.add_argument("--output_dir", type=Path, default=Path("aaai_resub/qwen_instability_intervention/pride_long_context/data"), help="Directory for GT datasets and manifest.")
    parser.add_argument("--min_length", type=int, default=25000, help="Inclusive minimum character length.")
    parser.add_argument("--shared_max_length", type=int, default=34000, help="Exclusive upper bound for samples shared by 1036 and 1288.")
    parser.add_argument("--max_length", type=int, default=48000, help="Exclusive maximum character length for 1288 samples.")
    parser.add_argument("--bin_size", type=int, default=1000, help="Character width of each bin.")
    parser.add_argument("--samples_per_bin", type=int, default=20, help="Required samples in every bin.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--max_attempts_per_bin", type=int, default=500000, help="Maximum random combinations attempted for each bin.")
    return parser.parse_args()


def build_ranges(min_length: int, max_length: int, bin_size: int) -> list[tuple[int, int]]:
    if min_length >= max_length or bin_size <= 0:
        raise ValueError("Invalid length range or bin size.")
    if (max_length - min_length) % bin_size != 0:
        raise ValueError("The requested range must be divisible by bin_size.")
    return [(start, start + bin_size) for start in range(min_length, max_length, bin_size)]


def sample_bin(blocks: list, bin_start: int, bin_end: int, count: int, rng: random.Random, used_combinations: set[tuple[int, ...]], max_attempts: int) -> list[dict]:
    block_lengths = [len(block.text) for block in blocks]
    mean_block_length = statistics.mean(block_lengths)
    target_center = (bin_start + bin_end - 1) / 2
    central_group_size = max(1, round(target_center / mean_block_length))
    candidate_group_sizes = list(range(max(1, central_group_size - 4), min(len(blocks), central_group_size + 4) + 1))
    records = []
    attempts = 0
    while len(records) < count and attempts < max_attempts:
        attempts += 1
        group_size = rng.choice(candidate_group_sizes)
        combination = tuple(sorted(rng.sample(range(len(blocks)), group_size)))
        if combination in used_combinations:
            continue
        combined_length = sum(block_lengths[index] for index in combination) + group_size - 1
        if not bin_start <= combined_length < bin_end:
            continue
        combined_text = " ".join(blocks[index].text for index in combination)
        if len(combined_text) != combined_length:
            raise ValueError("Combined text length calculation mismatch.")
        used_combinations.add(combination)
        records.append({
            "gt": combined_text,
            "context": combined_text,
            "text_len": combined_length,
            "bin_start": bin_start,
            "bin_end": bin_end,
            "group_label": f"{group_size}_blocks",
            "blocks": [block_code(blocks[index]) for index in combination],
        })
    if len(records) != count:
        raise RuntimeError(f"Only found {len(records)}/{count} samples for [{bin_start}, {bin_end}) after {attempts} attempts.")
    LOGGER.info("Built bin [%d, %d): %d samples in %d attempts", bin_start, bin_end, len(records), attempts)
    return records


def assign_ids(records: list[dict]) -> None:
    for index, record in enumerate(records):
        sample_id = f"pride_long_{record['bin_start']:05d}_{index:04d}"
        record["id"] = sample_id
        record["image"] = f"{sample_id}.jpg"
        record["pdf"] = f"{sample_id}.pdf"


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def split_exact(records: list[dict], shard_count: int) -> list[list[dict]]:
    if shard_count <= 0 or len(records) % shard_count != 0:
        raise ValueError(f"Cannot split {len(records)} records evenly into {shard_count} shards.")
    shard_size = len(records) // shard_count
    return [records[index * shard_size:(index + 1) * shard_size] for index in range(shard_count)]


def validate(records: list[dict], ranges: list[tuple[int, int]], samples_per_bin: int) -> dict:
    ids = [str(record["id"]) for record in records]
    texts = [str(record["gt"]) for record in records]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate sample IDs.")
    if len(texts) != len(set(texts)):
        raise ValueError("Duplicate GT texts.")
    for record in records:
        if record["gt"] != record["context"] or len(record["gt"]) != int(record["text_len"]):
            raise ValueError(f"GT/context/length mismatch for {record['id']}")
        if not int(record["bin_start"]) <= int(record["text_len"]) < int(record["bin_end"]):
            raise ValueError(f"Sample outside its bin: {record['id']}")
    counts = Counter((int(record["bin_start"]), int(record["bin_end"])) for record in records)
    expected = {item: samples_per_bin for item in ranges}
    if dict(counts) != expected:
        raise ValueError(f"Unexpected bin counts: {dict(counts)}")
    return {
        "sample_count": len(records),
        "unique_ids": len(set(ids)),
        "unique_gt_texts": len(set(texts)),
        "min_text_len": min(int(record["text_len"]) for record in records),
        "max_text_len": max(int(record["text_len"]) for record in records),
        "bin_counts": {f"{start}-{end}": counts[(start, end)] for start, end in ranges},
    }


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    ranges = build_ranges(args.min_length, args.max_length, args.bin_size)
    if args.shared_max_length not in {start for start, _ in ranges} and args.shared_max_length != args.max_length:
        raise ValueError("shared_max_length must fall on a bin boundary.")
    blocks = load_blocks_from_dataset(args.source_json)
    if not blocks:
        raise ValueError(f"No Pride blocks found in {args.source_json}")
    rng = random.Random(args.seed)
    used_combinations = set()
    records = []
    for bin_start, bin_end in ranges:
        records.extend(sample_bin(blocks, bin_start, bin_end, args.samples_per_bin, rng, used_combinations, args.max_attempts_per_bin))
    records.sort(key=lambda record: (int(record["bin_start"]), int(record["text_len"]), tuple(record["blocks"])))
    assign_ids(records)

    shared_records = [record for record in records if int(record["text_len"]) < args.shared_max_length]
    expected_shared = ((args.shared_max_length - args.min_length) // args.bin_size) * args.samples_per_bin
    if len(shared_records) != expected_shared:
        raise ValueError(f"Expected {expected_shared} shared records, found {len(shared_records)}")

    all_path = args.output_dir / "pride_25k_48k_all_gt.json"
    resolution_1036_path = args.output_dir / "pride_25k_34k_for_1036_gt.json"
    resolution_1288_path = args.output_dir / "pride_25k_48k_for_1288_gt.json"
    write_json(all_path, records)
    write_json(resolution_1036_path, shared_records)
    write_json(resolution_1288_path, records)

    extra_records = [record for record in records if int(record["text_len"]) >= args.shared_max_length]
    shard_dir = args.output_dir / "shards"
    shared_shards = split_exact(shared_records, 3)
    extra_shards = split_exact(extra_records, 5)
    shared_shard_paths = []
    extra_shard_paths = []
    for index, shard in enumerate(shared_shards):
        shard_path = shard_dir / f"pride_25k_34k_part_{index}.json"
        write_json(shard_path, shard)
        shared_shard_paths.append(str(shard_path))
    for index, shard in enumerate(extra_shards):
        shard_path = shard_dir / f"pride_34k_48k_part_{index}.json"
        write_json(shard_path, shard)
        extra_shard_paths.append(str(shard_path))

    all_validation = validate(records, ranges, args.samples_per_bin)
    shared_ranges = [item for item in ranges if item[1] <= args.shared_max_length]
    shared_validation = validate(shared_records, shared_ranges, args.samples_per_bin)
    manifest = {
        "source_json": str(args.source_json),
        "source_blocks": len(blocks),
        "seed": args.seed,
        "bin_size": args.bin_size,
        "samples_per_bin": args.samples_per_bin,
        "shared_range": [args.min_length, args.shared_max_length],
        "full_range": [args.min_length, args.max_length],
        "outputs": {
            "all_unique_samples": str(all_path),
            "resolution_1036": str(resolution_1036_path),
            "resolution_1288": str(resolution_1288_path),
            "shared_render_shards": shared_shard_paths,
            "extra_1288_render_shards": extra_shard_paths,
        },
        "validation": {"all": all_validation, "resolution_1036": shared_validation, "resolution_1288": all_validation},
    }
    write_json(args.output_dir / "manifest.json", manifest)
    LOGGER.info("Saved %d unique Pride samples: %d for 1036 and %d for 1288", len(records), len(shared_records), len(records))


if __name__ == "__main__":
    main()
