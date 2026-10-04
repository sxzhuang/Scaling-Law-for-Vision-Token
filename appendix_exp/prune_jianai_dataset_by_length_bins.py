import argparse
import json
import logging
import random
from collections import Counter, defaultdict
from pathlib import Path


LOGGER = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Randomly remove sampled Jianai records from every text_len bin and sync rendered image folders.")
    parser.add_argument("--dataset_json", type=Path, default=Path("appendix_exp/text_samples/jianai_sampled_text_dataset.json"), help="Path to sampled text dataset JSON.")
    parser.add_argument("--summary_json", type=Path, default=Path("appendix_exp/text_samples/jianai_sampled_text_summary.json"), help="Path to sampled text summary JSON.")
    parser.add_argument("--image_root", type=Path, default=Path("appendix_exp/jianai_dataset_square"), help="Root containing rendered layout folders.")
    parser.add_argument("--bin_size", type=int, default=1000, help="Length bin width.")
    parser.add_argument("--remove_per_bin", type=int, default=20, help="Number of records to remove from each bin.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducible pruning.")
    parser.add_argument("--histogram_png", type=Path, default=Path("appendix_exp/text_samples/jianai_sampled_text_length_histogram.png"), help="Path to updated text length histogram.")
    parser.add_argument("--apply", action="store_true", help="Actually write files and delete images. Without this flag the script only performs a dry run.")
    parser.add_argument("--dry_run", action="store_true", help="Print what would be removed without writing files.")
    return parser.parse_args()


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, data) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def text_len_bin(record: dict, bin_size: int) -> int:
    return (int(record["text_len"]) // bin_size) * bin_size


def choose_records_to_remove(records: list[dict], bin_size: int, remove_per_bin: int, seed: int) -> tuple[set[str], dict[str, int], dict[str, int]]:
    rng = random.Random(seed)
    records_by_bin = defaultdict(list)
    for record in records:
        records_by_bin[text_len_bin(record, bin_size)].append(record)

    remove_ids = set()
    counts_before = {}
    removed_by_bin = {}
    for bin_start in sorted(records_by_bin):
        bin_records = records_by_bin[bin_start]
        bin_label = f"{bin_start}-{bin_start + bin_size}"
        counts_before[bin_label] = len(bin_records)
        remove_count = min(remove_per_bin, len(bin_records))
        removed_records = rng.sample(bin_records, remove_count)
        removed_by_bin[bin_label] = remove_count
        remove_ids.update(record["sample_id"] for record in removed_records)

    return remove_ids, counts_before, removed_by_bin


def summarize(records: list[dict], base_summary: dict, bin_size: int, removed_by_bin: dict[str, int], remove_per_bin: int, seed: int) -> dict:
    text_lens = [int(record["text_len"]) for record in records]
    counts_by_group = Counter(record.get("group_label", "unknown") for record in records)
    counts_by_bin = Counter(text_len_bin(record, bin_size) for record in records)

    summary = dict(base_summary)
    summary["total_samples"] = len(records)
    summary["counts_by_group"] = dict(sorted(counts_by_group.items()))
    summary["text_len"] = {
        "min": min(text_lens) if text_lens else None,
        "max": max(text_lens) if text_lens else None,
        "mean": sum(text_lens) / len(text_lens) if text_lens else None,
    }
    summary["text_len_bins"] = {
        f"{bin_start}-{bin_start + bin_size}": counts_by_bin[bin_start]
        for bin_start in sorted(counts_by_bin)
    }
    summary["length_bin_pruning"] = {
        "bin_size": bin_size,
        "remove_per_bin": remove_per_bin,
        "seed": seed,
        "removed_by_bin": removed_by_bin,
    }
    return summary


def prune_rendered_outputs(image_root: Path, remove_ids: set[str], dry_run: bool) -> tuple[int, int, int]:
    metadata_files = sorted(image_root.glob("*/jianai_square_gt.json"))
    metadata_removed = 0
    images_removed = 0
    folders_touched = 0

    for metadata_path in metadata_files:
        records = load_json(metadata_path)
        kept_records = []
        removed_records = []
        for record in records:
            image_name = record.get("image", "")
            sample_id = Path(image_name).stem
            if sample_id in remove_ids:
                removed_records.append(record)
            else:
                kept_records.append(record)

        if not removed_records:
            continue

        folders_touched += 1
        metadata_removed += len(removed_records)

        image_dir = metadata_path.parent / "images"
        for record in removed_records:
            image_path = image_dir / record["image"]
            if image_path.exists():
                images_removed += 1
                if not dry_run:
                    image_path.unlink()

        if not dry_run:
            write_json(metadata_path, kept_records)

    return folders_touched, metadata_removed, images_removed


def write_histogram(records: list[dict], histogram_png: Path, bin_size: int) -> None:
    import matplotlib.pyplot as plt

    text_lens = [int(record["text_len"]) for record in records]
    if not text_lens:
        LOGGER.warning("No records left; skip histogram.")
        return

    max_len = max(text_lens)
    bins = list(range(0, ((max_len // bin_size) + 2) * bin_size, bin_size))
    histogram_png.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(10, 6))
    plt.hist(text_lens, bins=bins, edgecolor="black")
    plt.xlabel("text_len")
    plt.ylabel("sample count")
    plt.title("Jianai sampled text length distribution")
    plt.xticks(bins, rotation=45)
    plt.tight_layout()
    plt.savefig(histogram_png, dpi=200)
    plt.close()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = parse_args()

    records = load_json(args.dataset_json)
    summary = load_json(args.summary_json) if args.summary_json.exists() else {}
    dry_run = args.dry_run or not args.apply

    remove_ids, counts_before, removed_by_bin = choose_records_to_remove(records, args.bin_size, args.remove_per_bin, args.seed)
    kept_records = [record for record in records if record["sample_id"] not in remove_ids]
    new_summary = summarize(kept_records, summary, args.bin_size, removed_by_bin, args.remove_per_bin, args.seed)

    LOGGER.info("Counts before pruning: %s", counts_before)
    LOGGER.info("Removed by bin: %s", removed_by_bin)
    LOGGER.info("Text dataset records: %d -> %d", len(records), len(kept_records))

    folders_touched, metadata_removed, images_removed = prune_rendered_outputs(args.image_root, remove_ids, dry_run)
    LOGGER.info("Rendered folders touched: %d", folders_touched)
    LOGGER.info("Metadata records removed across rendered folders: %d", metadata_removed)
    LOGGER.info("JPG files removed across rendered folders: %d", images_removed)

    if dry_run:
        LOGGER.info("Dry run only; no files were modified.")
        return

    write_json(args.dataset_json, kept_records)
    write_json(args.summary_json, new_summary)
    write_histogram(kept_records, args.histogram_png, args.bin_size)
    LOGGER.info("Updated %s, %s, and %s", args.dataset_json, args.summary_json, args.histogram_png)


if __name__ == "__main__":
    main()
