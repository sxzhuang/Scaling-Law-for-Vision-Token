import argparse
import json
import logging
import math
import random
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from datasets import load_dataset


LOGGER = logging.getLogger(__name__)
DEFAULT_REPO_ID = "MLLM-CL/VTCBench"
DEFAULT_SPLIT = "Retrieval"


@dataclass(frozen=True)
class Evidence:
    name: str
    text: str
    start: int
    end: int


@dataclass(frozen=True)
class SourceItem:
    source_index: int
    problem: str
    context: str
    answers: list[str]
    evidences: list[Evidence]
    evidence_span_start: int
    evidence_span_end: int

    @property
    def context_len(self) -> int:
        return len(self.context)

    @property
    def evidence_span_len(self) -> int:
        return self.evidence_span_end - self.evidence_span_start


def parse_source(source) -> dict:
    if isinstance(source, str):
        return json.loads(source)
    return source


def contains_gold_answer(text: str, answers: list[str]) -> bool:
    return any(answer in text for answer in answers)


def find_evidence(context: str, name: str, text: str) -> Evidence | None:
    if not text:
        return None
    start = context.find(text)
    if start < 0:
        return None
    return Evidence(name=name, text=text, start=start, end=start + len(text))


def build_source_items(dataset) -> list[SourceItem]:
    source_items = []
    skipped = Counter()

    for source_index, row in enumerate(dataset):
        context = row.get("_context")
        if not context:
            skipped["missing_context"] += 1
            continue

        answers = [str(answer) for answer in row.get("answers", [])]
        if not answers:
            skipped["missing_answers"] += 1
            continue

        source = parse_source(row.get("_source", {}))
        evidences = []

        needle = find_evidence(context, "needle", source.get("needle"))
        if needle is None:
            skipped["missing_needle"] += 1
            continue
        evidences.append(needle)

        distractor_text = source.get("distractor")
        if distractor_text and contains_gold_answer(distractor_text, answers):
            distractor = find_evidence(context, "distractor", distractor_text)
            if distractor is None:
                skipped["missing_gold_distractor"] += 1
                continue
            evidences.append(distractor)

        span_start = min(evidence.start for evidence in evidences)
        span_end = max(evidence.end for evidence in evidences)
        source_items.append(SourceItem(source_index=source_index, problem=row["problem"], context=context, answers=answers, evidences=evidences, evidence_span_start=span_start, evidence_span_end=span_end))

    if skipped:
        LOGGER.info("Skipped source samples: %s", dict(skipped))
    return source_items


def feasible_for_bin(item: SourceItem, bin_start: int, bin_end: int) -> bool:
    min_target_len = max(bin_start, item.evidence_span_len)
    max_target_len = min(bin_end - 1, item.context_len)
    return min_target_len <= max_target_len


def choose_target_length(item: SourceItem, bin_start: int, bin_end: int, rng: random.Random) -> int:
    min_target_len = max(bin_start, item.evidence_span_len)
    max_target_len = min(bin_end - 1, item.context_len)
    if min_target_len > max_target_len:
        raise ValueError("Item is not feasible for this bin.")
    return rng.randint(min_target_len, max_target_len)


def crop_context(item: SourceItem, target_length: int, anchor_percent: int) -> str | None:
    anchor_position = (item.evidence_span_start + item.evidence_span_end) / 2
    ideal_start = round(anchor_position - target_length * anchor_percent / 100)

    min_crop_start = max(0, item.evidence_span_end - target_length)
    max_crop_start = min(item.evidence_span_start, item.context_len - target_length)
    if min_crop_start > max_crop_start:
        return None

    crop_start = min(max(ideal_start, min_crop_start), max_crop_start)
    crop_end = crop_start + target_length
    cropped_context = item.context[crop_start:crop_end]

    for answer in item.answers:
        if answer not in cropped_context:
            return None
    for evidence in item.evidences:
        if not (crop_start <= evidence.start and evidence.end <= crop_end):
            return None
    return cropped_context


def make_sample(sample_id: int, item: SourceItem, bin_start: int, bin_end: int, rng: random.Random) -> dict | None:
    for _ in range(100):
        target_length = choose_target_length(item, bin_start, bin_end, rng)
        anchor_percent = rng.randint(0, 100)
        cropped_context = crop_context(item, target_length, anchor_percent)
        if cropped_context is None:
            continue
        return {
            "id": f"retrieval_crop_{sample_id:06d}",
            "source_index": item.source_index,
            "problem": item.problem,
            "context": cropped_context,
            "answers": item.answers,
            "text_len": len(cropped_context),
        }
    return None


def build_bin_ranges(min_length: int, max_length: int, bin_size: int) -> list[tuple[int, int]]:
    ranges = []
    start = min_length
    while start < max_length:
        if start == min_length and min_length % bin_size != 0:
            end = min(((min_length // bin_size) + 1) * bin_size, max_length + 1)
        else:
            end = min(start + bin_size, max_length + 1)
        ranges.append((start, end))
        start = end
    return ranges


def largest_remainder_counts(weights: list[float], total: int) -> list[int]:
    raw_counts = [weight / sum(weights) * total for weight in weights]
    counts = [int(math.floor(count)) for count in raw_counts]
    remaining = total - sum(counts)
    remainders = sorted(((raw_count - count, index) for index, (raw_count, count) in enumerate(zip(raw_counts, counts))), reverse=True)
    for _, index in remainders[:remaining]:
        counts[index] += 1
    return counts


def build_bin_quotas(num_bins: int, total_samples: int, distribution: str, normal_sigma: float, edge_floor_weight: float) -> list[int]:
    if distribution == "uniform":
        weights = [1.0] * num_bins
    elif distribution == "normal":
        center = (num_bins - 1) / 2
        weights = []
        for index in range(num_bins):
            z = (index - center) / normal_sigma
            weights.append(edge_floor_weight + math.exp(-0.5 * z * z))
    else:
        raise ValueError(f"Unsupported distribution: {distribution}")
    return largest_remainder_counts(weights, total_samples)


def build_dataset(source_items: list[SourceItem], total_samples: int, min_length: int, max_length: int, bin_size: int, seed: int, distribution: str, normal_sigma: float, edge_floor_weight: float) -> list[dict]:
    rng = random.Random(seed)
    bin_ranges = build_bin_ranges(min_length, max_length, bin_size)
    quotas = build_bin_quotas(len(bin_ranges), total_samples, distribution, normal_sigma, edge_floor_weight)
    samples = []
    used_source_indices = set()

    LOGGER.info("Bin distribution: %s", distribution)
    LOGGER.info("Bin quotas: %s", quotas)

    for bin_index, ((bin_start, bin_end), quota) in enumerate(zip(bin_ranges, quotas)):
        selected = []
        unique_candidates = [item for item in source_items if item.source_index not in used_source_indices and feasible_for_bin(item, bin_start, bin_end)]
        rng.shuffle(unique_candidates)
        selected.extend(unique_candidates[:quota])

        if len(selected) < quota:
            duplicate_candidates = [item for item in source_items if feasible_for_bin(item, bin_start, bin_end)]
            if duplicate_candidates:
                selected.extend(rng.choice(duplicate_candidates) for _ in range(quota - len(selected)))

        created_in_bin = 0
        for item in selected:
            sample = make_sample(len(samples), item, bin_start, bin_end, rng)
            if sample is None:
                continue
            samples.append(sample)
            used_source_indices.add(item.source_index)
            created_in_bin += 1

        LOGGER.info("Bin %02d [%d, %d): requested=%d created=%d", bin_index, bin_start, bin_end, quota, created_in_bin)

    if len(samples) < total_samples:
        LOGGER.warning("Created %d samples, fewer than requested %d.", len(samples), total_samples)
    return samples[:total_samples]


def save_json(samples: list[dict], output_file: Path) -> None:
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with output_file.open("w", encoding="utf-8") as f:
        json.dump(samples, f, ensure_ascii=False, indent=2)
        f.write("\n")


def log_length_distribution(samples: list[dict], min_length: int, max_length: int, bin_size: int) -> None:
    bin_ranges = build_bin_ranges(min_length, max_length, bin_size)
    counts = Counter()
    for sample in samples:
        text_len = sample["text_len"]
        for bin_index, (bin_start, bin_end) in enumerate(bin_ranges):
            if bin_start <= text_len < bin_end:
                counts[bin_index] += 1
                break
    for bin_index, (bin_start, bin_end) in enumerate(bin_ranges):
        LOGGER.info("Output bin %02d [%d, %d): %d", bin_index, bin_start, bin_end, counts[bin_index])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a cropped Retrieval dataset with configurable character length bins.")
    parser.add_argument("--repo_id", type=str, default=DEFAULT_REPO_ID, help="Hugging Face dataset repo id.")
    parser.add_argument("--split", type=str, default=DEFAULT_SPLIT, help="Dataset split to crop.")
    parser.add_argument("--total_samples", type=int, default=600, help="Number of output samples.")
    parser.add_argument("--min_length", type=int, default=500, help="Minimum cropped context length.")
    parser.add_argument("--max_length", type=int, default=25000, help="Maximum cropped context length.")
    parser.add_argument("--bin_size", type=int, default=1000, help="Character length bin size.")
    parser.add_argument("--distribution", type=str, choices=["uniform", "normal"], default="normal", help="Target length-bin distribution.")
    parser.add_argument("--normal_sigma", type=float, default=6.0, help="Sigma in bin units for the normal target distribution.")
    parser.add_argument("--edge_floor_weight", type=float, default=0.35, help="Baseline weight added to every bin so edge bins are not too sparse.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--output_file", type=Path, default=Path("VQAExp/data/Retrieval/retrieval_cropped_0_25k_600.json"), help="Output JSON file.")
    parser.add_argument("--cache_dir", type=Path, default=None, help="Optional Hugging Face cache directory.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    LOGGER.info("Loading %s split from %s", args.split, args.repo_id)
    dataset = load_dataset(args.repo_id, split=args.split, cache_dir=str(args.cache_dir) if args.cache_dir else None)
    remove_columns = [column for column in ["images", "_render_args"] if column in dataset.column_names]
    if remove_columns:
        dataset = dataset.remove_columns(remove_columns)

    source_items = build_source_items(dataset)
    LOGGER.info("Usable source samples: %d", len(source_items))

    samples = build_dataset(source_items, args.total_samples, args.min_length, args.max_length, args.bin_size, args.seed, args.distribution, args.normal_sigma, args.edge_floor_weight)
    save_json(samples, args.output_file)
    log_length_distribution(samples, args.min_length, args.max_length, args.bin_size)
    LOGGER.info("Saved %d samples to %s", len(samples), args.output_file)


if __name__ == "__main__":
    main()
