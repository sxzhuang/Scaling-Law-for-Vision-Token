import argparse
import json
import logging
from pathlib import Path


DATASETS = {
    "novels": (Path("build_dataset_from_ebook/pride_dataset_square/pride_square_gt.json"), Path("build_dataset_from_ebook/pride_dataset_square/images")),
    "laws": (Path("build_dataset_from_ebook/laws_dataset_square/laws_square_gt.json"), Path("build_dataset_from_ebook/laws_dataset_square/images")),
    "economics": (Path("build_dataset_from_ebook/economics_dataset_square/economics_square_gt.json"), Path("build_dataset_from_ebook/economics_dataset_square/images")),
    "medicine": (Path("build_dataset_from_ebook/medicine_dataset_square/medicine_square_gt.json"), Path("build_dataset_from_ebook/medicine_dataset_square/images")),
    "newspapers": (Path("build_dataset_from_ebook/newspaper_dataset_square/newspaper_square_gt.json"), Path("build_dataset_from_ebook/newspaper_dataset_square/images")),
    "letters": (Path("build_dataset_from_ebook/letters_dataset_square/letters_square_gt.json"), Path("build_dataset_from_ebook/letters_dataset_square/images")),
}
TARGET_SIZES = [504, 644, 1036, 1288]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build Qwen OCR metadata for the six Figure 1 genres.")
    parser.add_argument("--output_dir", type=Path, default=Path("aaai_resub/qwen_fig1_vision_token/metadata"), help="Directory for per-genre metadata and the experiment manifest.")
    return parser.parse_args()


def build_genre_metadata(genre: str, source_json: Path, image_dir: Path) -> tuple[list[dict], dict]:
    records = json.loads(source_json.read_text(encoding="utf-8"))
    if not isinstance(records, list):
        raise ValueError(f"Expected a list in {source_json}")

    output_records = []
    metadata_names = set()
    for record in records:
        image_name = str(record.get("image", "")).strip()
        gt_text = record.get("gt")
        if not image_name or not isinstance(gt_text, str):
            raise ValueError(f"Invalid Figure 1 metadata record in {source_json}: {record}")
        image_path = image_dir / image_name
        if not image_path.is_file():
            raise FileNotFoundError(f"Missing source image: {image_path}")
        image_id = Path(image_name).stem
        if image_id in metadata_names:
            raise ValueError(f"Duplicate image id in {source_json}: {image_id}")
        metadata_names.add(image_id)
        output_records.append({
            "id": image_id,
            "context": gt_text,
            "image": image_name,
            "genre": genre,
            "source_image": str(image_path),
            "text_len": int(record.get("text_len", len(gt_text))),
            "layout": "28_6_0",
        })

    source_names = {path.stem for path in image_dir.glob("*.jpg")}
    missing_metadata = sorted(source_names - metadata_names)
    missing_images = sorted(metadata_names - source_names)
    if missing_metadata or missing_images:
        raise ValueError(f"Image/metadata mismatch for {genre}: images_without_metadata={len(missing_metadata)}, metadata_without_images={len(missing_images)}")

    summary = {
        "genre": genre,
        "source_metadata": str(source_json),
        "source_image_dir": str(image_dir),
        "sample_count": len(output_records),
    }
    return output_records, summary


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    genre_summaries = []
    for genre, (source_json, image_dir) in DATASETS.items():
        output_records, summary = build_genre_metadata(genre, source_json, image_dir)
        output_path = args.output_dir / f"{genre}.json"
        output_path.write_text(json.dumps(output_records, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        genre_summaries.append(summary)
        logging.info("Wrote %d %s records to %s", len(output_records), genre, output_path)

    manifest = {
        "layout": {"font_size": 28, "line_spacing": 6, "letter_spacing": 0},
        "target_sizes": TARGET_SIZES,
        "vision_tokens": {str(size): (size // 28) ** 2 for size in TARGET_SIZES},
        "genres": genre_summaries,
        "total_source_images": sum(item["sample_count"] for item in genre_summaries),
        "total_resized_images": sum(item["sample_count"] for item in genre_summaries) * len(TARGET_SIZES),
        "ocr_prompt_source": "VQAExp/run_ocr.py:OCR_PROMPT",
        "dynamic_max_tokens": "min(8192, max(64, ceil(1.2 * qwen_gt_token_length) + 32))",
    }
    manifest_path = args.output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    logging.info("Wrote experiment manifest to %s", manifest_path)


if __name__ == "__main__":
    main()
