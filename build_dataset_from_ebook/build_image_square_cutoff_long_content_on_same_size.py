#!/usr/bin/env python3
"""Generate cut-off text variants, render images/PDFs, and record token lengths."""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
from typing import Any, List, Tuple

from PIL import Image
from transformers import AutoTokenizer

from utils.generate_square_images import render_text_assets


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render cut-off combinations from a GT entry and paste to base canvas.")
    parser.add_argument(
        "--ids",
        type=str,
        nargs="+",
        default=[
            "pride_prejudice_quintuple_C10B4_C18B2_C20B3_C51B4_C52B1",
            "pride_prejudice_quintuple_C18B4_C32B2_C40B1_C43B4_C53B1",
            "pride_prejudice_quintuple_C5B2_C10B4_C18B1_C43B1_C58B4",
            "pride_prejudice_quintuple_C9B3_C10B1_C41B1_C41B4_C45B3"
        ],
        help="Target ids (e.g., pride_prejudice_quintuple_C5B2_C10B4_C18B1_C43B1_C58B4).",
    )
    parser.add_argument("--metadata", type=Path, default=Path("build_dataset_from_ebook/pride_dataset_square/pride_square_gt.json"), help="Path to pride_square_gt.json.")
    parser.add_argument("--output_root", type=Path, default=Path("build_dataset_from_ebook/pride_dataset_square_cutoff_content"), help="Root directory for outputs (images/pdf/json).")
    parser.add_argument("--font_path", type=Path, default=None, help="Optional font path.")
    parser.add_argument("--font_size", type=int, default=28, help="Font size.")
    parser.add_argument("--line_spacing", type=int, default=6, help="Line spacing.")
    parser.add_argument("--letter_spacing", type=int, default=0, help="Letter spacing.")
    parser.add_argument("--padding", type=int, default=20, help="Padding in pixels.")
    parser.add_argument("--dpi", type=int, default=300, help="DPI for rendered assets.")
    parser.add_argument("--ratio_min", type=float, default=0.9, help="Minimum width/height ratio.")
    parser.add_argument("--ratio_max", type=float, default=1.1, help="Maximum width/height ratio.")
    parser.add_argument("--tokenizer_name", type=str, default="Qwen/Qwen2.5-0.5B-Instruct", help="Tokenizer for token length counting.")
    return parser.parse_args()


def load_metadata_text(metadata_path: Path, sample_id: str) -> str:
    data: Any = json.loads(metadata_path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"{metadata_path} must be a list of records.")
    target = Path(sample_id.strip()).stem
    for record in data:
        if not isinstance(record, dict):
            continue
        image_name = str(record.get("image", "")).strip()
        if not image_name:
            continue
        if Path(image_name).stem == target:
            gt_text = str(record.get("gt", "")).strip()
            if not gt_text:
                raise ValueError(f"Found {target} but gt is empty.")
            return gt_text
    raise KeyError(f"id not found in metadata: {target}")


def split_into_segments(text: str, segments: int = 5) -> List[str]:
    words = text.split()
    if segments <= 0:
        return [" ".join(words)]
    total = len(words)
    base = total // segments
    remainder = total % segments
    result: List[str] = []
    start = 0
    for idx in range(segments):
        length = base + (1 if idx < remainder else 0)
        end = start + length
        slice_words = words[start:end]
        result.append(" ".join(slice_words).strip())
        start = end
    return result


def combinations_by_length(parts: List[str]) -> List[Tuple[int, int, str]]:
    combos: List[Tuple[int, int, str]] = []
    total_parts = len(parts)
    for k in range(total_parts, 0, -1):
        for combo_idx, indices in enumerate(itertools.combinations(range(total_parts), k), start=1):
            fragments = [parts[i] for i in indices if parts[i]]
            text = " ".join(fragments).strip()
            combos.append((k, combo_idx, text))
    return combos


def compute_token_len(text: str, tokenizer: AutoTokenizer) -> int:
    if not text:
        return 0
    tokens = tokenizer(text, add_special_tokens=True, truncation=False)
    return len(tokens.get("input_ids", []))


def main() -> None:
    args = parse_args()
    output_images = args.output_root / "images"
    output_pdfs = args.output_root / "pdf"
    output_json = args.output_root / "pride_square_cutoff_gt.json"
    output_images.mkdir(parents=True, exist_ok=True)
    output_pdfs.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name)

    records: List[dict[str, Any]] = []
    for sample_id in args.ids:
        base_text = load_metadata_text(args.metadata, sample_id)
        segments = split_into_segments(base_text, segments=5)
        combos = combinations_by_length(segments)

        base_canvas_size: Tuple[int, int] | None = None
        stem = Path(sample_id).stem

        for k, combo_idx, text in combos:
            if not text:
                continue
            render_result = render_text_assets(
                text=text,
                font_size=args.font_size,
                line_spacing=args.line_spacing,
                letter_spacing=args.letter_spacing,
                font_path=args.font_path,
                pdf_dpi=args.dpi,
                padding=args.padding,
                ratio_min=args.ratio_min,
                ratio_max=args.ratio_max,
            )
            image = render_result["image"].convert("RGB")
            if base_canvas_size is None and k == 5:
                base_canvas_size = image.size
            canvas_size = base_canvas_size or image.size
            canvas = Image.new("RGB", canvas_size, "white")
            offset = ((canvas_size[0] - image.size[0]) // 2, (canvas_size[1] - image.size[1]) // 2)
            canvas.paste(image, offset)

            filename = f"{stem}_{k}_{combo_idx}.jpg"
            image_path = output_images / filename
            pdf_path = output_pdfs / f"{stem}_{k}_{combo_idx}.pdf"
            canvas.save(image_path, "JPEG", quality=95, dpi=(args.dpi, args.dpi))
            canvas.save(pdf_path, "PDF", resolution=args.dpi)

            records.append(
                {
                    "image": filename,
                    "gt": text,
                    "text_token_len": compute_token_len(text, tokenizer),
                }
            )

    output_json.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Generated {len(records)} variants for {len(args.ids)} ids. Records saved to {output_json}.")


if __name__ == "__main__":
    main()
