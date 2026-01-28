#!/usr/bin/env python3
"""Render a single GT entry into JPG/PDF with custom spacing."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

from PIL import ImageDraw

from utils.generate_square_images import load_font, render_text_assets, save_image_and_pdf


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render one GT entry into JPG/PDF with custom typography.")
    parser.add_argument("--id", type=str, required=True, help="Sample id (e.g., pride_prejudice_double_C20B1_C25B3).")
    parser.add_argument("--metadata", type=Path, default=Path("build_dataset_from_ebook/pride_dataset_square/pride_square_gt.json"), help="Path to metadata JSON.")
    parser.add_argument("--output_dir", type=Path, default=Path("test_render"), help="Root directory for outputs (image/pdf).")
    parser.add_argument("--font_path", type=Path, default=None, help="Optional TTF/OTF font path.")
    parser.add_argument("--font_size", type=int, default=28, help="Font size.")
    parser.add_argument("--line_spacing", type=int, default=24, help="Line spacing.")
    parser.add_argument("--letter_spacing", type=int, default=6, help="Letter spacing.")
    parser.add_argument("--padding", type=int, default=20, help="Padding around text in pixels.")
    parser.add_argument("--dpi", type=int, default=300, help="DPI for saved assets.")
    parser.add_argument("--ratio_min", type=float, default=0.9, help="Minimum width/height ratio allowed.")
    parser.add_argument("--ratio_max", type=float, default=1.1, help="Maximum width/height ratio allowed.")
    return parser.parse_args()


def load_gt_text(metadata_path: Path, sample_id: str) -> str:
    data: Any = json.loads(metadata_path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"{metadata_path} 必须是记录列表。")
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
                raise ValueError(f"在 {metadata_path} 中找到 {target} ，但 gt 为空。")
            return gt_text
    raise KeyError(f"在 {metadata_path} 中未找到 id: {target}")


def build_output_paths(output_dir: Path, sample_id: str) -> tuple[Path, Path]:
    image_dir = output_dir / "image"
    pdf_dir = output_dir / "pdf"
    image_dir.mkdir(parents=True, exist_ok=True)
    pdf_dir.mkdir(parents=True, exist_ok=True)
    return image_dir / f"{sample_id}.jpg", pdf_dir / f"{sample_id}.pdf"


def estimate_char_size(text: str, font_path: Path | None, font_size: int) -> tuple[float, float]:
    font = load_font(font_size, font_path)
    char_count = len(text)
    total_width = font.getlength(text) if char_count else 0.0
    avg_width = total_width / char_count if char_count else 0.0
    bbox = font.getbbox("Hg")
    avg_height = float(bbox[3] - bbox[1])
    return avg_width, avg_height


def main() -> None:
    args = parse_args()
    gt_text = load_gt_text(args.metadata, args.id)
    avg_char_width, avg_char_height = estimate_char_size(gt_text, args.font_path, args.font_size)
    estimated_char_width = avg_char_width + args.letter_spacing / 2.0
    estimated_char_height = avg_char_height + args.line_spacing / 2.0
    render_result = render_text_assets(
        text=gt_text,
        font_size=args.font_size,
        line_spacing=args.line_spacing,
        letter_spacing=args.letter_spacing,
        font_path=args.font_path,
        pdf_dpi=args.dpi,
        padding=args.padding,
        ratio_min=args.ratio_min,
        ratio_max=args.ratio_max,
    )
    image_path, pdf_path = build_output_paths(args.output_dir, Path(args.id).stem)
    image_with_rect = render_result["image"].copy()
    draw = ImageDraw.Draw(image_with_rect)
    rect_width = max(estimated_char_width, 1.0)
    rect_height = max(estimated_char_height, 1.0)
    draw.rectangle((0, 0, rect_width, rect_height), outline="red", width=2)
    img_w, img_h = image_with_rect.size
    max_x_red = max(img_w - rect_width, 0)
    max_y_red = max(img_h - rect_height, 0)
    for _ in range(10):
        x0 = random.uniform(0, max_x_red) if max_x_red > 0 else 0.0
        y0 = random.uniform(0, max_y_red) if max_y_red > 0 else 0.0
        draw.rectangle((x0, y0, x0 + rect_width, y0 + rect_height), outline="red", width=2)
    base_width = max(avg_char_width, 1.0)
    base_height = max(avg_char_height, 1.0)
    max_x_green = max(img_w - base_width, 0)
    max_y_green = max(img_h - base_height, 0)
    for _ in range(10):
        x0 = random.uniform(0, max_x_green) if max_x_green > 0 else 0.0
        y0 = random.uniform(0, max_y_green) if max_y_green > 0 else 0.0
        draw.rectangle((x0, y0, x0 + base_width, y0 + base_height), outline="green", width=2)
    save_image_and_pdf(image=image_with_rect, pdf=None, image_path=image_path, pdf_path=pdf_path, dpi=args.dpi)
    image_w, image_h = render_result["image_shape"]
    scale_w = 1024 / image_w if image_w else 0.0
    scale_h = 1024 / image_h if image_h else 0.0
    resized_char_w = estimated_char_width * scale_w
    resized_char_h = estimated_char_height * scale_h
    patch_area = 16 * 16
    char_area = resized_char_w * resized_char_h if resized_char_w > 0 and resized_char_h > 0 else 0.0
    density = (patch_area / char_area) if char_area > 0 else 0.0
    print(f"估计字符宽度: {avg_char_width:.2f}px，高度: {avg_char_height:.2f}px")
    print(f"含间距后估计字符占用: {estimated_char_width:.2f}px × {estimated_char_height:.2f}px")
    print(f"resize到1024后字符占用: {resized_char_w:.2f}px × {resized_char_h:.2f}px")
    print(f"16x16 patch 近似密度: {density:.4f}")
    print(f"生成完成：{image_path} 和 {pdf_path}，尺寸 {render_result['image_shape']}")


if __name__ == "__main__":
    main()
