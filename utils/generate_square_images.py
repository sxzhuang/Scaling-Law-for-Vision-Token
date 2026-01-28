#!/usr/bin/env python3
"""Estimate vs actual rendered image sizes for square-ish layouts."""

from __future__ import annotations

import math
import re
import textwrap
from io import BytesIO
from pathlib import Path
from typing import TypedDict

from PIL import Image, ImageDraw, ImageFont

MIN_WRAP_WIDTH = 20
MAX_WRAP_WIDTH = 400


def wrap_text(text: str, width: int) -> list[str]:
    text = text.replace("\n", " ").strip()
    paragraphs = [p.strip() for p in re.split(r"\s{2,}", text) if p.strip()]
    if not paragraphs:
        paragraphs = [text]

    lines: list[str] = []
    for paragraph in paragraphs:
        wrapped = textwrap.wrap(paragraph, width=width)
        lines.extend(wrapped or [""])
        lines.append("")
    if lines:
        lines.pop()
    return lines or [""]


def adjust_wrap_width(
    current: int,
    ratio: float,
    ratio_min: float = 0.9,
    ratio_max: float = 1.1,
) -> int:
    if not math.isfinite(ratio) or ratio <= 0:
        return current
    if ratio > ratio_max:
        scale = 1.0 / math.sqrt(ratio)
        next_value = max(MIN_WRAP_WIDTH, int(round(current * scale)))
        if next_value >= current:
            next_value = max(MIN_WRAP_WIDTH, current - 5)
        return next_value
    if ratio < ratio_min:
        scale = math.sqrt(1.0 / ratio)
        next_value = min(MAX_WRAP_WIDTH, int(round(current * scale)))
        if next_value <= current:
            next_value = min(MAX_WRAP_WIDTH, current + 5)
        return next_value
    return current


def load_font(font_size: int, font_path: Path | None = None) -> ImageFont.ImageFont:
    if font_path is not None:
        return ImageFont.truetype(str(font_path), font_size)
    try:
        return ImageFont.truetype("DejaVuSans.ttf", font_size)
    except OSError:
        return ImageFont.load_default()


def line_width(text: str, font: ImageFont.ImageFont) -> float:
    if hasattr(font, "getlength"):
        return float(font.getlength(text))
    bbox = font.getbbox(text or " ")
    return float(bbox[2] - bbox[0])


def average_character_width(font: ImageFont.ImageFont) -> float:
    sample = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
    widths = [line_width(ch, font) for ch in sample]
    return sum(widths) / len(widths)


def estimated_line_width(line: str, avg_char_width: float, letter_spacing: int) -> float:
    if not line:
        return 0.0
    spacing = max(len(line) - 1, 0) * letter_spacing
    return len(line) * avg_char_width + spacing


def precise_line_width(line: str, font: ImageFont.ImageFont, letter_spacing: int) -> float:
    width = sum(line_width(ch, font) for ch in line)
    spacing = max(len(line) - 1, 0) * letter_spacing
    return width + spacing


def draw_text_with_letter_spacing(
    draw: ImageDraw.ImageDraw,
    text: str,
    font: ImageFont.ImageFont,
    x: float,
    y: float,
    letter_spacing: int,
) -> None:
    cursor = x
    for idx, char in enumerate(text):
        draw.text((cursor, y), char, font=font, fill="black")
        cursor += line_width(char, font)
        if idx < len(text) - 1:
            cursor += letter_spacing


def draw_justified_line(
    draw: ImageDraw.ImageDraw,
    line: str,
    font: ImageFont.ImageFont,
    x: int,
    y: int,
    max_width: float,
    letter_spacing: int,
) -> None:
    stripped = line.strip()
    if not stripped:
        draw_text_with_letter_spacing(draw, line, font, x, y, letter_spacing)
        return
    words = stripped.split()
    if len(words) == 1:
        draw_text_with_letter_spacing(draw, stripped, font, x, y, letter_spacing)
        return
    word_width = sum(precise_line_width(word, font, letter_spacing) for word in words)
    space_width = line_width(" ", font) + letter_spacing
    gaps = len(words) - 1
    base_line_width = word_width + space_width * gaps
    extra_space = max(max_width - base_line_width, 0.0)
    per_gap = space_width + (extra_space / gaps if gaps else 0.0)
    cursor = x
    for idx, word in enumerate(words):
        draw_text_with_letter_spacing(draw, word, font, cursor, y, letter_spacing)
        cursor += precise_line_width(word, font, letter_spacing)
        if idx < gaps:
            cursor += per_gap


def initial_wrap_width(
    text: str,
    font: ImageFont.ImageFont,
    line_spacing: int,
    letter_spacing: int,
) -> int:
    normalized = re.sub(r"\s+", " ", text.strip())
    if not normalized:
        return MIN_WRAP_WIDTH
    char_count = len(normalized)
    avg_char_width = average_character_width(font) + max(letter_spacing, 0)
    bbox = font.getbbox("Hg")
    line_height = (bbox[3] - bbox[1]) + line_spacing
    numerator = char_count * line_height
    denom = max(avg_char_width, 1e-6)
    approx = int(round(math.sqrt(numerator / denom)))
    approx = max(MIN_WRAP_WIDTH, min(MAX_WRAP_WIDTH, approx))
    return approx or MIN_WRAP_WIDTH


def estimate_dimensions(
    lines: list[str],
    font: ImageFont.ImageFont,
    line_spacing: int,
    letter_spacing: int,
    padding: int = 20,
) -> tuple[int, int]:
    avg_char_width = average_character_width(font)
    bbox = font.getbbox("Hg")
    line_height = (bbox[3] - bbox[1]) + line_spacing
    max_width = max(
        (estimated_line_width(line, avg_char_width, letter_spacing) for line in lines),
        default=0.0,
    )
    width = int(max_width) + padding * 2 or padding * 2
    height = line_height * max(1, len(lines)) + padding * 2
    return width, height


def render_actual_image(
    lines: list[str],
    font: ImageFont.ImageFont,
    line_spacing: int,
    letter_spacing: int,
    padding: int = 20,
) -> Image.Image:
    bbox = font.getbbox("Hg")
    line_height = (bbox[3] - bbox[1]) + line_spacing
    max_width = max(
        (precise_line_width(line, font, letter_spacing) for line in lines),
        default=0.0,
    )
    width = int(math.ceil(max_width)) + padding * 2 or padding * 2
    height = line_height * max(1, len(lines)) + padding * 2
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    y = padding
    for idx, line in enumerate(lines):
        is_last = idx == len(lines) - 1
        if not line.strip() or is_last:
            draw_text_with_letter_spacing(draw, line, font, padding, y, letter_spacing)
        else:
            draw_justified_line(
                draw=draw,
                line=line,
                font=font,
                x=padding,
                y=y,
                max_width=max_width,
                letter_spacing=letter_spacing,
            )
        y += line_height
    return image


def attempt_layout(
    text: str,
    wrap_width: int,
    font: ImageFont.ImageFont,
    line_spacing: int,
    letter_spacing: int,
    padding: int = 20,
) -> tuple[SizeComparison, float]:
    lines = wrap_text(text, wrap_width)
    estimated_size = estimate_dimensions(
        lines, font, line_spacing, letter_spacing, padding=padding
    )
    actual_image = render_actual_image(
        lines, font, line_spacing, letter_spacing, padding=padding
    )
    actual_size = actual_image.size
    width, height = actual_size
    ratio = (width / height) if height else float("inf")
    difference = (width - estimated_size[0], height - estimated_size[1])
    result: SizeComparison = {
        "estimated_size": estimated_size,
        "actual_size": actual_size,
        "difference": difference,
        "image": actual_image,
        "lines": lines,
    }
    return result, ratio


class SizeComparison(TypedDict):
    estimated_size: tuple[int, int]
    actual_size: tuple[int, int]
    difference: tuple[int, int]
    lines: list[str]
    image: Image.Image


class RenderOutput(TypedDict):
    image_shape: tuple[int, int]
    image: Image.Image
    pdf: bytes
    character_rect: tuple[float, float]


def estimate_image_size(
    text: str,
    font_size: int,
    line_spacing: int,
    letter_spacing: int,
    font: ImageFont.ImageFont | None = None,
    padding: int = 20,
    ratio_min: float = 0.9,
    ratio_max: float = 1.1,
    max_layout_attempts: int = 3,
) -> SizeComparison:
    font = font or load_font(font_size)
    normalized_text = text.strip()
    wrap_width = initial_wrap_width(normalized_text, font, line_spacing, letter_spacing)
    best_result: SizeComparison | None = None
    best_delta = float("inf")
    tried: set[int] = set()

    for _ in range(max_layout_attempts):
        tried.add(wrap_width)
        result, ratio = attempt_layout(
            text=normalized_text,
            wrap_width=wrap_width,
            font=font,
            line_spacing=line_spacing,
            letter_spacing=letter_spacing,
            padding=padding,
        )
        delta = abs(1.0 - ratio)
        if delta < best_delta:
            best_delta = delta
            best_result = result
        if ratio_min <= ratio <= ratio_max:
            return result
        next_wrap = adjust_wrap_width(wrap_width, ratio, ratio_min=ratio_min, ratio_max=ratio_max)
        if next_wrap in tried or next_wrap == wrap_width:
            break
        wrap_width = next_wrap

    if best_result is None:
        raise RuntimeError("Unable to generate layout for the provided text.")
    return best_result


def character_rectangle(
    font: ImageFont.ImageFont,
    line_spacing: int,
    letter_spacing: int,
) -> tuple[float, float]:
    avg_width = average_character_width(font)
    glyph_bbox = font.getbbox("Hg")
    glyph_height = glyph_bbox[3] - glyph_bbox[1]
    width = max(avg_width + (letter_spacing / 2.0), 1e-6)
    height = max(glyph_height + (line_spacing / 2.0), 1e-6)
    return width, height


def render_text_assets(
    text: str,
    font_size: int,
    line_spacing: int,
    letter_spacing: int,
    font_path: Path | None = None,
    pdf_dpi: int = 300,
    padding: int = 20,
    ratio_min: float = 0.9,
    ratio_max: float = 1.1,
    max_layout_attempts: int = 3,
) -> RenderOutput:
    """Render text into an image and PDF bytes along with shape metadata."""
    font = load_font(font_size, font_path)
    size_result = estimate_image_size(
        text=text,
        font_size=font_size,
        line_spacing=line_spacing,
        letter_spacing=letter_spacing,
        font=font,
        padding=padding,
        ratio_min=ratio_min,
        ratio_max=ratio_max,
        max_layout_attempts=max_layout_attempts,
    )
    char_rect = character_rectangle(
        font=font,
        line_spacing=line_spacing,
        letter_spacing=letter_spacing,
    )
    rgb_image = size_result["image"].convert("RGB")
    buffer = BytesIO()
    rgb_image.save(buffer, "PDF", resolution=pdf_dpi)
    pdf_bytes = buffer.getvalue()
    return {
        "image_shape": size_result["actual_size"],
        "image": size_result["image"],
        "pdf": pdf_bytes,
        "character_rect": char_rect,
    }


def estimate_information_density(
    image_size: tuple[int, int],
    character_rect: tuple[float, float],
    patch_size: int = 16,
    resolution: int = 1024,
) -> float:
    """Estimate single-character capacity inside a patch at a given resolution."""
    if patch_size <= 0 or resolution <= 0:
        raise ValueError("patch_size and resolution must be positive.")
    width, height = image_size
    if width <= 0 or height <= 0:
        return 0.0
    char_w, char_h = character_rect
    if char_w <= 0 or char_h <= 0:
        return 0.0
    scale = resolution / max(width, height)
    char_w_resized = char_w * scale
    char_h_resized = char_h * scale
    if char_w_resized <= 0 or char_h_resized <= 0:
        return 0.0
    patch_area = float(patch_size * patch_size)
    char_area = char_w_resized * char_h_resized
    return round(patch_area / char_area, 5)


def save_image_and_pdf(
    image: Image.Image,
    pdf: bytes,
    image_path: Path,
    pdf_path: Path,
    dpi: int = 300,
) -> None:
    """Save the rendered image and PDF to disk following the builder convention."""
    image_path = Path(image_path)
    pdf_path = Path(pdf_path)
    image_path.parent.mkdir(parents=True, exist_ok=True)
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    rgb_image = image.convert("RGB")
    if pdf:
        pdf_path.write_bytes(pdf)
    else:
        rgb_image.save(pdf_path, "PDF", resolution=dpi)
    rgb_image.save(image_path, "JPEG", quality=95, dpi=(dpi, dpi))


def main() -> None:
    sample_text: str = "I am, therefore, by no means discouraged by what you have just said, and shall hope to lead you to the altar ere long.” “Upon my word, sir,” cried Elizabeth, “your hope is rather an extraordinary one after my declaration. I do assure you that I am not one of those young ladies (if such young ladies there are) who are so daring as to risk their happiness on the chance of being asked a second time. I am perfectly serious in my refusal. You could not make _me_ happy, and I am convinced that I am the last woman in the world who would make _you_ so. Nay, were your friend Lady Catherine to know me, I am persuaded she would find me in every respect ill qualified for the situation.” “Were it certain that Lady Catherine would think so,” said Mr. Collins, very gravely--“but I cannot imagine that her Ladyship would at all disapprove of you. And you may be certain that when I have the honour of seeing her again I shall speak in the highest terms of your modesty, economy, and other amiable qualifications.” “Indeed, Mr. Collins, all praise of me will be unnecessary. You must give me leave to judge for myself, and pay me the compliment of believing what I say. I wish you very happy and very rich, and by refusing your hand, do all in my power to prevent your being otherwise. In making me the offer, you must have satisfied the delicacy of your feelings with regard to my family, and may take possession of Longbourn estate whenever it falls, without any self-reproach. This matter may be considered, therefore, as finally settled.” And rising as she thus spoke, she would have quitted the room, had not Mr. Collins thus addressed her,-- “When I do myself the honour of speaking to you next on the subject, I shall hope to receive a more favourable answer than you have now given me; though I am far from accusing you of cruelty at present, because I know it to be the established custom of your sex to reject a man on the first application, and, perhaps, you have even now said as much to encourage my suit as would be consistent with the true delicacy of the female character.” “Really, Mr. Collins,” cried Elizabeth, with some warmth, “you puzzle me exceedingly. If what I have hitherto said can appear to you in the form of encouragement, I know not how to express my refusal in such a way as may convince you of its being one.” “You must give me leave to flatter myself, my dear cousin, that your refusal of my addresses are merely words of course. My reasons for believing it are briefly these:--It does not appear to me that my hand is unworthy of your acceptance, or that the establishment I can offer would be any other than highly desirable. My situation in life, my connections with the family of De Bourgh, and my relationship to your own, are circumstances highly in my favour; and you should take it into further consideration that, in spite of your manifold attractions, it is by no means certain that another offer of marriage may ever be made you."
    font_size = 28
    line_spacing = 2
    letter_spacing = -1
    render_result = render_text_assets(
        text=sample_text,
        font_size=font_size,
        line_spacing=line_spacing,
        letter_spacing=letter_spacing,
    )
    capacity = estimate_information_density(
        image_size=render_result["image_shape"],
        character_rect=render_result["character_rect"],
        patch_size=16,
        resolution=1024,
    )
    print("Image shape:", render_result["image_shape"])
    print("Estimated characters per 16x16 patch:", capacity)
    save_image_and_pdf(
        image=render_result["image"],
        pdf=render_result["pdf"],
        image_path=Path("sample_square_layout_gc.jpg"),
        pdf_path=Path("sample_square_layout_gc.pdf"),
    )


if __name__ == "__main__":
    main()
