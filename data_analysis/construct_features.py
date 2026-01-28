import argparse
import csv
import json
import logging
from pathlib import Path
from typing import Any

from PIL import Image


logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

CHARACTER_SIZE_BY_FONT = {
    20.0: (10.0369, 19.0),
    28.0: (14.0533, 26.0),
    36.0: (18.0624, 33.0),
}


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def build_token_length_map(meta_data: Any) -> dict[str, int]:
    token_map: dict[str, int] = {}
    if isinstance(meta_data, dict):
        items = meta_data.values()
    else:
        items = meta_data
    for entry in items:
        image_name = entry.get("image")
        token_len = entry.get("text_token_len")
        if isinstance(image_name, str) and image_name.endswith(".jpg") and isinstance(token_len, int):
            token_map[image_name[:-4]] = token_len
    return token_map


def build_text_length_map(meta_data: Any) -> dict[str, int]:
    text_map: dict[str, int] = {}
    if isinstance(meta_data, dict):
        items = meta_data.values()
    else:
        items = meta_data
    for entry in items:
        image_name = entry.get("image")
        text_len = entry.get("text_len")
        if isinstance(image_name, str) and image_name.endswith(".jpg") and isinstance(text_len, int):
            text_map[image_name[:-4]] = text_len
    return text_map


def build_edit_distance_map(edit_distance_data: Any) -> dict[str, float | None]:
    edit_distance_map: dict[str, float | None] = {}
    if edit_distance_data is None:
        return edit_distance_map
    if isinstance(edit_distance_data, dict):
        entries = edit_distance_data.values()
    else:
        entries = edit_distance_data
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        sample_id = entry.get("id")
        if not isinstance(sample_id, str):
            continue
        value = entry.get("edit_distance")
        if value is None:
            value = entry.get("edit_dist")
        if value is None:
            edit_distance_map[sample_id] = None
            continue
        try:
            edit_distance_map[sample_id] = float(value)
        except (TypeError, ValueError):
            logging.warning("Skipping %s: invalid edit distance value %s", sample_id, value)
    return edit_distance_map


def build_rows(
    density_data: Any,
    token_map: dict[str, int],
    text_map: dict[str, int],
    image_dir: Path,
    font_size: int,
    line_spacing: int,
    letter_spacing: int,
    resolution: int,
    character_width: float,
    character_height: float,
    edit_distance_map: dict[str, float | None] | None,
) -> list[list[Any]]:
    rows: list[list[Any]] = []
    if isinstance(density_data, dict):
        entries = density_data.values()
    else:
        entries = density_data

    for entry in entries:
        if not isinstance(entry, dict):
            continue
        raw_id = entry.get("id")
        if raw_id is None:
            raw_id = entry.get("image_id")
        if not isinstance(raw_id, str) or not raw_id.strip():
            continue
        image_id = Path(raw_id.strip()).stem
        image_path = image_dir / f"{image_id}.jpg"
        if not image_path.exists():
            logging.warning("Skipping %s: missing image file %s", image_id, image_path)
            continue

        patch_density = entry.get("capacity")
        if patch_density is None:
            patch_density = entry.get("patch_density")
        if patch_density is None:
            continue
        try:
            patch_density = float(patch_density)
        except (TypeError, ValueError):
            logging.warning("Skipping %s: invalid patch density value %s", image_id, patch_density)
            continue

        token_len = entry.get("text_token_len")
        if token_len is None:
            token_len = entry.get("token_len")
        if token_len is None:
            token_len = token_map.get(image_id)
        if token_len is None:
            logging.warning("Skipping %s: missing token length", image_id)
            continue
        try:
            token_len = int(token_len)
        except (TypeError, ValueError):
            logging.warning("Skipping %s: invalid token length value %s", image_id, token_len)
            continue

        text_len = text_map.get(image_id)
        if text_len is None:
            logging.warning("Skipping %s: missing text length", image_id)
            continue
        try:
            text_len = int(text_len)
        except (TypeError, ValueError):
            logging.warning("Skipping %s: invalid text length value %s", image_id, text_len)
            continue
        try:
            with Image.open(image_path) as image:
                image_w, image_h = image.size
        except OSError as exc:
            logging.warning("Skipping %s: cannot open image %s (%s)", image_id, image_path, exc)
            continue
        if not image_w or not image_h:
            logging.warning("Skipping %s: invalid image size %s x %s", image_id, image_w, image_h)
            continue
        model_w = character_width * resolution / image_w
        model_h = character_height * resolution / image_h

        edit_distance = entry.get("edit_distance")
        if edit_distance is None:
            edit_distance = entry.get("edit_dist")
        if edit_distance is not None:
            try:
                edit_distance = float(edit_distance)
            except (TypeError, ValueError):
                logging.warning("Skipping %s: invalid edit distance value %s", image_id, edit_distance)
                edit_distance = None
        elif edit_distance_map is not None:
            edit_distance = edit_distance_map.get(image_id)

        rows.append([
            image_id,
            patch_density,
            font_size,
            line_spacing,
            letter_spacing,
            token_len,
            text_len,
            resolution,
            image_w,
            image_h,
            character_width,
            character_height,
            model_w,
            model_h,
            edit_distance,
        ])
    return rows


def write_csv(rows: list[list[Any]], output_csv: Path, overwrite: bool) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    file_exists = output_csv.exists()
    if overwrite and file_exists:
        confirm = input(f"Overwrite existing file {output_csv}? [y/N]: ").strip().lower()
        if confirm not in {"y", "yes"}:
            logging.info("Aborted overwrite. No file changes made.")
            return
    mode = "w" if overwrite or not file_exists else "a"
    write_header = overwrite or not file_exists
    with output_csv.open(mode, newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        if write_header:
            writer.writerow([
                "image_id",
                "patch_density",
                "font_size",
                "line_spacing",
                "letter_spacing",
                "token_len",
                "text_len",
                "resolution",
                "image_W",
                "image_H",
                "character_width",
                "character_height",
                "model_W",
                "model_H",
                "edit_distance",
            ])
        processed_rows = [[value if value is not None else "Null" for value in row] for row in rows]
        writer.writerows(processed_rows)
    logging.info("Wrote %d rows to %s", len(rows), output_csv)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a CSV linking patch density and token length.")
    parser.add_argument("--density_json", type=Path, default=Path("build_dataset_from_ebook/pride_dataset_square_density_experiment/GroupA/density_record.json"), help="Path to density JSON (supports density_record.json with 'capacity' or ed_pd_len.json with 'patch_density').")
    parser.add_argument("--meta_data", type=Path, default=Path("build_dataset_from_ebook/pride_dataset_square/pride_square_gt.json"), help="Path to metadata JSON containing text_token_len.")
    parser.add_argument("--edit_distance_json", type=Path, default=Path("eval_results/pride_square_density_experiment/GroupA/1024/dpsk_eval_metric.json"), help="Optional path to evaluation JSON containing edit distance per id.")
    parser.add_argument("--image_dir", type=Path, required=True, help="Directory containing source images named by image_id.")
    parser.add_argument("--output_csv", type=Path, default=Path("data_analysis/data4analysis.csv"), help="Output CSV path.")
    parser.add_argument("--font_size", type=int, default=28, help="Font size to record.")
    parser.add_argument("--line_spacing", type=int, default=6, help="Line spacing to record.")
    parser.add_argument("--letter_spacing", type=int, default=0, help="Letter spacing to record.")
    parser.add_argument("--resolution", type=int, default=1024, help="Image resolution to record.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite the output CSV instead of appending.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    font_key = float(args.font_size)
    if font_key not in CHARACTER_SIZE_BY_FONT:
        logging.error("Unsupported font_size %s. Supported sizes: %s", args.font_size, sorted(CHARACTER_SIZE_BY_FONT.keys()))
        return
    character_width, character_height = CHARACTER_SIZE_BY_FONT[font_key]
    density_data = load_json(args.density_json)
    meta_data = load_json(args.meta_data)
    token_map = build_token_length_map(meta_data)
    text_map = build_text_length_map(meta_data)
    edit_distance_map = None
    if args.edit_distance_json is not None:
        if args.edit_distance_json.exists():
            edit_distance_data = load_json(args.edit_distance_json)
            edit_distance_map = build_edit_distance_map(edit_distance_data)
        else:
            logging.warning("edit_distance_json %s does not exist; edit_distance values will be null.", args.edit_distance_json)
    rows = build_rows(
        density_data,
        token_map,
        text_map,
        args.image_dir,
        args.font_size,
        args.line_spacing,
        args.letter_spacing,
        args.resolution,
        character_width,
        character_height,
        edit_distance_map,
    )
    if not rows:
        logging.warning("No rows generated; ensure inputs are valid.")
        return
    write_csv(rows, args.output_csv, args.overwrite)


if __name__ == "__main__":
    main()
