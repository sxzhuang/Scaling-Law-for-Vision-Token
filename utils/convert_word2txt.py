"""Convert DOCX documents into plain text files."""

import argparse
import logging
from pathlib import Path

from docx import Document


def convert_docx_to_txt(docx_path: Path) -> Path:
    """Convert a DOCX file to a TXT file with the same name."""
    if not docx_path.exists():
        raise FileNotFoundError(f"DOCX file not found: {docx_path}")
    if docx_path.suffix.lower() != ".docx":
        raise ValueError("Input file must have a .docx extension.")

    document = Document(docx_path)
    lines = [paragraph.text for paragraph in document.paragraphs]
    txt_content = "\n".join(lines).strip() + "\n" if lines else ""

    txt_path = docx_path.with_suffix(".txt")
    txt_path.write_text(txt_content, encoding="utf-8")
    logging.info("Saved text to %s", txt_path)
    return txt_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert DOCX files to TXT.")
    parser.add_argument("--docx_path", type=Path, required=True, help="Path to the input DOCX file.")
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = parse_args()
    convert_docx_to_txt(args.docx_path)


if __name__ == "__main__":
    main()
