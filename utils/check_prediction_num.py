"""Utility to count OCR prediction files per dataset.

Iterates over directories in ``ocr_predictions`` whose names end with ``_square``
and reports:

1. The number of files inside each immediate subdirectory.
2. The number of files under the matching ``build_dataset_from_ebook/<name>/images`` folder.
"""

from __future__ import annotations

from pathlib import Path


def count_files_in_dir(directory: Path) -> int:
    """Count direct child files in a directory."""
    if not directory.exists() or not directory.is_dir():
        return 0
    return sum(1 for entry in directory.iterdir() if entry.is_file())


def main() -> None:
    root = Path("ocr_predictions")
    if not root.exists():
        print("[WARN] ocr_predictions directory not found.")
        return

    for dataset_dir in sorted(root.iterdir()):
        if not dataset_dir.is_dir() or not dataset_dir.name.endswith("_square"):
            continue

        dataset_name = dataset_dir.name
        print(f"=== Dataset: {dataset_name} ===")

        subdirs = sorted(d for d in dataset_dir.iterdir() if d.is_dir())
        if not subdirs:
            print("  [WARN] No subdirectories found.")
        else:
            for subdir in subdirs:
                file_count = count_files_in_dir(subdir)
                print(f"  {subdir.name}: {file_count} files")

        images_dir = Path("build_dataset_from_ebook") / dataset_name / "images"
        images_count = count_files_in_dir(images_dir)
        print(f"  Images dir: {images_dir} -> {images_count} files\n")


if __name__ == "__main__":
    main()
