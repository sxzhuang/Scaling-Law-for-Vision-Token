from pathlib import Path

def delete_png_files(folder: Path) -> None:
    for png_path in folder.glob("*.png"):
        try:
            png_path.unlink()
            print(f"Removed: {png_path.name}")
        except Exception as exc:
            print(f"Failed to remove {png_path.name}: {exc}")


def main() -> None:
    folder = Path("craft_dataset/image")
    if not folder.exists():
        raise FileNotFoundError(f"Folder not found: {folder}")
    delete_png_files(folder)


if __name__ == "__main__":
    main()
