from pathlib import Path
from PIL import Image


def convert_png_files(folder: Path) -> None:
    for png_path in folder.glob("*.png"):
        jpg_path = png_path.with_suffix(".jpg")
        try:
            with Image.open(png_path) as img:
                if img.mode in ("RGBA", "LA"):
                    background = Image.new("RGB", img.size, (255, 255, 255))
                    background.paste(img, mask=img.split()[-1])
                    img = background
                else:
                    img = img.convert("RGB")
                img.save(jpg_path, format="JPEG", quality=95)
            png_path.unlink()
            print(f"Converted {png_path.name} -> {jpg_path.name}")
        except Exception as exc:
            print(f"Failed to process {png_path.name}: {exc}")


def main() -> None:
    folder = Path("focus_benchmark_test/focus_benchmark_test/demo_test")
    if not folder.exists():
        raise FileNotFoundError(f"Folder not found: {folder}")
    convert_png_files(folder)


if __name__ == "__main__":
    main()
