"""Copy experiment source and input texts into an isolated Git checkout."""

import argparse
import ast
import hashlib
import json
import re
import shutil
from pathlib import Path


SOURCE_ROOTS = {"build_dataset_from_ebook", "build_dataset_from_OmniDocBench", "deepencoder", "process", "utils", "data_analysis", "scatter_plot", "VQAExp", "aaai_resub", "appendix_exp", "opensource_model", "focus_benchmark_test"}
SOURCE_SUFFIXES = {".py", ".sh", ".sbatch", ".yaml", ".yml"}
INPUT_ROOTS = ("build_dataset_from_ebook/src_book/", "VQAExp/data/Retrieval/", "VQAExp/data/Reasoning/", "build_dataset_from_ebook/fonts/", "aaai_resub/configs/")
SKIP_DIRS = {".git", "__pycache__", ".venv", "SLURM_Logs", "ocr_predictions", "eval_results", "evaluation", "evaluation_stage", "vqa_predictions", "reasoning_predictions", "rendered", "rendered_datasets", "mother_images", "images", "figures", "logs", "Fitting", "em_fiting_res", "link"}


def selected(path: Path) -> bool:
    relative = path.as_posix()
    if relative == "aaai_resub/migration/README.md":
        return True
    if path.is_symlink() or any(part in SKIP_DIRS for part in path.parts):
        return False
    if len(path.parts) == 1:
        return path.suffix in SOURCE_SUFFIXES or path.name in {"README.md", ".gitattributes"}
    if relative.startswith(INPUT_ROOTS):
        return path.suffix in {".txt", ".json", ".jsonl", ".otf", ".ttf"}
    return path.parts[0] in SOURCE_ROOTS and path.suffix in SOURCE_SUFFIXES


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True, help="Existing experiment workspace.")
    parser.add_argument("--destination", type=Path, required=True, help="Isolated Git checkout created for migration.")
    args = parser.parse_args()
    source = args.source.resolve()
    destination = args.destination.resolve()
    if source == destination or not (destination / ".git").exists():
        raise ValueError("Destination must be a separate Git checkout.")
    paths = []
    # Restrict traversal to source directories, pruning generated artifact trees.
    import os
    for root, dirs, files in os.walk(source, followlinks=False):
        relative_root = Path(root).relative_to(source)
        dirs[:] = [name for name in dirs if name not in SKIP_DIRS and not (Path(root) / name).is_symlink() and (relative_root != Path('.') or name in SOURCE_ROOTS)]
        for name in files:
            relative = relative_root / name
            if selected(relative) and not (source / relative).is_symlink():
                paths.append(relative)
    allowed = {path.as_posix() for path in paths}
    for root, dirs, files in os.walk(destination):
        dirs[:] = [name for name in dirs if name != ".git"]
        for name in files:
            path = Path(root) / name
            if path.relative_to(destination).as_posix() not in allowed:
                path.unlink()
    manifest = []
    for relative in sorted(paths):
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source / relative, target)
        # Remove hard-coded cluster hostnames from the exported scripts only.
        if relative.suffix in SOURCE_SUFFIXES:
            text = target.read_text(encoding="utf-8")
            text = re.sub(r"compute-(?:a100|a30)-\d+", "localhost", text)
            text = re.sub(r"(?m)^#SBATCH\s+(?:-w\s+|--nodelist[= ]).*\n", "", text)
            target.write_text(text, encoding="utf-8")
            if relative.suffix == ".py":
                ast.parse(text, filename=str(relative))
        manifest.append({"path": relative.as_posix(), "bytes": target.stat().st_size, "sha256": hashlib.sha256(target.read_bytes()).hexdigest()})
    output = destination / "aaai_resub/migration/file_manifest.json"
    output.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"files": len(manifest), "bytes": sum(item["bytes"] for item in manifest), "destination": str(destination)}, indent=2))


if __name__ == "__main__":
    main()
