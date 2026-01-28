## Workflow Overview

This repository’s end-to-end preparation and evaluation pipeline that must be executed in a fixed order. Each stage consumes the artifacts produced by the previous one, so run them sequentially from the repository root.

### 1. `run_extract_data_from_txt.sh`
- **Invokes** `build_dataset_from_ebook/build_json_from_txt.py`.
- **Input**: raw novel text at `build_dataset_from_ebook/src_book/pride-and-prejudice.txt`.
- **Output**: segmented dataset JSON `build_dataset_from_ebook/src_book/pride_and_prejudice_dataset.json`, where each record contains the book name and 4-block splits (controlled by `--blocks 4`).

### 2. `run_generate_dataset_from_json.sh`
- **Invokes** `build_dataset_from_ebook/build_images_from_json.py`.
- **Input**: the JSON generated in step 1 (`pride_and_prejudice_dataset.json`).
- **Output**: rendered evaluation corpus under `build_dataset_from_ebook/pride_dataset/`, including:
  - `images/` – block-level page renderings (JPGs).
  - `pride_prejudice_blocks_gt.json` – metadata with `image`/`gt` pairs and layout annotations.

### 3. `run_add_text_len.sh`
- **Invokes** `utils/compute_text_token_lengths.py`.
- **Input**: `build_dataset_from_ebook/pride_dataset/pride_prejudice_blocks_gt.json` from step 2.
- **Output**: the same JSON rewritten in place, now enriched with a `text_token_len` field for every record (tokenized with the Qwen tokenizer).

### 4. `run_ds_ocr_eval_batch.sh`
- **Invokes** the main OCR engine via `python -m run_dpsk_ocr_eval_batch`.
- **Input**: relies on paths configured in `config.py` (typically `INPUT_PATH=build_dataset_from_ebook/pride_dataset/images` and `OUTPUT_PATH=ocr_predictions/pride_dataset/640`).
- **Output**: DeepSeek OCR predictions per image, stored as Markdown files (`*.md` and `*_det.md`) inside `ocr_predictions/pride_dataset/640/`.

### 5. `run_prepare_eval_json.sh`
- **Invokes** `build_dataset_from_ebook/build_eval_json.py`.
- **Input**:
  - Rendered images: `build_dataset_from_ebook/pride_dataset/images`.
  - Ground-truth metadata: `build_dataset_from_ebook/pride_dataset/pride_prejudice_blocks_gt.json`.
  - OCR predictions: `ocr_predictions/pride_dataset/640`.
- **Output**: consolidated evaluation JSON (e.g., `pride/640/prepare_json_for_eval.json`) that stores `{id, label, answer, text_token_len}` per sample and aligns predictions with ground truth.

### 6. `run_cal_eval_metric.sh`
- **Invokes** `evaluation_dpsk_ocr_predictions.py`.
- **Input**: the evaluation JSON produced in step 5 (`--json_for_eval pride/640/prepare_json_for_eval.json` or your chosen file).
- **Output**: per-sample metric dump `pride/640/dpsk_eval_metric.json`, containing BLEU, F-measure, precision, recall, edit distance, and `text_token_len` for every record. The script logs progress after each quarter of the dataset and checkpoints intermediate metric batches.

> **Note**: All `.sh` files are SLURM job descriptions. Before submitting them, ensure the designated compute node, conda environment (`deepseek-ocr`), and `config.py` paths match your cluster setup.


---

### Selecting High-Variance Samples

Use `utils/selected_data.py` to extract problematic samples (e.g., high edit distance and long text length) and optionally copy their corresponding images for focused debugging.

**CLI overview**

- Filters IDs from a metrics JSON (default `eval_results/pride_square/1024/dpsk_eval_metric.json`) by `edit_dist` and `text_token_len` ranges.
- Copies matching images from `--image_folder` to `--output_folder` while saving the ID list (defaults to `<output_folder>/../selected_ids.json`).
- Can also be used as a library via `select_ids()` and `copy_selected_images()`.

**Example**

```bash
python utils/selected_data.py \
  --metrics_json eval_results/pride_square/1024/dpsk_eval_metric.json \
  --edit_dist_threshold 0.19 \
  --min_token_len 1500 \
  --max_token_len 4050 \
  --image_folder build_dataset_from_ebook/pride_dataset_square/images \
  --output_folder build_dataset_from_ebook/pride_dataset_square_large_variance/selected_from1024/images
```

The script logs how many IDs were selected, copies the corresponding images, and writes the selected IDs to `selected_ids.json`. Adjust thresholds and paths as needed for other datasets.

### Padding + Shift Augmentation

`utils/paddling_shifted_inputs.py` creates a grid of shifted variants for every image in a folder by pasting a resized version (`resolution - 16`) onto a white canvas of size `resolution × resolution` at multiple offsets.

**Usage**

```bash
python utils/paddling_shifted_inputs.py \
  --folder build_dataset_from_ebook/pride_dataset_square_large_variance/selected_from640/images \
  --resolution 640 \
  --stride 2
```

Key notes:

- The script scans `.jpg/.jpeg/.png` files under `--folder`, resizes them, and produces all shifts within a 16-pixel border using the specified `stride`.
- Outputs are written to a sibling directory named `<folder>_paddling_shift`, keeping the original images untouched.
- Logs warn if no images are found and detail how many variants are produced per batch.

### Square Layout Helpers

`utils/generate_square_images.py` exposes three helpers for square-ish text renders:
- `render_text_assets(text, font_size, line_spacing, letter_spacing, font_path=None, pdf_dpi=300, padding=20, ratio_min=0.9, ratio_max=1.1, max_layout_attempts=3)`: fits text into a near-square layout, returning a dict with `image_shape` `(width, height)`, the PIL `image`, raw `pdf` bytes, and `character_rect` for a typical glyph.
- `save_image_and_pdf(image, pdf, image_path, pdf_path, dpi=300)`: writes the JPEG and PDF pair to disk (PDF bytes are used if provided; otherwise the PDF is rendered from the image) using the same quality/DPI conventions as the dataset builders.
- `estimate_information_density(image_size, character_rect, patch_size=16, resolution=1024)`: estimates how many single characters fit in a `patch_size × patch_size` region after scaling the image to `resolution`, returning a float rounded to five decimals.

**Example**

```python
from pathlib import Path
from utils.generate_square_images import render_text_assets, save_image_and_pdf, estimate_information_density

result = render_text_assets(text="eee", font_size=28, line_spacing=6, letter_spacing=0)
capacity = estimate_information_density(result["image_shape"], result["character_rect"], patch_size=16, resolution=1024)
print("Image shape:", result["image_shape"])
print("Characters per 16x16 patch:", capacity)
save_image_and_pdf(result["image"], result["pdf"], image_path=Path("sample_square_layout.jpg"), pdf_path=Path("sample_square_layout.pdf"))
```

### Variance Visualization

`scatter_plot/integrate_variance_points.py` overlays three layers of information to show how high-variance (padded/shifted) samples change edit distance:
- **Baseline**: faint background cloud of all original samples.
- **Before shift**: highlighted rose-red markers for overlapping IDs that triggered the variance selection.
- **After shift**: highlighted teal triangles at the corresponding shifted outputs, with pastel lines linking each pair to show the magnitude/direction of change relative to the hard-wall token length.

Run it like:

```bash
python scatter_plot/integrate_variance_points.py \
  --org_metric eval_results/pride_square/640/dpsk_eval_metric.json \
  --high_variance_metric eval_results/pride_square_large_variance/selected_from640/dpsk_eval_metric.json \
  --output_path eval_results/pride_large_variance_overlay_640.png
```

Adjust `--hard_wall` and input paths to target other datasets or resolutions; the resulting PNG provides an immediate visual read on which regions remain recoverable versus saturated.
