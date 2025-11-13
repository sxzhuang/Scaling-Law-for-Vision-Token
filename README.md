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
