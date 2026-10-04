# Server migration snapshot

This branch includes experiment source, source books, preprocessed book text
JSON, the two 600-example VTCBench input datasets, and the Chinese rendering
font. It excludes predictions, evaluation outputs, fitted parameter results,
images, figures, logs, caches, and model weights. `file_manifest.json` records
the selected files and their checksums.

## Main pipeline

- Text extraction: `build_dataset_from_ebook/build_json_from_txt*.py` and
  `run_extract_data_from_txt.sh`; inputs in `build_dataset_from_ebook/src_book`.
- Rendering: `build_dataset_from_ebook/build_image*.py`,
  `utils/generate_square_images.py`, and `run_generate_dataset*.sh`.
- DeepSeek OCR: `config.py`, `run_dpsk_ocr_eval_batch.py`, `deepseek_ocr.py`,
  `process/`, `deepencoder/`, and `run_ds_ocr_eval_batch.sh`.
- Evaluation: `build_dataset_from_ebook/build_eval_json*.py`,
  `run_prepare_eval_json.sh`, `evaluation_dpsk_ocr_predictions.py`, and
  `run_cal_eval_metric.sh`.
- Capacity-law features and fitting: `data_analysis/data_preprocess.py` and
  `data_analysis/fit_em_scaling_law_shared_parameters_exceptw.py`.
- Qwen OCR / Retrieval / Reasoning: `VQAExp/run_ocr.py`, `run_vqa.py`,
  `run_reasoning.py`, the associated shell wrappers, and
  `opensource_model/configs/Qwen2.5_VL_7B_Instruct.yaml`.
- Task input construction and rendering: `VQAExp/build_cropped_*_dataset.py`,
  `render_*_context_images.py`, and `VQAExp/data/{Retrieval,Reasoning}`.
- Qwen fitting: `VQAExp/scaling_law/build_qwen_*_scaling_csv.py`,
  `fit_qwen_ocr_scaling_law.py`, and `fit_qwen_downstream_scaling_law_scheme2.py`.
- Additional calibration/interventions: `aaai_resub/`; these scripts require
  regenerated result files because existing experiment results are omitted.

## New-server prerequisites

Use Python >=3.10. Typical packages are torch, vllm, transformers, datasets,
openai, Pillow, pymupdf, img2pdf, nltk, jieba, matplotlib, pandas, numpy, scipy,
tqdm, python-Levenshtein and python-docx. DeepSeek custom vLLM integration may
require a compatible vLLM release; these dependencies are not a pinned lockfile.
Install DejaVu Sans for consistent English rendering. NLTK resources and
Hugging Face models must be installed or downloaded separately.

Historical SLURM wrappers preserve their environment and filesystem paths.
Before submitting on a new server, update Conda/venv activation, working
directory, cache paths, partition/resource requests, endpoint URL, and vLLM
`allowed-local-media-path`. Cluster hostnames are replaced with localhost in
the exported snapshot and pinned node directives are removed. A localhost URL
is valid only when client and model service run on the same host. The old
`tests1` privacy symlink and `link/` launcher symlinks are not migrated.

No experiments are run by snapshot preparation. The corrected budgeting
formula and legend-only changes in the workspace are preserved as code;
historical intervention scripts must still be updated before new experiments.
