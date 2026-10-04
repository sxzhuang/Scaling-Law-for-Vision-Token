#!/usr/bin/env bash
set -euo pipefail

DATATYPES=(economics laws letters medicine newspaper pride)
RESOLUTIONS=(512 640 1024 1280)

for dtype in "${DATATYPES[@]}"; do
  images_dir="build_dataset_from_ebook/${dtype}_dataset_square/images"
  metadata_path="build_dataset_from_ebook/${dtype}_dataset_square/${dtype}_square_gt.json"
  for res in "${RESOLUTIONS[@]}"; do
    predictions_dir="ocr_predictions/${dtype}_dataset_square/${res}"
    output_path="eval_results/${dtype}_square/${res}/prepare_json_for_eval.json"
    mkdir -p "$(dirname "${output_path}")"
    echo "Running build_eval_json for ${dtype} @ ${res}"
    python3 build_dataset_from_ebook/build_eval_json.py \
      --images_dir "${images_dir}" \
      --metadata_path "${metadata_path}" \
      --predictions_dir "${predictions_dir}" \
      --output_path "${output_path}"
  done
done
