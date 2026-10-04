#!/usr/bin/env bash
set -euo pipefail

DATATYPES=(economics laws letters medicine newspaper pride)
RESOLUTIONS=(512 640 1024 1280)

for dtype in "${DATATYPES[@]}"; do
  for res in "${RESOLUTIONS[@]}"; do
    input_json="eval_results/${dtype}_square/${res}/prepare_json_for_eval.json"
    output_file="eval_results/${dtype}_square/${res}/dpsk_eval_metric.json"
    if [[ ! -f "${input_json}" ]]; then
      echo "Skipping ${dtype} @ ${res}: ${input_json} not found."
      continue
    fi
    echo "Running evaluation for ${dtype} @ ${res}"
    python3 evaluation_dpsk_ocr_predictions.py \
      --json_for_eval "${input_json}" \
      --output_file "${output_file}"
  done
done
