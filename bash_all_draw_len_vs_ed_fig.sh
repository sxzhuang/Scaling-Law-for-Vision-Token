#!/usr/bin/env bash
set -euo pipefail

DATATYPES=(economics laws letters medicine newspaper pride)

for dtype in "${DATATYPES[@]}"; do
  input_dir="eval_results/${dtype}_square"
  output="eval_results/${dtype}_square/exp2_tokenlen_vs_editdist.png"
  if [[ ! -d "${input_dir}" ]]; then
    echo "Skipping ${dtype}: ${input_dir} not found."
    continue
  fi
  mkdir -p "$(dirname "${output}")"
  echo "Running evaluation for ${dtype}"
  python3 scatter_plot/exp2_tokenlen_vs_editdistance_nearly_square.py \
    --input_dir "${input_dir}" \
    --output "${output}"
done
