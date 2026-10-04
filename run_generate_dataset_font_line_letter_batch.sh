#!/bin/bash
set -euo pipefail

METADATA="build_dataset_from_ebook/newspaper_dataset_square/newspaper_square_gt.json"
OUTPUT_ROOT="build_dataset_from_ebook/newspaper_dataset_square_density_experiment"
FONT_SIZES=(20 28 36)
LINE_SPACINGS=(0 6 24 42)

for font_size in "${FONT_SIZES[@]}"; do
  for line_spacing in "${LINE_SPACINGS[@]}"; do
    sbatch run_generate_dataset_font_line_letter.sh \
      --metadata "${METADATA}" \
      --output_root "${OUTPUT_ROOT}" \
      --font_sizes "${font_size}" \
      --line_spacings "${line_spacing}"
  done
done
