#!/bin/bash

set -euo pipefail

FONT_SIZES=(20 28 36)
LINE_SPACINGS=(0 6 24 42)
LETTER_SPACINGS=(-1 0 7)
FONT_PATH="${FONT_PATH:-build_dataset_from_ebook/fonts/SourceHanSansSC-Regular.otf}"
JOB_SCRIPT="appendix_exp/run_build_image_square_from_sampled_text_dataset.sh"

mkdir -p SLURM_Logs

for font_size in "${FONT_SIZES[@]}"; do
    for line_spacing in "${LINE_SPACINGS[@]}"; do
        for letter_spacing in "${LETTER_SPACINGS[@]}"; do
            echo "Submitting font_size=${font_size}, line_spacing=${line_spacing}, letter_spacing=${letter_spacing}"
            FONT_SIZE="${font_size}" LINE_SPACING="${line_spacing}" LETTER_SPACING="${letter_spacing}" FONT_PATH="${FONT_PATH}" sbatch "${JOB_SCRIPT}"
        done
    done
done

echo "Submitted all ${#FONT_SIZES[@]} x ${#LINE_SPACINGS[@]} x ${#LETTER_SPACINGS[@]} layout jobs."

# nohup bash appendix_exp/submit_build_image_square_grid.sh > appendix_exp/submit_build_image_square_grid.log 2>&1 &