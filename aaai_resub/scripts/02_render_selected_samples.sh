#!/usr/bin/env bash
# Render the selected Figure 1 contents with the law-guided 16_60_-1 layout.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
CONDA_ENV_NAME="${AAAI_RESUB_CONDA_ENV:-deepseek-ocr}"
DOMAINS=(novels laws economics medicine newspapers letters)
RESOLUTIONS=(512 640 1024 1280)
FONT_SIZE=16
LINE_SPACING=60
LETTER_SPACING=-1

for domain in "${DOMAINS[@]}"; do
    for resolution in "${RESOLUTIONS[@]}"; do
        metadata_path="${REPO_ROOT}/aaai_resub/selected_samples/${domain}/${resolution}/selected_metadata.json"
        output_root="${REPO_ROOT}/aaai_resub/rendered_datasets/${domain}/${resolution}"
        if [[ ! -f "${metadata_path}" ]]; then
            echo "Missing selection metadata: ${metadata_path}" >&2
            exit 1
        fi
        conda run -n "${CONDA_ENV_NAME}" python "${REPO_ROOT}/build_dataset_from_ebook/build_image_square_font_line_letter.py" \
            --metadata "${metadata_path}" \
            --output_root "${output_root}" \
            --font_sizes "${FONT_SIZE}" \
            --line_spacings "${LINE_SPACING}" \
            --letter_spacings "${LETTER_SPACING}" \
            --padding 20 \
            --dpi 300 \
            --ratio_min 0.9 \
            --ratio_max 1.1
    done
done
