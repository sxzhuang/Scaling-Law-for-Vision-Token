#!/usr/bin/env bash
# Run the existing DeepSeek-OCR batch evaluator on the rerendered intervention images.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
CONDA_ENV_NAME="${AAAI_RESUB_CONDA_ENV:-deepseek-ocr}"
BATCH_SIZE="${AAAI_RESUB_BATCH_SIZE:-300}"
DOMAINS=(novels laws economics medicine newspapers letters)
RESOLUTIONS=(512 640 1024 1280)
LAYOUT_NAME="16_60_-1"

for domain in "${DOMAINS[@]}"; do
    for resolution in "${RESOLUTIONS[@]}"; do
        image_dir="${REPO_ROOT}/aaai_resub/rendered_datasets/${domain}/${resolution}/${LAYOUT_NAME}/images"
        prediction_dir="${REPO_ROOT}/aaai_resub/ocr_predictions/${domain}/${resolution}/${LAYOUT_NAME}/"
        if [[ ! -d "${image_dir}" ]]; then
            echo "Missing rerendered images: ${image_dir}" >&2
            exit 1
        fi
        conda run -n "${CONDA_ENV_NAME}" python -m run_dpsk_ocr_eval_batch \
            --base_size "${resolution}" \
            --image_size "${resolution}" \
            --input_path "${image_dir}" \
            --output_path "${prediction_dir}" \
            --batch_size "${BATCH_SIZE}"
    done
done
