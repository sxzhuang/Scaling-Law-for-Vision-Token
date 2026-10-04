#!/usr/bin/env bash
# Reuse the existing JSON builder and per-sample OCR metric evaluator.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
CONDA_ENV_NAME="${AAAI_RESUB_CONDA_ENV:-deepseek-ocr}"
DEFAULT_DOMAINS=(novels laws economics medicine newspapers letters)
if [[ "$#" -gt 0 ]]; then
    DOMAINS=("$@")
else
    DOMAINS=("${DEFAULT_DOMAINS[@]}")
fi
RESOLUTIONS=(512 640 1024 1280)
LAYOUT_NAME="16_60_-1"

for domain in "${DOMAINS[@]}"; do
    for resolution in "${RESOLUTIONS[@]}"; do
        image_dir="${REPO_ROOT}/aaai_resub/rendered_datasets/${domain}/${resolution}/${LAYOUT_NAME}/images"
        metadata_path="${REPO_ROOT}/aaai_resub/selected_samples/${domain}/${resolution}/selected_metadata.json"
        prediction_dir="${REPO_ROOT}/aaai_resub/ocr_predictions/${domain}/${resolution}/${LAYOUT_NAME}"
        result_dir="${REPO_ROOT}/aaai_resub/eval_results/${domain}/${resolution}/${LAYOUT_NAME}"
        eval_json="${result_dir}/prepare_json_for_eval.json"
        metric_json="${result_dir}/dpsk_eval_metric.json"
        if [[ ! -d "${image_dir}" || ! -f "${metadata_path}" || ! -d "${prediction_dir}" ]]; then
            echo "Missing render, metadata, or OCR prediction input for ${domain}, R=${resolution}." >&2
            exit 1
        fi
        mkdir -p "${result_dir}"
        conda run -n "${CONDA_ENV_NAME}" python "${REPO_ROOT}/build_dataset_from_ebook/build_eval_json.py" \
            --images_dir "${image_dir}" \
            --metadata_path "${metadata_path}" \
            --predictions_dir "${prediction_dir}" \
            --output_path "${eval_json}"
        conda run -n "${CONDA_ENV_NAME}" python "${REPO_ROOT}/evaluation_dpsk_ocr_predictions.py" \
            --json_for_eval "${eval_json}" \
            --output_file "${metric_json}"
    done
done
