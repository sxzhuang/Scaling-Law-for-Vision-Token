#!/bin/bash
set -euo pipefail

ROOT_DIR="eval_results/pride_square_density_experiment"
SOURCE_JSON="build_dataset_from_ebook/pride_dataset_square/pride_square_gt.json"

# for square_dir in "${ROOT_DIR}"/*_square; do
for square_dir in "${ROOT_DIR}"/[0-9]*; do
    if [[ ! -d "${square_dir}" ]]; then
        continue
    fi
    for res_dir in "${square_dir}"/*; do
        if [[ ! -d "${res_dir}" ]]; then
            continue
        fi
        json_path="${res_dir}/dpsk_eval_metric.json"
        if [[ -f "${json_path}" ]]; then
            python utils/merge_text_len_from_source.py --source_json "${SOURCE_JSON}" --modify_json "${json_path}"
        fi
    done
done
