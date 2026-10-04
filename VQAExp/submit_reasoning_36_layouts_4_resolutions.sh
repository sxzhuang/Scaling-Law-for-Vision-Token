#!/bin/bash
# Submit Reasoning VQA jobs for 36 layouts and 4 resolutions in a strict sequence.

set -euo pipefail

cd /slurm/shuxin_zhuang/workspace/Scaling-Law-for-Vision-Token

FONT_SIZES=(20 28 36)
LINE_SPACINGS=(0 6 24 42)
LETTER_SPACINGS=(-1 0 7)
RESOLUTIONS=(512 640 1024 1280)

DATASET_JSON="VQAExp/data/Reasoning/reasoning_cropped_0_25k_600.json"
BASE_OUTPUT_ROOT="VQAExp/reasoning_predictions/Qwen/Qwen2.5-VL-7B-Instruct/Reasoning"
URL="${URL:-http://localhost:50018/v1}"
MODEL="${MODEL:-Qwen/Qwen2.5-VL-7B-Instruct}"
MAX_TOKENS="${MAX_TOKENS:-128}"
IMAGE_URL_MODE="${IMAGE_URL_MODE:-file}"
CONCURRENCY="${CONCURRENCY:-8}"

previous_job_id=""

for font_size in "${FONT_SIZES[@]}"; do
    for line_spacing in "${LINE_SPACINGS[@]}"; do
        for letter_spacing in "${LETTER_SPACINGS[@]}"; do
            layout_name="${font_size}_${line_spacing}_${letter_spacing}"

            for resolution in "${RESOLUTIONS[@]}"; do
                input_path="VQAExp/data/Reasoning/${layout_name}/images_${resolution}"
                output_path="${BASE_OUTPUT_ROOT}/${layout_name}/res_${resolution}"

                if [[ ! -d "$input_path" ]]; then
                    echo "Skip missing input: $input_path"
                    continue
                fi

                export_args="ALL,LAYOUT_NAME=${layout_name},RESOLUTION=${resolution},INPUT_PATH=${input_path},DATASET_JSON=${DATASET_JSON},OUTPUT_PATH=${output_path},URL=${URL},MODEL=${MODEL},MAX_TOKENS=${MAX_TOKENS},IMAGE_URL_MODE=${IMAGE_URL_MODE},CONCURRENCY=${CONCURRENCY}"
                job_name="reasoning_${layout_name}_${resolution}"

                if [[ -z "$previous_job_id" ]]; then
                    job_id=$(sbatch --parsable --job-name "$job_name" --export "$export_args" VQAExp/run_reasoning.sh)
                else
                    export_args="${export_args},PREVIOUS_JOB_ID=${previous_job_id}"
                    job_id=$(sbatch --parsable --job-name "$job_name" --dependency "afterany:${previous_job_id}" --export "$export_args" VQAExp/run_reasoning.sh)
                fi

                echo "Submitted $job_id for $layout_name res_${resolution}"
                previous_job_id="$job_id"
            done
        done
    done
done

echo "Last submitted job: $previous_job_id"

# nohup bash VQAExp/submit_reasoning_36_layouts_4_resolutions.sh > VQAExp/submit_reasoning_36x4.log 2>&1 &
