#!/bin/bash
# Submit OCR jobs for 36 layouts and 4 resolutions across fixed endpoints.
# At most one active SLURM job is kept per endpoint. When a job finishes,
# this driver submits the next task to the freed endpoint.

set -euo pipefail

cd /slurm/shuxin_zhuang/workspace/Scaling-Law-for-Vision-Token

# FONT_SIZES=(20 28 36)
# LINE_SPACINGS=(0 6 24 42)
FONT_SIZES=(36)
LINE_SPACINGS=(24 42)
LETTER_SPACINGS=(-1 0 7)
RESOLUTIONS=(512 640 1024 1280)
URLS=("http://localhost:50016/v1" "http://localhost:50017/v1" "http://localhost:50019/v1" "http://localhost:50018/v1" "http://localhost:50020/v1" "http://localhost:50021/v1")

BASE_OUTPUT_ROOT="VQAExp/ocr_predictions/Qwen/Qwen2.5-VL-7B-Instruct/OCR"
METADATA_JSON="${METADATA_JSON:-VQAExp/data/Retrieval/retrieval_cropped_0_25k_600.json}"
MODEL="${MODEL:-Qwen/Qwen2.5-VL-7B-Instruct}"
MAX_TOKENS="${MAX_TOKENS:-8192}"
MAX_TOKENS_RATIO="${MAX_TOKENS_RATIO:-1.2}"
MAX_TOKENS_MARGIN="${MAX_TOKENS_MARGIN:-32}"
MIN_MAX_TOKENS="${MIN_MAX_TOKENS:-64}"
IMAGE_URL_MODE="${IMAGE_URL_MODE:-file}"
CONCURRENCY="${CONCURRENCY:-8}"
POLL_INTERVAL="${POLL_INTERVAL:-10}"

TASK_LAYOUTS=()
TASK_RESOLUTIONS=()
TASK_INPUTS=()
TASK_OUTPUTS=()

for font_size in "${FONT_SIZES[@]}"; do
    for line_spacing in "${LINE_SPACINGS[@]}"; do
        for letter_spacing in "${LETTER_SPACINGS[@]}"; do
            layout_name="${font_size}_${line_spacing}_${letter_spacing}"

            for resolution in "${RESOLUTIONS[@]}"; do
                input_path="VQAExp/data/Retrieval/${layout_name}/images_${resolution}"
                output_path="${BASE_OUTPUT_ROOT}/${layout_name}/res_${resolution}"

                if [[ ! -d "$input_path" ]]; then
                    echo "Skip missing input: $input_path"
                    continue
                fi

                TASK_LAYOUTS+=("$layout_name")
                TASK_RESOLUTIONS+=("$resolution")
                TASK_INPUTS+=("$input_path")
                TASK_OUTPUTS+=("$output_path")
            done
        done
    done
done

TOTAL_TASKS=${#TASK_LAYOUTS[@]}
NEXT_TASK_INDEX=0
ACTIVE_JOB_IDS=()
ACTIVE_LABELS=()
for _ in "${URLS[@]}"; do
    ACTIVE_JOB_IDS+=("")
    ACTIVE_LABELS+=("")
done

job_in_queue() {
    local job_id="$1"
    [[ -n "$(squeue -h -j "$job_id" 2>/dev/null || true)" ]]
}

job_state() {
    local job_id="$1"
    sacct -j "$job_id" --format=JobIDRaw,State,ExitCode -n -P | awk -F'|' '$1 == job_id {print $2 " " $3; exit}' job_id="$job_id" || true
}

submit_next_for_slot() {
    local slot="$1"
    local task_index="$NEXT_TASK_INDEX"
    local layout_name="${TASK_LAYOUTS[$task_index]}"
    local resolution="${TASK_RESOLUTIONS[$task_index]}"
    local input_path="${TASK_INPUTS[$task_index]}"
    local output_path="${TASK_OUTPUTS[$task_index]}"
    local url="${URLS[$slot]}"
    local export_args="ALL,LAYOUT_NAME=${layout_name},RESOLUTION=${resolution},INPUT_PATH=${input_path},OUTPUT_PATH=${output_path},METADATA_JSON=${METADATA_JSON},URL=${url},MODEL=${MODEL},MAX_TOKENS=${MAX_TOKENS},MAX_TOKENS_RATIO=${MAX_TOKENS_RATIO},MAX_TOKENS_MARGIN=${MAX_TOKENS_MARGIN},MIN_MAX_TOKENS=${MIN_MAX_TOKENS},IMAGE_URL_MODE=${IMAGE_URL_MODE},CONCURRENCY=${CONCURRENCY}"
    local job_name="ocr_${layout_name}_${resolution}"
    local job_id

    job_id=$(sbatch --parsable --job-name "$job_name" --export "$export_args" VQAExp/run_ocr.sh)
    ACTIVE_JOB_IDS[$slot]="$job_id"
    ACTIVE_LABELS[$slot]="${layout_name} res_${resolution}"
    echo "Submitted $job_id for ${layout_name} res_${resolution} on slot_${slot} $url"
    NEXT_TASK_INDEX=$((NEXT_TASK_INDEX + 1))
}

active_count() {
    local count=0
    for job_id in "${ACTIVE_JOB_IDS[@]}"; do
        if [[ -n "$job_id" ]]; then
            count=$((count + 1))
        fi
    done
    echo "$count"
}

echo "Total tasks to schedule: $TOTAL_TASKS"
echo "Endpoint slots: ${#URLS[@]}"
echo "Poll interval: ${POLL_INTERVAL}s"

while true; do
    for slot in "${!URLS[@]}"; do
        job_id="${ACTIVE_JOB_IDS[$slot]}"
        if [[ -n "$job_id" ]] && ! job_in_queue "$job_id"; then
            state="$(job_state "$job_id")"
            if [[ -z "$state" ]]; then
                state="UNKNOWN"
            fi
            if [[ "$state" != COMPLETED* ]]; then
                echo "Job may have failed: job_id=$job_id label=${ACTIVE_LABELS[$slot]} state=$state"
            else
                echo "Job completed: job_id=$job_id label=${ACTIVE_LABELS[$slot]} state=$state"
            fi
            ACTIVE_JOB_IDS[$slot]=""
            ACTIVE_LABELS[$slot]=""
        fi
    done

    for slot in "${!URLS[@]}"; do
        if [[ -z "${ACTIVE_JOB_IDS[$slot]}" && "$NEXT_TASK_INDEX" -lt "$TOTAL_TASKS" ]]; then
            submit_next_for_slot "$slot"
        fi
    done

    if [[ "$NEXT_TASK_INDEX" -ge "$TOTAL_TASKS" && "$(active_count)" -eq 0 ]]; then
        break
    fi

    sleep "$POLL_INTERVAL"
done

echo "All scheduled OCR jobs have finished."

# nohup bash VQAExp/submit_ocr_36_layouts_4_resolutions.sh > VQAExp/submit_ocr_36x4.log 2>&1 &
