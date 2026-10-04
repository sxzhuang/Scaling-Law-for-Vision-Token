#!/bin/bash

set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
JOB_SCRIPT="${SCRIPT_DIR}/run_ds_ocr_eval_batch.sh"

FONT_SIZES=(28 36)
LINE_SPACINGS=(0 6 24 42)
LETTER_SPACINGS=(-1 0 7)
RESOLUTIONS=(512 640 1024 1280)

MAX_RUNNING=10
SUBMIT_INTERVAL=10
QUERY_INTERVAL=300
BATCH_SIZE="${BATCH_SIZE:-200}"

mkdir -p "${REPO_ROOT}/SLURM_Logs"
cd "${REPO_ROOT}" || exit 1

if [[ ! -f "${JOB_SCRIPT}" ]]; then
    echo "ERROR: job script not found: ${JOB_SCRIPT}"
    exit 1
fi

CONFIGS=()
for font_size in "${FONT_SIZES[@]}"; do
    for line_spacing in "${LINE_SPACINGS[@]}"; do
        for letter_spacing in "${LETTER_SPACINGS[@]}"; do
            for resolution in "${RESOLUTIONS[@]}"; do
                CONFIGS+=("${font_size}|${line_spacing}|${letter_spacing}|${resolution}")
            done
        done
    done
done

ACTIVE_JOB_IDS=()
ACTIVE_LABELS=()
NEXT_INDEX=0
TOTAL=${#CONFIGS[@]}

submit_next_job() {
    local config="${CONFIGS[$NEXT_INDEX]}"
    local font_size line_spacing letter_spacing resolution
    IFS="|" read -r font_size line_spacing letter_spacing resolution <<< "${config}"

    local dataset_name="${font_size}_${line_spacing}_${letter_spacing}"
    local input_path="appendix_exp/jianai_dataset_square/${dataset_name}/images/"
    local output_path="appendix_exp/ocr_predictions/jianai_dataset_square/${dataset_name}/${resolution}/"
    local label="${dataset_name}_${resolution}"

    if [[ ! -d "${input_path}" ]]; then
        echo "WARN: input path does not exist yet: ${input_path}"
    fi

    local job_id
    job_id=$(sbatch --parsable --export=ALL,BASE_SIZE="${resolution}",IMAGE_SIZE="${resolution}",INPUT_PATH="${input_path}",OUTPUT_PATH="${output_path}",BATCH_SIZE="${BATCH_SIZE}" "${JOB_SCRIPT}")
    ACTIVE_JOB_IDS+=("${job_id}")
    ACTIVE_LABELS+=("${label}")
    NEXT_INDEX=$((NEXT_INDEX + 1))

    echo "Submitted ${label}: job_id=${job_id} (${NEXT_INDEX}/${TOTAL})"
}

refresh_active_jobs() {
    local new_job_ids=()
    local new_labels=()
    local i job_id label

    for i in "${!ACTIVE_JOB_IDS[@]}"; do
        job_id="${ACTIVE_JOB_IDS[$i]}"
        label="${ACTIVE_LABELS[$i]}"
        if squeue -h -j "${job_id}" >/dev/null 2>&1 && [[ -n "$(squeue -h -j "${job_id}")" ]]; then
            new_job_ids+=("${job_id}")
            new_labels+=("${label}")
        else
            echo "Completed or left queue: ${label}, job_id=${job_id}"
        fi
    done

    ACTIVE_JOB_IDS=("${new_job_ids[@]}")
    ACTIVE_LABELS=("${new_labels[@]}")
}

while [[ ${NEXT_INDEX} -lt ${TOTAL} && ${#ACTIVE_JOB_IDS[@]} -lt ${MAX_RUNNING} ]]; do
    submit_next_job
    if [[ ${NEXT_INDEX} -lt ${TOTAL} && ${#ACTIVE_JOB_IDS[@]} -lt ${MAX_RUNNING} ]]; then
        sleep "${SUBMIT_INTERVAL}"
    fi
done

while [[ ${NEXT_INDEX} -lt ${TOTAL} || ${#ACTIVE_JOB_IDS[@]} -gt 0 ]]; do
    echo "Waiting ${QUERY_INTERVAL}s before checking active jobs. active=${#ACTIVE_JOB_IDS[@]}, submitted=${NEXT_INDEX}/${TOTAL}"
    sleep "${QUERY_INTERVAL}"
    refresh_active_jobs

    while [[ ${NEXT_INDEX} -lt ${TOTAL} && ${#ACTIVE_JOB_IDS[@]} -lt ${MAX_RUNNING} ]]; do
        submit_next_job
        if [[ ${NEXT_INDEX} -lt ${TOTAL} && ${#ACTIVE_JOB_IDS[@]} -lt ${MAX_RUNNING} ]]; then
            sleep "${SUBMIT_INTERVAL}"
        fi
    done
done

echo "All jobs submitted and completed."

# nohup bash appendix_exp/submit_ds_ocr_eval_grid.sh > appendix_exp/submit_ds_ocr_eval_grid.log 2>&1 &