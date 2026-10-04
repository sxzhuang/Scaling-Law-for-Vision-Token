#!/bin/bash

set -u

TARGET_COUNT="${1:-48}"
LOG_FILE="${2:-appendix_exp/submit_ds_ocr_eval_grid.log}"
SUBMIT_PATTERN="${3:-submit_ds_ocr_eval_grid.sh}"
CHECK_INTERVAL="${CHECK_INTERVAL:-10}"

if [[ ! "${TARGET_COUNT}" =~ ^[0-9]+$ ]]; then
    echo "ERROR: TARGET_COUNT must be an integer, got: ${TARGET_COUNT}"
    exit 1
fi

echo "Monitoring ${LOG_FILE}; will stop ${SUBMIT_PATTERN} after submitted >= ${TARGET_COUNT}."

while true; do
    if [[ ! -f "${LOG_FILE}" ]]; then
        echo "Waiting for log file: ${LOG_FILE}"
        sleep "${CHECK_INTERVAL}"
        continue
    fi

    submitted_count=$(grep -oE '\([0-9]+/[0-9]+\)' "${LOG_FILE}" | tail -1 | sed -E 's/^\(([0-9]+)\/[0-9]+\)$/\1/')
    submitted_count="${submitted_count:-0}"

    echo "Current submitted count: ${submitted_count}/${TARGET_COUNT}"

    if (( submitted_count >= TARGET_COUNT )); then
        pids=$(pgrep -f "${SUBMIT_PATTERN}" || true)
        own_pid="$$"
        stopped=0

        for pid in ${pids}; do
            if [[ "${pid}" == "${own_pid}" ]]; then
                continue
            fi
            cmdline=$(ps -p "${pid}" -o args= 2>/dev/null || true)
            if [[ "${cmdline}" == *"${SUBMIT_PATTERN}"* && "${cmdline}" != *"stop_submit_after_n.sh"* ]]; then
                echo "Stopping submit process pid=${pid}: ${cmdline}"
                kill "${pid}"
                stopped=1
            fi
        done

        if (( stopped == 0 )); then
            echo "No matching submit process found. It may have already stopped."
        fi
        exit 0
    fi

    sleep "${CHECK_INTERVAL}"
done
