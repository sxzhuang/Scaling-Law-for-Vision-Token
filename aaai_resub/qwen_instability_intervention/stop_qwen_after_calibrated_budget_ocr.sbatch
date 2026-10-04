#!/bin/bash
#SBATCH --job-name=test
#SBATCH --output=/slurm/shuxin_zhuang/tests1/SLURM_Logs/%j.out
#SBATCH --error=/slurm/shuxin_zhuang/tests1/SLURM_Logs/%j.err
#SBATCH --chdir=/slurm/shuxin_zhuang/tests1
#SBATCH --partition=a100,a30
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=4GB

set -euo pipefail

cd /slurm/shuxin_zhuang/tests1
source /home/sxzhuangch/miniconda3/etc/profile.d/conda.sh
conda activate deepseek-ocr

OCR_ARRAY_ID="${OCR_ARRAY_ID:-83033}"
SERVICE_JOB_IDS=(83024 83025)

mapfile -t OCR_STATES < <(sacct -j "$OCR_ARRAY_ID" --format=JobID,State -n -P | awk -F'|' '$1 ~ /^[0-9]+_[0-9]+$/ {print $2}')
if [[ "${#OCR_STATES[@]}" -ne 2 ]]; then
    echo "Expected two OCR array-task states for ${OCR_ARRAY_ID}, found ${#OCR_STATES[@]}" >&2
    exit 1
fi
for STATE in "${OCR_STATES[@]}"; do
    if [[ "$STATE" != "COMPLETED" ]]; then
        echo "OCR array ${OCR_ARRAY_ID} did not complete successfully: state=${STATE}" >&2
        exit 1
    fi
done

python aaai_resub/qwen_instability_intervention/verify_calibrated_budget_ocr.py

ACTIVE_SERVICE_JOB_IDS=()
for SERVICE_JOB_ID in "${SERVICE_JOB_IDS[@]}"; do
    if squeue -h -j "$SERVICE_JOB_ID" -o "%T" | grep -Eq 'RUNNING|PENDING|CONFIGURING'; then
        ACTIVE_SERVICE_JOB_IDS+=("$SERVICE_JOB_ID")
    fi
done
if [[ "${#ACTIVE_SERVICE_JOB_IDS[@]}" -gt 0 ]]; then
    scancel "${ACTIVE_SERVICE_JOB_IDS[@]}"
fi
echo "Cancelled active Qwen service jobs: ${ACTIVE_SERVICE_JOB_IDS[*]:-none}"
