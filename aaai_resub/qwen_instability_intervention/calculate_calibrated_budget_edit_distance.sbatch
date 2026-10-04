#!/bin/bash
#SBATCH --job-name=test
#SBATCH --output=/slurm/shuxin_zhuang/tests1/SLURM_Logs/%A_%a.out
#SBATCH --error=/slurm/shuxin_zhuang/tests1/SLURM_Logs/%A_%a.err
#SBATCH --chdir=/slurm/shuxin_zhuang/tests1
#SBATCH --partition=a100,a30
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=5
#SBATCH --mem-per-cpu=8GB
#SBATCH --array=0-7%8

set -euo pipefail

cd /slurm/shuxin_zhuang/tests1
source /home/sxzhuangch/miniconda3/etc/profile.d/conda.sh
conda activate deepseek-ocr

TASK_GROUP_NAMES=(504_to_672 504_to_868 644_to_868 644_to_1120 1036_to_1400 1036_to_1792 1288_to_1764 1288_to_2212)
TARGET_RESOLUTIONS=(672 868 868 1120 1400 1792 1764 2212)
TASK_GROUP_NAME=${TASK_GROUP_NAMES[$SLURM_ARRAY_TASK_ID]}
TARGET_RESOLUTION=${TARGET_RESOLUTIONS[$SLURM_ARRAY_TASK_ID]}
ROOT="aaai_resub/qwen_instability_intervention/calibrated_budget_intervention/evaluation_stage"

python VQAExp/cal_ocr_edit_distance.py \
    --gt_json "${ROOT}/metadata/${TASK_GROUP_NAME}.json" \
    --predictions_root "${ROOT}/predictions" \
    --output_root "${ROOT}/evaluation" \
    --layouts "$TASK_GROUP_NAME" \
    --resolutions "$TARGET_RESOLUTION"

OUTPUT="${ROOT}/evaluation/${TASK_GROUP_NAME}/ocr_edit_distance_res_${TARGET_RESOLUTION}.jsonl"
EXPECTED=$(python -c "import json; print(len(json.load(open('${ROOT}/metadata/${TASK_GROUP_NAME}.json', encoding='utf-8'))))")
ACTUAL=$(wc -l < "$OUTPUT")
if [[ "$EXPECTED" -ne "$ACTUAL" ]]; then
    echo "Evaluation count mismatch for ${TASK_GROUP_NAME}: expected=${EXPECTED} actual=${ACTUAL}" >&2
    exit 1
fi
echo "evaluation_complete group=${TASK_GROUP_NAME} samples=${ACTUAL}"
