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
#SBATCH --array=0-3

set -euo pipefail

cd /slurm/shuxin_zhuang/workspace/Scaling-Law-for-Vision-Token
source /home/sxzhuangch/miniconda3/etc/profile.d/conda.sh
conda activate deepseek-ocr

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

TASK_GROUPS=(504_to_952 644_to_1092 1036_to_1624 1288_to_2100)
TARGET_RESOLUTIONS=(952 1092 1624 2100)
GROUP=${TASK_GROUPS[$SLURM_ARRAY_TASK_ID]}
TARGET_RESOLUTION=${TARGET_RESOLUTIONS[$SLURM_ARRAY_TASK_ID]}
STAGE2_ROOT="${STAGE2_ROOT:-aaai_resub/qwen_instability_intervention/stage2}"

python VQAExp/cal_ocr_edit_distance.py \
    --gt_json "${STAGE2_ROOT}/metadata/${GROUP}.json" \
    --predictions_root "${STAGE2_ROOT}/predictions" \
    --output_root "${STAGE2_ROOT}/evaluation" \
    --layouts "$GROUP" \
    --resolutions "$TARGET_RESOLUTION"
