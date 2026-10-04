#!/bin/bash
#SBATCH --job-name=test
#SBATCH --output=/slurm/shuxin_zhuang/tests1/SLURM_Logs/%A_%a.out
#SBATCH --error=/slurm/shuxin_zhuang/tests1/SLURM_Logs/%A_%a.err
#SBATCH --chdir=/slurm/shuxin_zhuang/tests1
#SBATCH --partition=a100,a30
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=8GB
#SBATCH --array=0-6%7

set -euo pipefail

cd /slurm/shuxin_zhuang/tests1
source /home/sxzhuangch/miniconda3/etc/profile.d/conda.sh
conda activate deepseek-ocr

TARGETS=(672 868 1120 1400 1764 1792 2212)
TARGET=${TARGETS[$SLURM_ARRAY_TASK_ID]}
ROOT="aaai_resub/qwen_instability_intervention/calibrated_budget_intervention"
INPUT_DIR="${ROOT}/targets/target_${TARGET}/raw_images"
OUTPUT_DIR="${ROOT}/targets/target_${TARGET}/images"

python utils/image_resize.py --input_path "$INPUT_DIR" --output_path "$OUTPUT_DIR" --resolution "$TARGET"

INPUT_COUNT=$(find "$INPUT_DIR" -maxdepth 1 -type f -name '*.jpg' | wc -l)
OUTPUT_COUNT=$(find "$OUTPUT_DIR" -maxdepth 1 -type f -name '*.jpg' | wc -l)
if [[ "$INPUT_COUNT" -ne "$OUTPUT_COUNT" ]]; then
    echo "Resize count mismatch for target ${TARGET}: input=${INPUT_COUNT} output=${OUTPUT_COUNT}" >&2
    exit 1
fi
echo "resize_complete target=${TARGET} images=${OUTPUT_COUNT}"
