#!/bin/bash
#SBATCH --job-name=test
#SBATCH --output=/slurm/shuxin_zhuang/tests1/SLURM_Logs/%A_%a.out
#SBATCH --error=/slurm/shuxin_zhuang/tests1/SLURM_Logs/%A_%a.err
#SBATCH --chdir=/slurm/shuxin_zhuang/tests1
#SBATCH --partition=a100,a30
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=4GB
#SBATCH --array=0-3%4

set -euo pipefail

cd /slurm/shuxin_zhuang/workspace/Scaling-Law-for-Vision-Token
source /home/sxzhuangch/miniconda3/etc/profile.d/conda.sh
conda activate deepseek-ocr

TASK_GROUPS=(504_to_952 644_to_1092 1036_to_1624 1288_to_2100)
TARGETS=(952 1092 1624 2100)
GROUP=${TASK_GROUPS[$SLURM_ARRAY_TASK_ID]}
TARGET=${TARGETS[$SLURM_ARRAY_TASK_ID]}
STAGE_ROOT="${STAGE_ROOT:-aaai_resub/qwen_instability_intervention/stage1}"
GROUP_ROOT="${STAGE_ROOT}/data/${GROUP}"
INPUT_DIR="${GROUP_ROOT}/raw_images"
OUTPUT_DIR="${GROUP_ROOT}/images"

python utils/image_resize.py --input_path "$INPUT_DIR" --output_path "$OUTPUT_DIR" --resolution "$TARGET"

SOURCE_COUNT=$(find "$INPUT_DIR" -maxdepth 1 -type f -name '*.jpg' | wc -l)
OUTPUT_COUNT=$(find "$OUTPUT_DIR" -maxdepth 1 -type f -name '*.jpg' | wc -l)
if [[ "$SOURCE_COUNT" -ne "$OUTPUT_COUNT" ]]; then
    echo "Resize count mismatch for $GROUP: source=$SOURCE_COUNT output=$OUTPUT_COUNT" >&2
    exit 1
fi
echo "resize_complete group=$GROUP target=$TARGET images=$OUTPUT_COUNT"
