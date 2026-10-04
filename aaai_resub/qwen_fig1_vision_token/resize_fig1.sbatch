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
#SBATCH --array=0-23%24

set -euo pipefail

cd /slurm/shuxin_zhuang/workspace/Scaling-Law-for-Vision-Token
source /home/sxzhuangch/miniconda3/etc/profile.d/conda.sh
conda activate deepseek-ocr

GENRES=(novels laws economics medicine newspapers letters)
DATASETS=(pride_dataset_square laws_dataset_square economics_dataset_square medicine_dataset_square newspaper_dataset_square letters_dataset_square)
SIZES=(504 644 1036 1288)

GENRE_INDEX=$((SLURM_ARRAY_TASK_ID / ${#SIZES[@]}))
SIZE_INDEX=$((SLURM_ARRAY_TASK_ID % ${#SIZES[@]}))
GENRE=${GENRES[$GENRE_INDEX]}
DATASET=${DATASETS[$GENRE_INDEX]}
SIZE=${SIZES[$SIZE_INDEX]}
INPUT_DIR="build_dataset_from_ebook/${DATASET}/images"
OUTPUT_DIR="aaai_resub/qwen_fig1_vision_token/images/${GENRE}/res_${SIZE}"

echo "genre=$GENRE source=$INPUT_DIR target=${SIZE}x${SIZE} output=$OUTPUT_DIR"
python utils/image_resize.py --input_path "$INPUT_DIR" --output_path "$OUTPUT_DIR" --resolution "$SIZE"

SOURCE_COUNT=$(find "$INPUT_DIR" -maxdepth 1 -type f -name '*.jpg' | wc -l)
OUTPUT_COUNT=$(find "$OUTPUT_DIR" -maxdepth 1 -type f -name '*.jpg' | wc -l)
if [[ "$SOURCE_COUNT" -ne "$OUTPUT_COUNT" ]]; then
    echo "Resize count mismatch: source=$SOURCE_COUNT output=$OUTPUT_COUNT" >&2
    exit 1
fi
echo "resize_complete genre=$GENRE size=$SIZE images=$OUTPUT_COUNT"
