#!/bin/bash
#SBATCH --job-name=resize_retrieval
#SBATCH --output=SLURM_Logs/%A_%a.out
#SBATCH --error=SLURM_Logs/%A_%a.err
#SBATCH --partition=a100,a30
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=4GB
#SBATCH --array=0-143%24

set -euo pipefail

cd /slurm/shuxin_zhuang/workspace/Scaling-Law-for-Vision-Token

source /home/sxzhuangch/miniconda3/etc/profile.d/conda.sh
conda activate deepseek-ocr

RETRIEVAL_ROOT="VQAExp/data/Retrieval"
FONT_SIZES=(20 28 36)
LINE_SPACINGS=(0 6 24 42)
LETTER_SPACINGS=(-1 0 7)
RESOLUTIONS=(512 640 1024 1280)

NUM_LINE=${#LINE_SPACINGS[@]}
NUM_LETTER=${#LETTER_SPACINGS[@]}
NUM_RESOLUTION=${#RESOLUTIONS[@]}
COMBO_INDEX=${SLURM_ARRAY_TASK_ID}

LAYOUT_INDEX=$((COMBO_INDEX / NUM_RESOLUTION))
RESOLUTION_INDEX=$((COMBO_INDEX % NUM_RESOLUTION))

FONT_INDEX=$((LAYOUT_INDEX / (NUM_LINE * NUM_LETTER)))
REMAINDER=$((LAYOUT_INDEX % (NUM_LINE * NUM_LETTER)))
LINE_INDEX=$((REMAINDER / NUM_LETTER))
LETTER_INDEX=$((REMAINDER % NUM_LETTER))

FONT_SIZE=${FONT_SIZES[$FONT_INDEX]}
LINE_SPACING=${LINE_SPACINGS[$LINE_INDEX]}
LETTER_SPACING=${LETTER_SPACINGS[$LETTER_INDEX]}
RESOLUTION=${RESOLUTIONS[$RESOLUTION_INDEX]}

LAYOUT_NAME="${FONT_SIZE}_${LINE_SPACING}_${LETTER_SPACING}"
INPUT_DIR="$RETRIEVAL_ROOT/$LAYOUT_NAME/images"
OUTPUT_DIR="$RETRIEVAL_ROOT/$LAYOUT_NAME/images_${RESOLUTION}"

if [[ ! -d "$INPUT_DIR" ]]; then
    echo "Skip missing input: $INPUT_DIR"
    exit 0
fi

if [[ -d "$OUTPUT_DIR" ]]; then
    echo "Skip existing: $OUTPUT_DIR"
    exit 0
fi

echo "Resize $LAYOUT_NAME to ${RESOLUTION}x${RESOLUTION}"
python3 utils/image_resize.py \
    --input_path "$INPUT_DIR" \
    --output_path "$OUTPUT_DIR" \
    --resolution "$RESOLUTION"
