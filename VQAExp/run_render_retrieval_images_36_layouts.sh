#!/bin/bash
#SBATCH --job-name=render_retrieval_36
#SBATCH --output=SLURM_Logs/%A_%a.out
#SBATCH --error=SLURM_Logs/%A_%a.err
#SBATCH --partition=a100,a30
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=4GB
#SBATCH --array=0-35

cd /slurm/shuxin_zhuang/workspace/Scaling-Law-for-Vision-Token

source /home/sxzhuangch/miniconda3/etc/profile.d/conda.sh
conda activate deepseek-ocr

FONT_SIZES=(20 28 36)
LINE_SPACINGS=(0 6 24 42)
LETTER_SPACINGS=(-1 0 7)

NUM_LINE=${#LINE_SPACINGS[@]}
NUM_LETTER=${#LETTER_SPACINGS[@]}
COMBO_INDEX=${SLURM_ARRAY_TASK_ID}

FONT_INDEX=$((COMBO_INDEX / (NUM_LINE * NUM_LETTER)))
REMAINDER=$((COMBO_INDEX % (NUM_LINE * NUM_LETTER)))
LINE_INDEX=$((REMAINDER / NUM_LETTER))
LETTER_INDEX=$((REMAINDER % NUM_LETTER))

FONT_SIZE=${FONT_SIZES[$FONT_INDEX]}
LINE_SPACING=${LINE_SPACINGS[$LINE_INDEX]}
LETTER_SPACING=${LETTER_SPACINGS[$LETTER_INDEX]}
LAYOUT_NAME="${FONT_SIZE}_${LINE_SPACING}_${LETTER_SPACING}"

if [[ "$LAYOUT_NAME" == "28_6_0" ]]; then
    echo "Skip existing layout: $LAYOUT_NAME"
    exit 0
fi

echo "Render layout: $LAYOUT_NAME"

python3 VQAExp/render_retrieval_context_images.py \
    --input_json VQAExp/data/Retrieval/retrieval_cropped_0_25k_600.json \
    --output_root VQAExp/data \
    --dataset_name Retrieval \
    --font_size "$FONT_SIZE" \
    --line_spacing "$LINE_SPACING" \
    --letter_spacing "$LETTER_SPACING" \
    --padding 20 \
    --dpi 300
