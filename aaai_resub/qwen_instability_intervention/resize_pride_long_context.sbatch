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
#SBATCH --array=0-1%2

set -euo pipefail

cd /slurm/shuxin_zhuang/workspace/Scaling-Law-for-Vision-Token
source /home/sxzhuangch/miniconda3/etc/profile.d/conda.sh
conda activate deepseek-ocr

ROOT="aaai_resub/qwen_instability_intervention/pride_long_context"
MOTHER_DIR="${ROOT}/rendered/mother_images/28_6_0/images"

if [[ "$SLURM_ARRAY_TASK_ID" -eq 0 ]]; then
    RESOLUTION=1036
    INPUT_DIR="${ROOT}/resize_inputs/1036"
    mkdir -p "$INPUT_DIR"
    for old_link in "$INPUT_DIR"/*.jpg; do
        [[ -e "$old_link" || -L "$old_link" ]] && rm -f "$old_link"
    done
    for bin_start in 25000 26000 27000 28000 29000 30000 31000 32000 33000; do
        for image_path in "$MOTHER_DIR"/pride_long_${bin_start}_*.jpg; do
            [[ -f "$image_path" ]] || continue
            ln -s "$(realpath "$image_path")" "$INPUT_DIR/$(basename "$image_path")"
        done
    done
    EXPECTED=180
else
    RESOLUTION=1288
    INPUT_DIR="$MOTHER_DIR"
    EXPECTED=460
fi

OUTPUT_DIR="${ROOT}/resized/${RESOLUTION}/images"
INPUT_COUNT=$(find -L "$INPUT_DIR" -maxdepth 1 -type f -name '*.jpg' | wc -l)
if [[ "$INPUT_COUNT" -ne "$EXPECTED" ]]; then
    echo "Unexpected input count for ${RESOLUTION}: expected=${EXPECTED} actual=${INPUT_COUNT}" >&2
    exit 1
fi

python utils/image_resize.py --input_path "$INPUT_DIR" --output_path "$OUTPUT_DIR" --resolution "$RESOLUTION"

OUTPUT_COUNT=$(find "$OUTPUT_DIR" -maxdepth 1 -type f -name '*.jpg' | wc -l)
if [[ "$OUTPUT_COUNT" -ne "$EXPECTED" ]]; then
    echo "Resize count mismatch for ${RESOLUTION}: expected=${EXPECTED} output=${OUTPUT_COUNT}" >&2
    exit 1
fi
echo "resize_complete resolution=${RESOLUTION} images=${OUTPUT_COUNT}"
