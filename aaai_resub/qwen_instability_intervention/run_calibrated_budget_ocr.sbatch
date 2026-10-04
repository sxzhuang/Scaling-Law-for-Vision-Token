#!/bin/bash
#SBATCH --job-name=test
#SBATCH --output=/slurm/shuxin_zhuang/tests1/SLURM_Logs/%A_%a.out
#SBATCH --error=/slurm/shuxin_zhuang/tests1/SLURM_Logs/%A_%a.err
#SBATCH --chdir=/slurm/shuxin_zhuang/tests1
#SBATCH --partition=a100,a30
#SBATCH --time=08:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=5
#SBATCH --mem-per-cpu=8GB
#SBATCH --array=0-1%2

set -euo pipefail

cd /slurm/shuxin_zhuang/tests1
source /home/sxzhuangch/miniconda3/etc/profile.d/conda.sh
conda activate deepseek-ocr

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

ROOT="aaai_resub/qwen_instability_intervention/calibrated_budget_intervention"
if [[ "$SLURM_ARRAY_TASK_ID" -eq 0 ]]; then
    URL="http://localhost:50120/v1"
    TARGETS=(2212 1792 1400)
else
    URL="http://localhost:50121/v1"
    TARGETS=(1764 1120 868 672)
fi

curl -fsS --max-time 10 "${URL}/models" >/dev/null
for TARGET in "${TARGETS[@]}"; do
    INPUT_DIR="${ROOT}/targets/target_${TARGET}/images"
    METADATA_JSON="${ROOT}/targets/target_${TARGET}/metadata.json"
    OUTPUT_DIR="${ROOT}/ocr_predictions/target_${TARGET}/md_res"
    echo "target=${TARGET} endpoint=${URL} input=${INPUT_DIR} output=${OUTPUT_DIR}"
    python VQAExp/run_ocr.py \
        --input_path "$INPUT_DIR" \
        --output_path "$OUTPUT_DIR" \
        --url "$URL" \
        --model Qwen/Qwen2.5-VL-7B-Instruct \
        --max_tokens 8192 \
        --metadata_json "$METADATA_JSON" \
        --metadata_id_field id \
        --metadata_text_field context \
        --max_tokens_ratio 1.2 \
        --max_tokens_margin 32 \
        --min_max_tokens 64 \
        --temperature 0 \
        --image_url_mode file \
        --concurrency 8

    INPUT_COUNT=$(find "$INPUT_DIR" -maxdepth 1 -type f -name '*.jpg' | wc -l)
    OUTPUT_COUNT=$(find "$OUTPUT_DIR" -maxdepth 1 -type f -name '*.md' | wc -l)
    if [[ "$INPUT_COUNT" -ne "$OUTPUT_COUNT" ]]; then
        echo "OCR count mismatch for target ${TARGET}: input=${INPUT_COUNT} output=${OUTPUT_COUNT}" >&2
        exit 1
    fi
    echo "ocr_complete target=${TARGET} images=${OUTPUT_COUNT}"
done
