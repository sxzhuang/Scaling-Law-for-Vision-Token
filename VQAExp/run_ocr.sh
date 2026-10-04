#!/bin/bash
#SBATCH --job-name=ocr_retrieval
#SBATCH --output=SLURM_Logs/%j.out
#SBATCH --error=SLURM_Logs/%j.err
#SBATCH --partition=a100,a30
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=6
#SBATCH --mem-per-cpu=8GB

source /home/sxzhuangch/miniconda3/etc/profile.d/conda.sh
conda activate deepseek-ocr

LAYOUT_NAME="${LAYOUT_NAME:-28_6_0}"
RESOLUTION="${RESOLUTION:-640}"
INPUT_PATH="${INPUT_PATH:-VQAExp/data/Retrieval/${LAYOUT_NAME}/images_${RESOLUTION}}"
OUTPUT_PATH="${OUTPUT_PATH:-VQAExp/ocr_predictions/Qwen/Qwen2.5-VL-7B-Instruct/OCR/${LAYOUT_NAME}/res_${RESOLUTION}}"
METADATA_JSON="${METADATA_JSON:-VQAExp/data/Retrieval/retrieval_cropped_0_25k_600.json}"
URL="${URL:-http://localhost:50016/v1}"
MODEL="${MODEL:-Qwen/Qwen2.5-VL-7B-Instruct}"
MAX_TOKENS="${MAX_TOKENS:-8192}"
MAX_TOKENS_RATIO="${MAX_TOKENS_RATIO:-1.1}"
MAX_TOKENS_MARGIN="${MAX_TOKENS_MARGIN:-32}"
MIN_MAX_TOKENS="${MIN_MAX_TOKENS:-64}"
IMAGE_URL_MODE="${IMAGE_URL_MODE:-file}"
CONCURRENCY="${CONCURRENCY:-8}"
PREVIOUS_JOB_ID="${PREVIOUS_JOB_ID:-}"

if [[ -n "$PREVIOUS_JOB_ID" ]]; then
    previous_state=$(sacct -j "$PREVIOUS_JOB_ID" --format=JobIDRaw,State,ExitCode -n -P | awk -F'|' '$1 == job_id {print $2 " " $3; exit}' job_id="$PREVIOUS_JOB_ID" || true)
    if [[ -n "$previous_state" && "$previous_state" != COMPLETED* ]]; then
        echo "Previous job may have failed: job_id=$PREVIOUS_JOB_ID state=$previous_state"
    fi
fi

echo "LAYOUT_NAME=$LAYOUT_NAME"
echo "RESOLUTION=$RESOLUTION"
echo "INPUT_PATH=$INPUT_PATH"
echo "OUTPUT_PATH=$OUTPUT_PATH"
echo "METADATA_JSON=$METADATA_JSON"
echo "URL=$URL"

python VQAExp/run_ocr.py \
    --input_path "$INPUT_PATH" \
    --output_path "$OUTPUT_PATH" \
    --url "$URL" \
    --model "$MODEL" \
    --max_tokens "$MAX_TOKENS" \
    --metadata_json "$METADATA_JSON" \
    --max_tokens_ratio "$MAX_TOKENS_RATIO" \
    --max_tokens_margin "$MAX_TOKENS_MARGIN" \
    --min_max_tokens "$MIN_MAX_TOKENS" \
    --image_url_mode "$IMAGE_URL_MODE" \
    --concurrency "$CONCURRENCY"
