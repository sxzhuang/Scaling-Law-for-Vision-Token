#!/bin/bash
#SBATCH --job-name=vqa_retrieval
#SBATCH --output=SLURM_Logs/%j.out
#SBATCH --error=SLURM_Logs/%j.err
#SBATCH --partition=a100,a30
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=8GB

source /home/sxzhuangch/miniconda3/etc/profile.d/conda.sh
conda activate deepseek-ocr

LAYOUT_NAME="${LAYOUT_NAME:-28_6_0}"
RESOLUTION="${RESOLUTION:-640}"
INPUT_PATH="${INPUT_PATH:-VQAExp/data/Retrieval/${LAYOUT_NAME}/images_${RESOLUTION}}"
DATASET_JSON="${DATASET_JSON:-VQAExp/data/Retrieval/retrieval_cropped_0_25k_600.json}"
OUTPUT_PATH="${OUTPUT_PATH:-VQAExp/vqa_predictions/Qwen/Qwen2.5-VL-7B-Instruct/Retrieval/${LAYOUT_NAME}/res_${RESOLUTION}}"
URL="${URL:-http://localhost:50016/v1}"
MODEL="${MODEL:-Qwen/Qwen2.5-VL-7B-Instruct}"
MAX_TOKENS="${MAX_TOKENS:-128}"
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
echo "URL=$URL"

python VQAExp/run_vqa.py \
    --input_path "$INPUT_PATH" \
    --dataset_json "$DATASET_JSON" \
    --output_path "$OUTPUT_PATH" \
    --url "$URL" \
    --model "$MODEL" \
    --max_tokens "$MAX_TOKENS" \
    --image_url_mode "$IMAGE_URL_MODE" \
    --concurrency "$CONCURRENCY"
