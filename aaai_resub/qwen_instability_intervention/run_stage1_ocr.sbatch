#!/bin/bash
#SBATCH --job-name=test
#SBATCH --output=/slurm/shuxin_zhuang/tests1/SLURM_Logs/%A_%a.out
#SBATCH --error=/slurm/shuxin_zhuang/tests1/SLURM_Logs/%A_%a.err
#SBATCH --chdir=/slurm/shuxin_zhuang/tests1
#SBATCH --partition=a100,a30
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=5
#SBATCH --mem-per-cpu=8GB
#SBATCH --array=0-3%1

set -euo pipefail

cd /slurm/shuxin_zhuang/workspace/Scaling-Law-for-Vision-Token
source /home/sxzhuangch/miniconda3/etc/profile.d/conda.sh
conda activate deepseek-ocr

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

TASK_GROUPS=(504_to_952 644_to_1092 1036_to_1624 1288_to_2100)
GROUP=${TASK_GROUPS[$SLURM_ARRAY_TASK_ID]}
STAGE_ROOT="${STAGE_ROOT:-aaai_resub/qwen_instability_intervention/stage1}"
INPUT_DIR="${STAGE_ROOT}/data/${GROUP}/images"
METADATA_JSON="${STAGE_ROOT}/data/${GROUP}/metadata.json"
OUTPUT_DIR="${STAGE_ROOT}/ocr_predictions/${GROUP}/md_res"
URL="${QWEN_URL:-http://localhost:50117/v1}"

echo "group=$GROUP endpoint=$URL input=$INPUT_DIR output=$OUTPUT_DIR"
curl -fsS --max-time 10 "${URL}/models" >/dev/null
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
