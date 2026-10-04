#!/bin/bash
#SBATCH --job-name=test
#SBATCH --output=/slurm/shuxin_zhuang/tests1/SLURM_Logs/%A_%a.out
#SBATCH --error=/slurm/shuxin_zhuang/tests1/SLURM_Logs/%A_%a.err
#SBATCH --chdir=/slurm/shuxin_zhuang/tests1
#SBATCH --partition=a100,a30
#SBATCH --time=09:30:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=5
#SBATCH --mem-per-cpu=8GB
#SBATCH --array=0-1

set -euo pipefail

cd /slurm/shuxin_zhuang/tests1
source /home/sxzhuangch/miniconda3/etc/profile.d/conda.sh
conda activate deepseek-ocr

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

ROOT="aaai_resub/qwen_instability_intervention/pride_long_context"
MODEL="Qwen/Qwen2.5-VL-7B-Instruct"

run_ocr_group() {
    local input_dir=$1
    local metadata_json=$2
    local output_dir=$3
    local url=$4

    echo "endpoint=${url} input=${input_dir} output=${output_dir}"
    python VQAExp/run_ocr.py \
        --input_path "$input_dir" \
        --output_path "$output_dir" \
        --url "$url" \
        --model "$MODEL" \
        --timeout 1800 \
        --max_tokens 14000 \
        --metadata_json "$metadata_json" \
        --metadata_id_field id \
        --metadata_text_field context \
        --max_tokens_ratio 1.2 \
        --max_tokens_margin 64 \
        --min_max_tokens 64 \
        --temperature 0 \
        --image_url_mode file \
        --concurrency 4
}

if [[ "$SLURM_ARRAY_TASK_ID" == "0" ]]; then
    URL="http://localhost:50118/v1"
    curl -fsS --max-time 10 "${URL}/models" >/dev/null
    run_ocr_group \
        "$ROOT/resized/1036/images" \
        "$ROOT/data/pride_25k_34k_for_1036_gt.json" \
        "$ROOT/ocr_predictions/1036/md_res" \
        "$URL"
    run_ocr_group \
        "$ROOT/ocr_shards/1288/slot0/images" \
        "$ROOT/data/pride_25k_48k_for_1288_gt.json" \
        "$ROOT/ocr_predictions/1288/slot0/md_res" \
        "$URL"
else
    URL="http://localhost:50119/v1"
    curl -fsS --max-time 10 "${URL}/models" >/dev/null
    run_ocr_group \
        "$ROOT/ocr_shards/1288/slot1/images" \
        "$ROOT/data/pride_25k_48k_for_1288_gt.json" \
        "$ROOT/ocr_predictions/1288/slot1/md_res" \
        "$URL"
fi
