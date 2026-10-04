#!/bin/bash
#SBATCH --job-name=test
#SBATCH --output=/slurm/shuxin_zhuang/tests1/SLURM_Logs/%j.out
#SBATCH --error=/slurm/shuxin_zhuang/tests1/SLURM_Logs/%j.err
#SBATCH --chdir=/slurm/shuxin_zhuang/tests1
#SBATCH --partition=a100,a30
#SBATCH --time=00:15:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=5
#SBATCH --mem-per-cpu=8GB

set -euo pipefail

cd /slurm/shuxin_zhuang/workspace/Scaling-Law-for-Vision-Token
source /home/sxzhuangch/miniconda3/etc/profile.d/conda.sh
conda activate deepseek-ocr

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

STAGE_ROOT="aaai_resub/qwen_instability_intervention/stage1"
GROUP="504_to_952"
IMAGE_ID="retrieval_crop_000061"
RERUN_ROOT="${STAGE_ROOT}/ocr_predictions/${GROUP}/rerun_truncated"

mkdir -p "${RERUN_ROOT}/images" "${RERUN_ROOT}/md_res"
ln -sfn "$(realpath "${STAGE_ROOT}/data/${GROUP}/images/${IMAGE_ID}.jpg")" "${RERUN_ROOT}/images/${IMAGE_ID}.jpg"
rm -f "${RERUN_ROOT}/md_res/${IMAGE_ID}.md" "${RERUN_ROOT}/ocr_predictions_md_res.jsonl" "${RERUN_ROOT}/generation_records_md_res.csv"

python VQAExp/run_ocr.py \
    --input_path "${RERUN_ROOT}/images" \
    --output_path "${RERUN_ROOT}/md_res" \
    --url "http://localhost:50117/v1" \
    --model Qwen/Qwen2.5-VL-7B-Instruct \
    --max_tokens 8192 \
    --metadata_json "${STAGE_ROOT}/data/${GROUP}/metadata.json" \
    --metadata_id_field id \
    --metadata_text_field context \
    --max_tokens_ratio 3.0 \
    --max_tokens_margin 64 \
    --min_max_tokens 64 \
    --temperature 0 \
    --image_url_mode file \
    --concurrency 1
