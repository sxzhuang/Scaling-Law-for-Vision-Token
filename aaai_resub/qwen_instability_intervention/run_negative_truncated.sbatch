#!/bin/bash
#SBATCH --job-name=test
#SBATCH --output=/slurm/shuxin_zhuang/tests1/SLURM_Logs/%A_%a.out
#SBATCH --error=/slurm/shuxin_zhuang/tests1/SLURM_Logs/%A_%a.err
#SBATCH --chdir=/slurm/shuxin_zhuang/tests1
#SBATCH --partition=a100,a30
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=5
#SBATCH --mem-per-cpu=8GB
#SBATCH --array=0-1

set -euo pipefail

cd /slurm/shuxin_zhuang/workspace/Scaling-Law-for-Vision-Token
source /home/sxzhuangch/miniconda3/etc/profile.d/conda.sh
conda activate deepseek-ocr

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

if [[ "$SLURM_ARRAY_TASK_ID" -eq 0 ]]; then
    TASK_GROUPS=(1288_to_1652)
    URL="http://localhost:50118/v1"
else
    TASK_GROUPS=(644_to_840 1036_to_1260)
    URL="http://localhost:50119/v1"
fi

RERUN_ROOT="aaai_resub/qwen_instability_intervention/negative_control/rerun_truncated"
curl -fsS --max-time 10 "${URL}/models" >/dev/null
for GROUP in "${TASK_GROUPS[@]}"; do
    if [[ ! -d "${RERUN_ROOT}/${GROUP}/images" ]]; then
        continue
    fi
    python VQAExp/run_ocr.py \
        --input_path "${RERUN_ROOT}/${GROUP}/images" \
        --output_path "${RERUN_ROOT}/${GROUP}/md_res" \
        --url "$URL" \
        --model Qwen/Qwen2.5-VL-7B-Instruct \
        --max_tokens 12000 \
        --metadata_json "${RERUN_ROOT}/${GROUP}/metadata.json" \
        --metadata_id_field id \
        --metadata_text_field context \
        --max_tokens_ratio 6.0 \
        --max_tokens_margin 64 \
        --min_max_tokens 12000 \
        --temperature 0 \
        --image_url_mode file \
        --concurrency 4
done
