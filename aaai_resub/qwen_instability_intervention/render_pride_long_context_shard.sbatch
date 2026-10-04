#!/bin/bash
#SBATCH --job-name=test
#SBATCH --output=/slurm/shuxin_zhuang/tests1/SLURM_Logs/%j.out
#SBATCH --error=/slurm/shuxin_zhuang/tests1/SLURM_Logs/%j.err
#SBATCH --chdir=/slurm/shuxin_zhuang/tests1
#SBATCH --partition=a100,a30
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=8GB

set -euo pipefail

cd /slurm/shuxin_zhuang/workspace/Scaling-Law-for-Vision-Token
source /home/sxzhuangch/miniconda3/etc/profile.d/conda.sh
conda activate deepseek-ocr

: "${INPUT_JSON:?INPUT_JSON must point to one GT shard}"

python VQAExp/render_retrieval_context_images.py \
    --input_json "$INPUT_JSON" \
    --output_root aaai_resub/qwen_instability_intervention/pride_long_context/rendered \
    --dataset_name mother_images \
    --font_size 28 \
    --line_spacing 6 \
    --letter_spacing 0 \
    --ratio_min 0.9 \
    --ratio_max 1.1
