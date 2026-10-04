#!/bin/bash
#SBATCH --job-name=ds_pred_batch  # 作业名称（与你的-J参数一致）
#SBATCH --output=SLURM_Logs/%j.out                    # 标准输出日志路径（%j会被替换为作业ID）
#SBATCH --error=SLURM_Logs/%j.err                     # 错误输出日志路径
#SBATCH --partition=a100,a30
#SBATCH --nodes=1                                     # 申请1个计算节点
#SBATCH --ntasks-per-node=1                           # 每节点运行1个任务（推荐显式声明）
#SBATCH --cpus-per-task=8                             # 建议为每个任务分配CPU核心（根据需求调整）
#SBATCH --mem-per-cpu=8GB

set -euo pipefail

source /home/sxzhuangch/miniconda3/etc/profile.d/conda.sh
conda activate deepseek-ocr

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

mkdir -p SLURM_Logs

FONT_SIZE="${FONT_SIZE:-20}"
LINE_SPACING="${LINE_SPACING:-0}"
LETTER_SPACING="${LETTER_SPACING:--1}"
FONT_PATH="${FONT_PATH:-build_dataset_from_ebook/fonts/SourceHanSansSC-Regular.otf}"

python appendix_exp/build_image_square_from_sampled_text_dataset.py \
    --font_size "${FONT_SIZE}" \
    --line_spacing "${LINE_SPACING}" \
    --letter_spacing "${LETTER_SPACING}" \
    --font_path "${FONT_PATH}"

# FONT_SIZE=28 LINE_SPACING=6 LETTER_SPACING=0 sbatch appendix_exp/run_build_image_square_from_sampled_text_dataset.sh