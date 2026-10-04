#!/bin/bash
#SBATCH --job-name=test  # 作业名称（与你的-J参数一致）
#SBATCH --output=/slurm/shuxin_zhuang/tests1/SLURM_Logs/%j.out                    # 标准输出日志路径（%j会被替换为作业ID）
#SBATCH --error=/slurm/shuxin_zhuang/tests1/SLURM_Logs/%j.err                     # 标准错误日志路径（%j会被替换为作业ID）
#SBATCH --chdir=/slurm/shuxin_zhuang/tests1
#SBATCH --partition=a100,a30
#SBATCH --time=04:00:00
#SBATCH --nodes=1                                     # 申请1个节点
#SBATCH --ntasks-per-node=1                           # 每节点运行1个任务（推荐显式声明）
#SBATCH --cpus-per-task=5                             # 每个任务使用5个CPU核心
#SBATCH --mem-per-cpu=8GB

set -euo pipefail

source /home/sxzhuangch/miniconda3/etc/profile.d/conda.sh
conda activate deepseek-ocr

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

cd /slurm/shuxin_zhuang/workspace/Scaling-Law-for-Vision-Token
export PYTHONPATH="/slurm/shuxin_zhuang/workspace/Scaling-Law-for-Vision-Token:${PYTHONPATH:-}"
for resolution in 512 640 1024 1280; do
    python build_dataset_from_ebook/build_image_square_font_line_letter.py \
        --metadata "aaai_resub/selected_samples/medicine/${resolution}/selected_metadata.json" \
        --output_root "aaai_resub/rendered_datasets/medicine/${resolution}" \
        --font_sizes 16 \
        --line_spacings 60 \
        --letter_spacings -1 \
        --padding 20 \
        --dpi 300 \
        --ratio_min 0.9 \
        --ratio_max 1.1 \
        --num_workers 5
done
