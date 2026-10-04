#!/bin/bash
#SBATCH --job-name=test  # 作业名称（与你的-J参数一致）
#SBATCH --output=/slurm/shuxin_zhuang/tests1/SLURM_Logs/%j.out                    # 标准输出日志路径（%j会被替换为作业ID）
#SBATCH --error=/slurm/shuxin_zhuang/tests1/SLURM_Logs/%j.err                     # 标准错误日志路径（%j会被替换为作业ID）
#SBATCH --chdir=/slurm/shuxin_zhuang/tests1
#SBATCH --partition=a100,a30
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --nodes=1                                     # 申请1个节点
#SBATCH --ntasks-per-node=1                           # 每节点运行1个任务（推荐显式声明）
#SBATCH --cpus-per-task=8                             # 每个任务使用8个CPU核心
#SBATCH --mem-per-cpu=8GB

set -euo pipefail

source /home/sxzhuangch/miniconda3/etc/profile.d/conda.sh
conda activate deepseek-ocr

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

cd /slurm/shuxin_zhuang/workspace/Scaling-Law-for-Vision-Token
for resolution in 512 640 1024 1280; do
    python -m run_dpsk_ocr_eval_batch \
        --base_size "${resolution}" \
        --image_size "${resolution}" \
        --input_path "aaai_resub/rendered_datasets/medicine/${resolution}/16_60_-1/images" \
        --output_path "aaai_resub/ocr_predictions/medicine/${resolution}/16_60_-1/" \
        --batch_size 300
done
