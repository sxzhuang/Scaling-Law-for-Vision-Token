#!/bin/bash
#SBATCH --job-name=test  # 作业名称（与你的-J参数一致）
#SBATCH --output=/slurm/shuxin_zhuang/tests1/SLURM_Logs/%j.out                    # 标准输出日志路径（%j会被替换为作业ID）
#SBATCH --error=/slurm/shuxin_zhuang/tests1/SLURM_Logs/%j.err                     # 标准错误日志路径（%j会被替换为作业ID）
#SBATCH --chdir=/slurm/shuxin_zhuang/tests1
#SBATCH --partition=a100,a30
#SBATCH --time=00:15:00
#SBATCH --nodes=1                                     # 申请1个节点
#SBATCH --ntasks-per-node=1                           # 每节点运行1个任务（推荐显式声明）
#SBATCH --cpus-per-task=8                             # 建议为每个任务分配CPU核心
#SBATCH --mem-per-cpu=8GB

set -euo pipefail

source /home/sxzhuangch/miniconda3/etc/profile.d/conda.sh
conda activate deepseek-ocr

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

cd /slurm/shuxin_zhuang/workspace/Scaling-Law-for-Vision-Token
python aaai_resub/scripts/00_preflight_layout.py --overwrite
