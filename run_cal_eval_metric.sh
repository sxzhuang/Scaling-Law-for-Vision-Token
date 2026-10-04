#!/bin/bash
#SBATCH --job-name=cal_metric  # 作业名称（与你的-J参数一致）
#SBATCH --output=SLURM_Logs/%j.out                    # 标准输出日志路径（%j会被替换为作业ID）
#SBATCH --error=SLURM_Logs/%j.err                     # 错误输出日志路径
#SBATCH --partition=a100
#SBATCH --nodes=1                                     # 申请1个计算节点
#SBATCH --ntasks-per-node=1                           # 每节点运行1个任务（推荐显式声明）
#SBATCH --cpus-per-task=6                             # 建议为每个任务分配CPU核心（根据需求调整）
#SBATCH --mem-per-cpu=8GB


source /home/sxzhuangch/miniconda3/etc/profile.d/conda.sh    # 替换为你的conda安装路径
conda activate deepseek-ocr

# RESOLUTION=multi_paddling_1024
RESOLUTION=640

# python3 evaluation_dpsk_ocr_predictions.py \
#     --json_for_eval eval_results/pride_square/${RESOLUTION}/prepare_json_for_eval.json \
#     --output_file eval_results/pride_square/${RESOLUTION}/dpsk_eval_metric.json

python3 evaluation_dpsk_ocr_predictions.py \
    --json_for_eval eval_results/pride_dataset_square_random_phase/${RESOLUTION}/prepare_json_for_eval.json \
    --output_file eval_results/pride_dataset_square_random_phase/${RESOLUTION}/dpsk_eval_metric.json