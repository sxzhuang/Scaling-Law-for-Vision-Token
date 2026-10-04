#!/bin/bash
#SBATCH --job-name=ds_pred_batch  # 作业名称（与你的-J参数一致）
#SBATCH --output=SLURM_Logs/%j.out                    # 标准输出日志路径（%j会被替换为作业ID）
#SBATCH --error=SLURM_Logs/%j.err                     # 错误输出日志路径
#SBATCH --partition=a100,a30
#SBATCH --gres=gpu:1                                  # 每节点申请1块GPU
#SBATCH --nodes=1                                     # 申请1个计算节点
#SBATCH --ntasks-per-node=1                           # 每节点运行1个任务（推荐显式声明）
#SBATCH --cpus-per-task=8                             # 建议为每个任务分配CPU核心（根据需求调整）
#SBATCH --mem-per-cpu=8GB


source /home/sxzhuangch/miniconda3/etc/profile.d/conda.sh    # 替换为你的conda安装路径
conda activate deepseek-ocr

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

# BASE_SIZE=512
# IMAGE_SIZE=512

BASE_SIZE=640
IMAGE_SIZE=640

# BASE_SIZE=1024
# IMAGE_SIZE=1024

# BASE_SIZE=1152
# IMAGE_SIZE=1152
# 768, 896, 1152

# BASE_SIZE=1280
# IMAGE_SIZE=1280

python3 -m run_dpsk_ocr_eval_batch \
    --base_size ${BASE_SIZE} \
    --image_size ${IMAGE_SIZE} \
    --input_path build_dataset_from_ebook/pride_dataset_square_random_phase/images/ \
    --output_path ocr_predictions/pride_dataset_square_random_phase/${BASE_SIZE}/ \
    --batch_size 300

# python3 -m run_dpsk_ocr_eval_batch \
#     --base_size ${BASE_SIZE} \
#     --image_size ${IMAGE_SIZE} \
#     --input_path build_dataset_from_ebook/pride_dataset_square_density_experiment/36_42_7/images/ \
#     --output_path ocr_predictions/pride_dataset_square_density_experiment/36_42_7/${BASE_SIZE}/
