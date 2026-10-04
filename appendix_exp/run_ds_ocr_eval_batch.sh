#!/bin/bash
#SBATCH --job-name=ds_pred_batch  # 作业名称（与你的-J参数一致）
#SBATCH --output=SLURM_Logs/%j.out                    # 标准输出日志路径（%j会被替换为作业ID）
#SBATCH --error=SLURM_Logs/%j.err                     # 错误输出日志路径
#SBATCH --partition=a100
#SBATCH --gres=gpu:1                                  # 每节点申请1块GPU
#SBATCH --nodes=1                                     # 申请1个计算节点
#SBATCH --ntasks-per-node=1                           # 每节点运行1个任务（推荐显式声明）
#SBATCH --cpus-per-task=8                             # 建议为每个任务分配CPU核心（根据需求调整）
#SBATCH --mem-per-cpu=8GB


source /home/sxzhuangch/miniconda3/etc/profile.d/conda.sh    # 替换为你的conda安装路径
conda activate deepseek-ocr

export HF_HOME=/home/sxzhuangch/.cache/huggingface
export HUGGINGFACE_HUB_CACHE=/home/sxzhuangch/.cache/huggingface/hub
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

BASE_SIZE="${BASE_SIZE:-640}"
IMAGE_SIZE="${IMAGE_SIZE:-640}"
INPUT_PATH="${INPUT_PATH:-build_dataset_from_ebook/pride_dataset_square_random_phase/images/}"
OUTPUT_PATH="${OUTPUT_PATH:-ocr_predictions/pride_dataset_square_random_phase/${BASE_SIZE}/}"
BATCH_SIZE="${BATCH_SIZE:-200}"

echo "BASE_SIZE=${BASE_SIZE}"
echo "IMAGE_SIZE=${IMAGE_SIZE}"
echo "INPUT_PATH=${INPUT_PATH}"
echo "OUTPUT_PATH=${OUTPUT_PATH}"
echo "BATCH_SIZE=${BATCH_SIZE}"

python3 -m run_dpsk_ocr_eval_batch \
    --base_size "${BASE_SIZE}" \
    --image_size "${IMAGE_SIZE}" \
    --input_path "${INPUT_PATH}" \
    --output_path "${OUTPUT_PATH}" \
    --batch_size "${BATCH_SIZE}"

# Example:
# sbatch --export=ALL,BASE_SIZE=1024,IMAGE_SIZE=1024,INPUT_PATH=appendix_exp/jianai_dataset_square/20_0_-1/images/,OUTPUT_PATH=ocr_predictions/jianai_dataset_square/20_0_-1/1024/ appendix_exp/run_ds_ocr_eval_batch.sh
