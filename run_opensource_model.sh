#!/bin/bash
#SBATCH --job-name=ds_pred_batch  # 作业名称（与你的-J参数一致）
#SBATCH --output=SLURM_Logs/%j.out                    # 标准输出日志路径（%j会被替换为作业ID）
#SBATCH --error=SLURM_Logs/%j.err                     # 错误输出日志路径
#SBATCH --partition=a100
#SBATCH --nodes=1                                     # 申请1个计算节点
#SBATCH --ntasks-per-node=1                           # 每节点运行1个任务（推荐显式声明）
#SBATCH --cpus-per-task=8                             # 建议为每个任务分配CPU核心（根据需求调整）
#SBATCH --mem-per-cpu=8GB


source /home/sxzhuangch/miniconda3/etc/profile.d/conda.sh    # 替换为你的conda安装路径
conda activate deepseek-ocr

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

model_id=Qwen/Qwen2.5-VL-7B-Instruct
max_tokens=6800
image_url_mode=file

# model_id=OpenGVLab/InternVL3_5-8B
# max_tokens=6800
# image_url_mode=file

# model_id=deepseek-ai/deepseek-vl-1.3b-chat
# max_tokens=6800
# image_url_mode=url


# model_id=zai-org/Glyph
# max_tokens=6800
# image_url_mode=file


# for DATASET in pride_dataset_square economics_dataset_square laws_dataset_square letters_dataset_square medicine_dataset_square newspaper_dataset_square; do
# for DATASET in letters_dataset_square medicine_dataset_square newspaper_dataset_square; do
for DATASET in pride_dataset_square; do
    echo "========================================"
    echo "Processing dataset: $DATASET"
    echo "========================================"
    
    python3 -m run_opensource_model \
        --input_path build_dataset_from_ebook/${DATASET}/images/ \
        --output_path ocr_predictions/${model_id}/${DATASET}/md_res_896/ \
        --url http://10.74.197.104:50016/v1 \
        --model $model_id \
        --max_tokens $max_tokens \
        --image_url_mode $image_url_mode \
        --concurrency 20
done

wait

echo "All datasets processed successfully."