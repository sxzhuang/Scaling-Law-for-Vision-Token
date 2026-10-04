#!/bin/bash
#SBATCH --job-name=generate  # 作业名称（与你的-J参数一致）
#SBATCH --output=SLURM_Logs/%j.out                    # 标准输出日志路径（%j会被替换为作业ID）
#SBATCH --error=SLURM_Logs/%j.err                     # 错误输出日志路径
#SBATCH --partition=a100
#SBATCH --nodes=1                                     # 申请1个计算节点
#SBATCH --ntasks-per-node=1                           # 每节点运行1个任务（推荐显式声明）
#SBATCH --cpus-per-task=4                             # 建议为每个任务分配CPU核心（根据需求调整）
#SBATCH --mem-per-cpu=48GB


source /home/sxzhuangch/miniconda3/etc/profile.d/conda.sh    # 替换为你的conda安装路径
conda activate deepseek-ocr

python3 build_dataset_from_ebook/build_image_same_content_diff_shape.py \
    --dataset build_dataset_from_ebook/src_book/pride_and_prejudice_dataset.json \
    --output_dir build_dataset_from_ebook/pride_dataset_same_content_diff_shape/ \
    --font_size 28 \
    --padding 20 \
    --dpi 300 \
    --widths 512 640 728 1024 \
    --metadata_name pride_prejudice_same_content_diff_shape_gt.json