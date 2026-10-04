#!/bin/bash
#SBATCH --job-name=prepare_eval_json  # 作业名称（与你的-J参数一致）
#SBATCH --output=SLURM_Logs/%j.out                    # 标准输出日志路径（%j会被替换为作业ID）
#SBATCH --error=SLURM_Logs/%j.err                     # 错误输出日志路径
#SBATCH --partition=a100
#SBATCH --nodes=1                                     # 申请1个计算节点
#SBATCH --ntasks-per-node=1                           # 每节点运行1个任务（推荐显式声明）
#SBATCH --cpus-per-task=4                             # 建议为每个任务分配CPU核心（根据需求调整）
#SBATCH --mem-per-cpu=8GB


source /home/sxzhuangch/miniconda3/etc/profile.d/conda.sh    # 替换为你的conda安装路径
conda activate deepseek-ocr

# pride and prejudice
# python3 build_dataset_from_ebook/build_json_from_txt.py \
#     --input build_dataset_from_ebook/src_book/pride-and-prejudice.txt \
#     --output build_dataset_from_ebook/src_book/pride_and_prejudice_dataset.json \
#     --book_name pride_and_prejudice \
#     --blocks 4

# Laws
# python3 -m build_dataset_from_ebook.build_json_from_txt_laws \
#     --input build_dataset_from_ebook/src_book/Laws_PostconvictionRemedies.txt \
#     --output build_dataset_from_ebook/src_book/Laws_PostconvictionRemedies_dataset.json \
#     --book_name laws \
#     --chunk_num 5 \
#     --blocks 3


# Economics
# python3 -m build_dataset_from_ebook.build_json_from_txt_economics \
#     --input build_dataset_from_ebook/src_book/wealth_of_nations.txt \
#     --output build_dataset_from_ebook/src_book/wealth_of_nations_dataset.json \
#     --book_name wealth_of_nations \
#     --blocks 5

# Medicine
# python3 -m build_dataset_from_ebook.build_json_from_txt_medicine \
#     --input build_dataset_from_ebook/src_book/medicine.txt \
#     --output build_dataset_from_ebook/src_book/medicine_dataset.json \
#     --book_name medicine \
#     --chunk_num 7 \
#     --blocks 4

# Newspaper
python3 -m build_dataset_from_ebook.build_json_from_txt_newspaper \
    --input build_dataset_from_ebook/src_book/newspaper.txt \
    --output build_dataset_from_ebook/src_book/newspaper_dataset.json \
    --book_name newspaper \
    --blocks 2

# Letters
# python3 -m build_dataset_from_ebook.build_json_from_txt_letters \
#     --input build_dataset_from_ebook/src_book/letters.txt \
#     --output build_dataset_from_ebook/src_book/letters_dataset.json \
#     --book_name letters \
#     --blocks 3