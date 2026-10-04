#!/bin/bash
#SBATCH --job-name=generate  # 作业名称（与你的-J参数一致）
#SBATCH --output=SLURM_Logs/%j.out                    # 标准输出日志路径（%j会被替换为作业ID）
#SBATCH --error=SLURM_Logs/%j.err                     # 错误输出日志路径
#SBATCH --partition=a100
#SBATCH --nodes=1                                     # 申请1个计算节点
#SBATCH --ntasks-per-node=1                           # 每节点运行1个任务（推荐显式声明）
#SBATCH --cpus-per-task=4                             # 建议为每个任务分配CPU核心（根据需求调整）
#SBATCH --mem-per-cpu=8GB


source /home/sxzhuangch/miniconda3/etc/profile.d/conda.sh    # 替换为你的conda安装路径
conda activate deepseek-ocr


# Pride-and-Prejudice
# python3 build_dataset_from_ebook/build_image_square_from_json.py \
#     --dataset build_dataset_from_ebook/src_book/pride_and_prejudice_dataset.json \
#     --output_dir build_dataset_from_ebook/pride_dataset_square/ \
#     --output_image_prefix pride_prejudice \
#     --single_count 100 \
#     --double_count 200 \
#     --triple_count 150 \
#     --quadruple_count 200 \
#     --quintuple_count 400 \
#     --ratio_min 0.9 \
#     --ratio_max 1.1 \
#     --metadata_name pride_square_gt.json


# Laws
# python3 build_dataset_from_ebook/build_image_square_from_json.py \
#     --dataset build_dataset_from_ebook/src_book/Laws_PostconvictionRemedies_dataset.json \
#     --output_dir build_dataset_from_ebook/laws_dataset_square/ \
#     --output_image_prefix laws \
#     --single_count 100 \
#     --double_count 150 \
#     --triple_count 150 \
#     --quadruple_count 200 \
#     --quintuple_count 200 \
#     --sextuple_count 200 \
#     --ratio_min 0.9 \
#     --ratio_max 1.1 \
#     --metadata_name laws_square_gt.json

# Economics
# python3 build_dataset_from_ebook/build_image_square_from_json.py \
#     --dataset build_dataset_from_ebook/src_book/wealth_of_nations_dataset.json \
#     --output_dir build_dataset_from_ebook/economics_dataset_square/ \
#     --output_image_prefix economics \
#     --single_count 50 \
#     --double_count 80 \
#     --triple_count 50 \
#     --quadruple_count 150 \
#     --quintuple_count 180 \
#     --septuple_count 200 \
#     --octuple_count 80 \
#     --nonuple_count 120 \
#     --decuple_count 250 \
#     --ratio_min 0.9 \
#     --ratio_max 1.1 \
#     --metadata_name economics_square_gt.json

# Medicine
# python3 build_dataset_from_ebook/build_image_square_from_json.py \
#     --dataset build_dataset_from_ebook/src_book/medicine_dataset.json \
#     --output_dir build_dataset_from_ebook/medicine_dataset_square/ \
#     --output_image_prefix medicine \
#     --single_count 100 \
#     --double_count 200 \
#     --triple_count 200 \
#     --quadruple_count 150 \
#     --quintuple_count 350 \
#     --ratio_min 0.9 \
#     --ratio_max 1.1 \
#     --metadata_name medicine_square_gt.json

# Newspaper
# python3 build_dataset_from_ebook/build_image_square_from_json.py \
#     --dataset build_dataset_from_ebook/src_book/newspaper_dataset.json \
#     --output_dir build_dataset_from_ebook/newspaper_dataset_square/ \
#     --output_image_prefix newspaper \
#     --single_count 100 \
#     --double_count 200 \
#     --triple_count 200 \
#     --quadruple_count 200 \
#     --quintuple_count 200 \
#     --sextuple_count 100 \
#     --ratio_min 0.9 \
#     --ratio_max 1.1 \
#     --metadata_name newspaper_square_gt.json

# Letters
# python3 build_dataset_from_ebook/build_image_square_from_json.py \
#     --dataset build_dataset_from_ebook/src_book/letters_dataset.json \
#     --output_dir build_dataset_from_ebook/letters_dataset_square/ \
#     --output_image_prefix letters \
#     --single_count 150 \
#     --double_count 200 \
#     --triple_count 200 \
#     --quadruple_count 250 \
#     --quintuple_count 100 \
#     --ratio_min 0.9 \
#     --ratio_max 1.1 \
#     --metadata_name letters_square_gt.json

# jianai
python3 -m build_dataset_from_ebook.build_image_square_from_json_languages \
    --dataset build_dataset_from_ebook/src_book/jianai.json \
    --output_dir build_dataset_from_ebook/jianai_dataset_square/ \
    --metadata_name jianai_square_gt.json \
    --output_image_prefix jianai \
    --font_size 28 \
    --font_path build_dataset_from_ebook/fonts/SourceHanSansSC-Regular.otf \
    --wrap_width 60 \
    --padding 20 \
    --dpi 300 \
    --seed 42 \
    --single_count 200 \
    --double_count 150 \
    --triple_count 50 \
    --quadruple_count 0