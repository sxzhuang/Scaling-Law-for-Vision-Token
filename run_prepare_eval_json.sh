#!/bin/bash
#SBATCH --job-name=prepare_eval_json  # 作业名称（与你的-J参数一致）
#SBATCH --output=SLURM_Logs/%j.out                    # 标准输出日志路径（%j会被替换为作业ID）
#SBATCH --error=SLURM_Logs/%j.err                     # 错误输出日志路径
#SBATCH --partition=a100
#SBATCH --nodes=1                                     # 申请1个计算节点
#SBATCH --ntasks-per-node=1                           # 每节点运行1个任务（推荐显式声明）
#SBATCH --cpus-per-task=6                             # 建议为每个任务分配CPU核心（根据需求调整）
#SBATCH --mem-per-cpu=8GB


source /home/sxzhuangch/miniconda3/etc/profile.d/conda.sh    # 替换为你的conda安装路径
conda activate deepseek-ocr

# 640 resolution
# python3 build_dataset_from_ebook/build_eval_json.py \
#     --images_dir build_dataset_from_ebook/pride_dataset/images \
#     --metadata_path build_dataset_from_ebook/pride_dataset/pride_prejudice_blocks_gt.json \
#     --predictions_dir ocr_predictions/pride_dataset/640 \
#     --output_path eval_results/pride/640/prepare_json_for_eval.json

# 1024 resolution
# python3 build_dataset_from_ebook/build_eval_json.py \
#     --images_dir build_dataset_from_ebook/pride_dataset/images \
#     --metadata_path build_dataset_from_ebook/pride_dataset/pride_prejudice_blocks_gt.json \
#     --predictions_dir ocr_predictions/pride_dataset/1024 \
#     --output_path eval_results/pride/1024/prepare_json_for_eval.json

# 512 resolution
# python3 build_dataset_from_ebook/build_eval_json.py \
#     --images_dir build_dataset_from_ebook/pride_dataset/images \
#     --metadata_path build_dataset_from_ebook/pride_dataset/pride_prejudice_blocks_gt.json \
#     --predictions_dir ocr_predictions/pride_dataset/512 \
#     --output_path eval_results/pride/512/prepare_json_for_eval.json

# 1280 resolution
# python3 build_dataset_from_ebook/build_eval_json.py \
#     --images_dir build_dataset_from_ebook/pride_dataset/images \
#     --metadata_path build_dataset_from_ebook/pride_dataset/pride_prejudice_blocks_gt.json \
#     --predictions_dir ocr_predictions/pride_dataset/1280 \
    # --output_path eval_results/pride/1280/prepare_json_for_eval.json

# python3 build_dataset_from_ebook/build_eval_json.py \
#     --images_dir build_dataset_from_ebook/pride_dataset_square_cutoff_content/images \
#     --metadata_path build_dataset_from_ebook/pride_dataset_square_cutoff_content/pride_square_cutoff_gt.json \
#     --predictions_dir ocr_predictions/pride_dataset_square_cutoff_content/1024 \
#     --output_path eval_results/pride_square_cutoff_content/1024/prepare_json_for_eval.json

# build eval json for shifted images
python3 build_dataset_from_ebook/build_eval_json_shifted.py \
    --images_dir build_dataset_from_ebook/pride_dataset_square_random_phase/images \
    --metadata_path build_dataset_from_ebook/pride_dataset_square_random_phase/pride_square_gt.json \
    --predictions_dir ocr_predictions/pride_dataset_square_random_phase/640 \
    --output_path eval_results/pride_dataset_square_random_phase/640/prepare_json_for_eval.json

# build eval json for density images
# python3 build_dataset_from_ebook/build_eval_json_shifted.py \
#     --images_dir build_dataset_from_ebook/pride_square_density_experiment/28_42_7/images \
#     --metadata_path build_dataset_from_ebook/pride_dataset_square/pride_square_gt.json \
#     --predictions_dir ocr_predictions/pride_square_density_experiment/28_42_7/1280 \
#     --output_path eval_results/pride_square_density_experiment/28_42_7/1280/prepare_json_for_eval.json