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

# Resolution parameter
RESOLUTION=1280

# Define parameter arrays for density experiment
FONT_SIZES=(20 28 36)
LINE_SPACINGS=(0 6 24 42)
LETTER_SPACINGS=(-1 0 7)

echo "Starting eval JSON preparation for density experiment folders with resolution: ${RESOLUTION}"

# Triple-nested loop to iterate over all combinations
for font_size in "${FONT_SIZES[@]}"; do
    for line_spacing in "${LINE_SPACINGS[@]}"; do
        for letter_spacing in "${LETTER_SPACINGS[@]}"; do
            # Construct dataset name from parameters
            DATASET_NAME="${font_size}_${line_spacing}_${letter_spacing}"
            
            # Build paths dynamically
            images_dir="build_dataset_from_ebook/pride_dataset_square_density_experiment/${DATASET_NAME}/images"
            metadata_path="build_dataset_from_ebook/pride_dataset_square/pride_square_gt.json"
            predictions_dir="ocr_predictions/pride_dataset_square_density_experiment/${DATASET_NAME}/${RESOLUTION}"
            output_path="eval_results/pride_square_density_experiment/${DATASET_NAME}/${RESOLUTION}/prepare_json_for_eval.json"
            
            echo "Processing: ${DATASET_NAME} @ resolution ${RESOLUTION}"
            
            # Build eval json for density images
            python3 build_dataset_from_ebook/build_eval_json_shifted.py \
                --images_dir "${images_dir}" \
                --metadata_path "${metadata_path}" \
                --predictions_dir "${predictions_dir}" \
                --output_path "${output_path}"
            
            echo "Completed: ${DATASET_NAME}"
        done
    done
done

echo "All eval JSON preparations completed for resolution ${RESOLUTION}"