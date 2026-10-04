#!/bin/bash
#SBATCH --job-name=prepare_eval_json  # 作业名称（与你的-J参数一致）
#SBATCH --output=SLURM_Logs/%j.out                    # 标准输出日志路径（%j会被替换为作业ID）
#SBATCH --error=SLURM_Logs/%j.err                     # 错误输出日志路径
#SBATCH --partition=a100,a30
#SBATCH --nodes=1                                     # 申请1个计算节点
#SBATCH --ntasks-per-node=1                           # 每节点运行1个任务（推荐显式声明）
#SBATCH --cpus-per-task=6                             # 建议为每个任务分配CPU核心（根据需求调整）
#SBATCH --mem-per-cpu=8GB


source /home/sxzhuangch/miniconda3/etc/profile.d/conda.sh    # 替换为你的conda安装路径
conda activate deepseek-ocr

set -euo pipefail

if [ "$#" -ne 4 ]; then
    echo "Usage: bash appendix_exp/run_prepare_eval_json.sh <images_dir> <metadata_path> <predictions_dir> <output_path>"
    echo "Example: bash appendix_exp/run_prepare_eval_json.sh build_dataset_from_ebook/pride_dataset_square_density_experiment/28_6_0/images build_dataset_from_ebook/pride_dataset_square/pride_square_gt.json ocr_predictions/pride_dataset_square_density_experiment/28_6_0/1280 eval_results/pride_square_density_experiment/28_6_0/1280/prepare_json_for_eval.json"
    exit 1
fi

IMAGES_DIR="$1"
METADATA_PATH="$2"
PREDICTIONS_DIR="$3"
OUTPUT_PATH="$4"

if [ -n "${SLURM_SUBMIT_DIR:-}" ]; then
    REPO_ROOT="${SLURM_SUBMIT_DIR}"
else
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
fi
cd "${REPO_ROOT}"

if [ ! -f "build_dataset_from_ebook/build_eval_json_shifted.py" ]; then
    echo "ERROR: build_dataset_from_ebook/build_eval_json_shifted.py not found under ${REPO_ROOT}"
    echo "Please submit this job from the repository root."
    exit 1
fi

python3 build_dataset_from_ebook/build_eval_json_shifted.py \
    --images_dir "${IMAGES_DIR}" \
    --metadata_path "${METADATA_PATH}" \
    --predictions_dir "${PREDICTIONS_DIR}" \
    --output_path "${OUTPUT_PATH}"

#   sbatch appendix_exp/run_prepare_eval_json.sh \
#     appendix_exp/jianai_dataset_square/20_0_-1/images \
#     appendix_exp/jianai_dataset_square/20_0_-1/jianai_square_gt.json \
#     appendix_exp/ocr_predictions/jianai_dataset_square/20_0_-1/512 \
#     appendix_exp/eval_results/jianai_dataset_square/20_0_-1/512/prepare_json_for_eval.json
