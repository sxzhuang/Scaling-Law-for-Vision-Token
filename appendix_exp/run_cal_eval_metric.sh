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

set -euo pipefail

JSON_FOR_EVAL=""
OUTPUT_FILE=""

while [ "$#" -gt 0 ]; do
    case "$1" in
        --json_for_eval)
            JSON_FOR_EVAL="$2"
            shift 2
            ;;
        --output_file)
            OUTPUT_FILE="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: sbatch appendix_exp/run_cal_eval_metric.sh --json_for_eval <path> --output_file <path>"
            exit 0
            ;;
        *)
            echo "ERROR: unknown argument: $1"
            echo "Usage: sbatch appendix_exp/run_cal_eval_metric.sh --json_for_eval <path> --output_file <path>"
            exit 1
            ;;
    esac
done

if [ -z "${JSON_FOR_EVAL}" ] || [ -z "${OUTPUT_FILE}" ]; then
    echo "ERROR: both --json_for_eval and --output_file are required."
    echo "Usage: sbatch appendix_exp/run_cal_eval_metric.sh --json_for_eval <path> --output_file <path>"
    exit 1
fi

if [ -n "${SLURM_SUBMIT_DIR:-}" ]; then
    REPO_ROOT="${SLURM_SUBMIT_DIR}"
else
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
fi
cd "${REPO_ROOT}"

if [ ! -f "evaluation_dpsk_ocr_predictions.py" ]; then
    echo "ERROR: evaluation_dpsk_ocr_predictions.py not found under ${REPO_ROOT}"
    echo "Please submit this job from the repository root."
    exit 1
fi

python3 evaluation_dpsk_ocr_predictions.py \
    --json_for_eval "${JSON_FOR_EVAL}" \
    --output_file "${OUTPUT_FILE}"

#   sbatch appendix_exp/run_cal_eval_metric.sh \
#     --json_for_eval appendix_exp/eval_results/jianai_dataset_square/20_0_-1/512/prepare_json_for_eval.json \
#     --output_file appendix_exp/eval_results/jianai_dataset_square/20_0_-1/512/dpsk_eval_metric.json
