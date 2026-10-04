#!/bin/bash
#SBATCH --job-name=test
#SBATCH --output=/slurm/shuxin_zhuang/tests1/SLURM_Logs/%j.out
#SBATCH --error=/slurm/shuxin_zhuang/tests1/SLURM_Logs/%j.err
#SBATCH --chdir=/slurm/shuxin_zhuang/tests1
#SBATCH --partition=a100,a30
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=5
#SBATCH --mem-per-cpu=8GB

set -euo pipefail

cd /slurm/shuxin_zhuang/workspace/Scaling-Law-for-Vision-Token
source /home/sxzhuangch/miniconda3/etc/profile.d/conda.sh
conda activate deepseek-ocr

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

STAGE1_ROOT="${STAGE1_ROOT:-aaai_resub/qwen_instability_intervention/stage1}"
STAGE2_ROOT="${STAGE2_ROOT:-aaai_resub/qwen_instability_intervention/stage2}"

PREPARE_ARGS=()
if [[ "${ALLOW_NON_STOP:-0}" == "1" ]]; then
    PREPARE_ARGS+=(--allow_non_stop)
fi

python aaai_resub/qwen_instability_intervention/prepare_stage2.py --stage1_root "$STAGE1_ROOT" --output_root "$STAGE2_ROOT" "${PREPARE_ARGS[@]}"
