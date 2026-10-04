#!/bin/bash
#SBATCH --job-name=test
#SBATCH --output=/slurm/shuxin_zhuang/tests1/SLURM_Logs/%j.out
#SBATCH --error=/slurm/shuxin_zhuang/tests1/SLURM_Logs/%j.err
#SBATCH --chdir=/slurm/shuxin_zhuang/tests1
#SBATCH --partition=a100,a30
#SBATCH --time=00:15:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=5
#SBATCH --mem-per-cpu=8GB

set -euo pipefail

cd /slurm/shuxin_zhuang/workspace/Scaling-Law-for-Vision-Token
source /home/sxzhuangch/miniconda3/etc/profile.d/conda.sh
conda activate deepseek-ocr

MERGE_MODE="${MERGE_MODE:-strict}"
MERGE_ARGS=()
if [[ "$MERGE_MODE" == "successful" ]]; then
    MERGE_ARGS+=(--merge_successful_only)
elif [[ "$MERGE_MODE" == "all" ]]; then
    MERGE_ARGS+=(--merge_all)
elif [[ "$MERGE_MODE" != "strict" ]]; then
    echo "Unknown MERGE_MODE: $MERGE_MODE" >&2
    exit 2
fi

python aaai_resub/qwen_instability_intervention/recover_negative_truncated.py --mode merge "${MERGE_ARGS[@]}"
