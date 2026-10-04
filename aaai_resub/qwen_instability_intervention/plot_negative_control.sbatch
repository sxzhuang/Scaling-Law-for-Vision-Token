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

export MPLCONFIGDIR="${TMPDIR:-/tmp}/matplotlib-${SLURM_JOB_ID}"
python aaai_resub/qwen_instability_intervention/plot_stage2.py \
    --stage2_roots aaai_resub/qwen_instability_intervention/negative_control/stage2 \
    --output aaai_resub/qwen_instability_intervention/figures/qwen_ocr_negative_control_0.8R.jpg \
    --summary aaai_resub/qwen_instability_intervention/negative_control/evaluation_summary.json
